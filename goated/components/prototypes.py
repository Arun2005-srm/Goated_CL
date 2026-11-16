"""BN-statistics prototypes (components 5, 6, 7) and data-free domain routing (component 8).

Statistics are per image: for each tracked BN layer, the per-channel mean and log-variance of
the BN *input* over H, W. Per-image statistics make prototypes comparable across batch sizes
(current batch vs. small replay batch)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def select_bn_layers(model, k):
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bns:
        raise ValueError("BN-prototype components need a model with BatchNorm2d layers")
    if k in (None, "all") or k >= len(bns):
        return bns
    idx = torch.linspace(0, len(bns) - 1, k).round().long().tolist()  # spread over depth
    return [bns[i] for i in sorted(set(idx))]


class BNStatRecorder:
    """Forward-pre-hooks that record per-image (mu, logvar) of BN inputs while `active`."""

    def __init__(self, model, k):
        self.layers = select_bn_layers(model, k)
        self.stats, self.active = [], False
        self.handles = [m.register_forward_pre_hook(self._hook) for m in self.layers]

    def _hook(self, module, inp):
        if self.active:
            x = inp[0].float()
            mu = x.mean((2, 3))
            var = x.var((2, 3), unbiased=False)
            self.stats.append((mu, torch.log(var + 1e-5)))

    def __enter__(self):
        self.stats, self.active = [], True
        return self

    def __exit__(self, *a):
        self.active = False

    def take(self):
        s, self.stats = self.stats, []
        return s

    def remove(self):
        for h in self.handles:
            h.remove()


def stack_obs(stats):
    """[(mu [N,C_l], lv [N,C_l])]_l -> list of [N, 2C_l]."""
    return [torch.cat([mu, lv], 1) for mu, lv in stats]


class DomainPrototypeBank:
    """Per-domain, per-layer prototype of BN statistics.

    mode='stats_aware': Kalman filter per channel with observation noise R from a Welford
    estimate, process noise q*R, warm-up with the exact cumulative mean and robust gating by
    the normalised innovation (NIS). Distances are precision-weighted (Mahalanobis).
    mode='ema': DoSe's P <- m P + (1-m) b with plain MSE distance."""

    def __init__(self, mode, ema_momentum=0.9, process_noise=1e-3, gate=9.0, warmup=8, shrinkage=0.5, margin=2.0):
        self.mode, self.m_ema, self.q, self.gate, self.warmup = mode, ema_momentum, process_noise, gate, warmup
        self.shrinkage, self.margin = shrinkage, margin
        self.state = {}

    def _init(self, obs):
        z = torch.zeros_like(obs)
        return {"m": obs.clone(), "P": z.clone(), "n": 0, "wm": z.clone(), "M2": z.clone(), "nw": 0, "rejected": 0}

    @torch.no_grad()
    def update(self, domain, stats):
        obs_layers = [o.detach().float().cpu() for o in stack_obs(stats)]
        st = self.state.setdefault(domain, [self._init(o[0]) for o in obs_layers])
        for s, obs in zip(st, obs_layers):
            for b in obs:
                self._update_one(s, b)

    def _welford(self, s, b):
        s["nw"] += 1
        d = b - s["wm"]
        s["wm"] += d / s["nw"]
        s["M2"] += d * (b - s["wm"])

    def _R(self, s):
        R = s["M2"] / max(s["nw"] - 1, 1)
        return R + 1e-4 * R.mean() + 1e-8

    def _update_one(self, s, b):
        s["n"] += 1
        if self.mode == "ema":
            s["m"] = b.clone() if s["n"] == 1 else self.m_ema * s["m"] + (1 - self.m_ema) * b
            self._welford(s, b)
            return
        if s["n"] <= self.warmup:  # exact cumulative mean while nothing is known
            s["m"] += (b - s["m"]) / s["n"]
            self._welford(s, b)
            s["P"] = self._R(s) / s["n"]
            return
        R = self._R(s)
        Pp = s["P"] + self.q * R
        nis = float(((b - s["m"]) ** 2 / (Pp + R)).mean())
        omega = min(1.0, self.gate / max(nis, 1e-12))
        K = Pp / (Pp + R / omega)
        s["m"] += K * (b - s["m"])
        s["P"] = (1 - K) * Pp
        if nis <= 4 * self.gate:
            self._welford(s, b)
        else:
            s["rejected"] += 1

    def ready(self, domain):
        st = self.state.get(domain)
        return bool(st) and st[0]["nw"] >= max(2, min(self.warmup, 2))

    def _scale(self, s):
        """Loss scale R + P, shrunk towards the layer average separately for the mean and
        log-variance halves, so near-constant channels cannot produce exploding gradients."""
        v = self._R(s) + s["P"]
        h = v.shape[0] // 2
        a = self.shrinkage
        return torch.cat([(1 - a) * v[:h] + a * v[:h].mean(), (1 - a) * v[h:] + a * v[h:].mean()])

    def scale(self, domain, device):
        return [self._scale(s).to(device) for s in self.state[domain]]

    def distance(self, domain, stats, normalized=None):
        """Per-image distance [N] of `stats` (with grad) to the prototype of `domain`."""
        normalized = self.mode == "stats_aware" if normalized is None else normalized
        obs = stack_obs(stats)
        st = self.state[domain]
        out = 0.0
        for o, s in zip(obs, st):
            m = s["m"].to(o.device)
            if normalized:
                z = (o - m) / self._scale(s).to(o.device).sqrt()
                out = out + hinge_z(z, self.margin).mean(1)
            else:
                out = out + ((o - m) ** 2).mean(1)
        return out / len(obs)

    def summary(self):
        return {d: {"observations": st[0]["n"], "rejected_outliers": sum(s["rejected"] for s in st)}
                for d, st in self.state.items()}

    def state_dict(self):
        return {"mode": self.mode, "state": self.state}

    def load_state_dict(self, sd):
        self.state = sd["state"]


def hinge_z(z, margin):
    """Penalise only deviations beyond `margin` natural standard deviations (Huber beyond)."""
    e = F.relu(z.abs() - margin)
    return F.huber_loss(e, torch.zeros_like(e), reduction="none", delta=1.0)


def layer_distance(student_stats, old_stats, scales=None, margin=0.0):
    """DoSe Eq. 6 between student and old student; with `scales`, stats-aware (z-units, hinge). -> [N]."""
    out = 0.0
    so, oo = stack_obs(student_stats), stack_obs(old_stats)
    for i, (a, b) in enumerate(zip(so, oo)):
        if scales is not None:
            d = hinge_z((a - b.detach()) / scales[i].sqrt(), margin)
        else:
            d = (a - b.detach()) ** 2
        out = out + d.mean(1)
    return out / len(so)


class TeacherRouter:
    """Data-free 'is this image external to the teacher?' score.

    A BN layer's running mean/var summarise the teacher's training data. For each image we
    take KL(N(mu_img, var_img) || N(running_mean, running_var)) averaged over channels and
    tracked layers, calibrate on images from the teacher's own domain(s) and map to a soft
    score r = sigmoid((d - (mean + z*std)) / (sharpness*std)). r ~ 1 means external."""

    def __init__(self, teacher, k, z=2.0, sharpness=0.5):
        self.rec = BNStatRecorder(teacher, k)
        self.z, self.sharp = z, sharpness
        self.mu = self.sd = None
        self.calibrated_on = None

    def distances(self, stats):
        out = 0.0
        for (mu, lv), bn in zip(stats, self.rec.layers):
            rm, rv = bn.running_mean.float(), bn.running_var.float() + bn.eps
            var = lv.exp()
            kl = 0.5 * (var / rv + (mu - rm) ** 2 / rv - 1 + torch.log(rv) - lv)
            out = out + kl.mean(1)
        return out / len(stats)

    def calibrate(self, dists, source):
        d = torch.cat(dists)
        self.mu, self.sd = float(d.mean()), float(d.std().clamp_min(1e-6)) if len(d) > 1 else 1.0
        self.calibrated_on = source

    def score(self, dists):
        thr = self.mu + self.z * self.sd
        return torch.sigmoid((dists - thr) / (self.sharp * self.sd))

    def remove(self):
        self.rec.remove()
