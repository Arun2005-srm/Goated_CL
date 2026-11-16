"""Exemplar buffer with teacher echoes (components 3, 4), Gaussian class prototypes (10)
and stochastic restore (11)."""
import random
import torch
import torch.nn.functional as F


def vog_scores(models, x):
    """Variance of input gradients across models (DoSe Sec. 3.3, two checkpoints).
    Per image: mean over pixels of std across models of d(sum max-logit)/dx."""
    grads = []
    for m in models:
        was = m.training
        m.eval()
        xi = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            score = m(xi).float().max(1).values.sum()
            g, = torch.autograd.grad(score, xi)
        m.train(was)
        grads.append(g.abs().mean(1))  # channel-pooled gradient magnitude map [B,H,W]
    G = torch.stack(grads)
    return G.std(0, unbiased=False).flatten(1).mean(1)


def entropy_scores(p1):
    return -(p1 * torch.log(p1.clamp_min(1e-8))).sum(1).flatten(1).mean(1)


class ExemplarBuffer:
    """Per-stage buckets of the highest-priority images. With echoes, each entry also stores
    the stage teacher's top-k T=1 probabilities, pooled to 1/stride of the input."""

    def __init__(self, per_stage, topk, stride, store_labels):
        self.cap, self.topk, self.stride, self.store_labels = per_stage, topk, stride, store_labels
        self.buckets = {}

    def __len__(self):
        return sum(len(b) for b in self.buckets.values())

    def _compress(self, p1, size):
        h, w = max(1, size[0] // self.stride), max(1, size[1] // self.stride)
        p = F.adaptive_avg_pool2d(p1.float(), (h, w))
        k = min(self.topk, p.shape[1])
        vals, idx = p.topk(k, dim=1)
        return vals.half().cpu(), idx.to(torch.int16).cpu(), p.shape[1]

    def offer(self, stage, images, domains, priorities, teacher_p1=None, labels=None):
        bucket = self.buckets.setdefault(stage, [])
        echo = self._compress(teacher_p1, images.shape[-2:]) if teacher_p1 is not None else None
        for i in range(images.shape[0]):
            pr = float(priorities[i])
            if len(bucket) >= self.cap:
                j = min(range(len(bucket)), key=lambda t: bucket[t]["priority"])
                if pr <= bucket[j]["priority"]:
                    continue
                bucket.pop(j)
            bucket.append({
                "image": images[i].detach().half().cpu(),
                "domain": int(domains[i]),
                "priority": pr,
                "echo": (echo[0][i], echo[1][i], echo[2]) if echo else None,
                "label": labels[i].to(torch.uint8).cpu() if (labels is not None and self.store_labels) else None,
            })

    @staticmethod
    def _expand(vals, idx, C):
        vals, idx = vals.float(), idx.long()
        k = vals.shape[0]
        rest = ((1 - vals.sum(0, keepdim=True)).clamp_min(0) / max(C - k, 1)) if C > k else torch.zeros_like(vals[:1])
        p = rest.expand(C, *vals.shape[1:]).clone()
        p.scatter_(0, idx, vals)
        return p / p.sum(0, keepdim=True).clamp_min(1e-8)

    def sample(self, n, size, device, exclude=()):
        keys = [k for k, b in self.buckets.items() if b and k not in exclude]
        if not keys:
            return None
        picks = [random.choice(self.buckets[random.choice(keys)]) for _ in range(n)]
        imgs = torch.stack([F.interpolate(e["image"].float()[None], size=size, mode="bilinear",
                                          align_corners=False)[0] if e["image"].shape[-2:] != tuple(size)
                            else e["image"].float() for e in picks]).to(device)
        out = {"image": imgs, "domain": torch.tensor([e["domain"] for e in picks], device=device)}
        if all(e["echo"] is not None for e in picks):
            ps = [self._expand(*e["echo"]) for e in picks]
            hw = ps[0].shape[-2:]
            out["echo"] = torch.stack([F.interpolate(p[None], size=hw, mode="bilinear", align_corners=False)[0]
                                       if p.shape[-2:] != hw else p for p in ps]).to(device)
        if all(e["label"] is not None for e in picks):
            labs = [e["label"].long() for e in picks]
            out["label"] = torch.stack([F.interpolate(l[None, None].float(), size=size, mode="nearest")[0, 0].long()
                                        if l.shape != tuple(size) else l for l in labs]).to(device)
        return out

    def state_dict(self):
        return {"buckets": self.buckets}

    def load_state_dict(self, sd):
        self.buckets = sd["buckets"]

    def summary(self):
        return {k: {"size": len(b), "min_priority": min((e["priority"] for e in b), default=None)}
                for k, b in self.buckets.items()}


class ClassGaussianBank:
    """Welford mean/variance of decoder features per (domain, class). Replay draws
    z ~ N(mu, diag var) and trains the classifier head to keep old-domain decision regions."""

    def __init__(self, num_classes, max_pixels=512, min_count=50):
        self.C, self.max_pixels, self.min_count = num_classes, max_pixels, min_count
        self.state = {}

    @torch.no_grad()
    def update(self, domain, feats, labels, ignore):
        f = feats.detach().float().permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
        y = F.interpolate(labels[:, None].float(), size=feats.shape[-2:], mode="nearest").long().flatten()
        st = self.state.setdefault(domain, {})
        for c in y.unique().tolist():
            if c == ignore or c >= self.C:
                continue
            fc = f[y == c]
            if len(fc) > self.max_pixels:
                fc = fc[torch.randperm(len(fc), device=fc.device)[: self.max_pixels]]
            fc = fc.cpu()
            s = st.setdefault(c, {"n": 0, "mean": torch.zeros(f.shape[1]), "M2": torch.zeros(f.shape[1])})
            nb, mb = len(fc), fc.mean(0)
            d = mb - s["mean"]
            tot = s["n"] + nb
            s["mean"] += d * nb / tot
            s["M2"] += ((fc - mb) ** 2).sum(0) + d ** 2 * s["n"] * nb / tot
            s["n"] = tot

    def sample(self, domains, per_class, device):
        zs, ys = [], []
        for d in domains:
            for c, s in self.state.get(d, {}).items():
                if s["n"] < self.min_count:
                    continue
                std = (s["M2"] / (s["n"] - 1)).clamp_min(1e-8).sqrt()
                zs.append(s["mean"] + std * torch.randn(per_class, len(std)))
                ys.append(torch.full((per_class,), c, dtype=torch.long))
        if not zs:
            return None, None
        return torch.cat(zs).to(device), torch.cat(ys).to(device)

    def domains(self):
        return list(self.state)

    def state_dict(self):
        return {"state": self.state}

    def load_state_dict(self, sd):
        self.state = sd["state"]


class StochasticRestore:
    """theta <- M * theta_anchor + (1 - M) * theta, M ~ Bernoulli(p) per weight (DoSe/CoTTA)."""

    def __init__(self, model, p):
        self.p = p
        self.anchor = {n: q.detach().clone() for n, q in model.named_parameters() if q.requires_grad}
        self.model = model

    @torch.no_grad()
    def step(self):
        for n, q in self.model.named_parameters():
            if n in self.anchor:
                mask = torch.rand_like(q) < self.p
                q.copy_(torch.where(mask, self.anchor[n], q))
