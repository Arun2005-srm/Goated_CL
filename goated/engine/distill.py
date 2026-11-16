"""Phase 2: continual distillation of one student through the stage sequence.

At stage t only teacher T_t is available. Every component is a construction-time switch:
a disabled component builds nothing, runs no forward pass and contributes no gradient."""
from copy import deepcopy
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from ..config import build_stages, teacher_domains
from ..data.dataset import make_loader
from ..models.segmenter import Segmenter, count_params
from ..components.losses import multi_source_kd, pseudo_label_ce, supervised_ce
from ..components.prototypes import BNStatRecorder, DomainPrototypeBank, TeacherRouter, layer_distance
from ..components.memory import ExemplarBuffer, ClassGaussianBank, StochasticRestore, vog_scores, entropy_scores
from .common import amp_ctx, build_optimizer, build_scheduler, evaluate, write_json
from .teachers import load_teacher
from .reports import write_reports


class GoatedDistiller:
    def __init__(self, cfg, splits, device, eval_loaders):
        self.cfg, self.splits, self.device, self.eval_loaders = cfg, splits, device, eval_loaders
        self.comp, self.hp = cfg["components"], cfg["hparams"]
        self.C, self.ign = len(cfg["classes"]), cfg["ignore_index"]
        self.T = cfg["distill"]["temperature"]
        self.domains = list(cfg["domains"])
        self.didx = {d: i for i, d in enumerate(self.domains)}
        self.out = Path(cfg["output_dir"])
        self.stages = build_stages(cfg)
        self.student = Segmenter(cfg["models"]["student"], self.C).to(device)
        self.old = None
        c, h = self.comp, self.hp
        uses_bn = c["bn_layer_distill"] or c["bn_domain_prototypes"]
        self.bank = DomainPrototypeBank("stats_aware" if c["stats_aware_prototypes"] else "ema",
                                        **h["prototypes"]["bank"]) if uses_bn else None
        self.buffer = ExemplarBuffer(h["buffer"]["per_stage"], h["buffer"]["echo_topk"], h["buffer"]["echo_stride"],
                                     store_labels=c["supervised_ce"]) if c["exemplar_buffer"] else None
        self.cbank = ClassGaussianBank(self.C, **h["class_proto"]["bank"]) if c["class_prototype_replay"] else None
        self.R, self.history, self.start_stage = [], [], 0
        total, trainable = count_params(self.student)
        write_json(self.out / "model_components.json", {
            "components": c, "student_params": total, "student_trainable": trainable,
            "constructed": {"bn_prototype_bank": self.bank is not None and self.bank.mode,
                            "exemplar_buffer": self.buffer is not None,
                            "class_gaussian_bank": self.cbank is not None}})

    # ------------------------------------------------------------------ checkpointing
    def save(self, si):
        torch.save({"stage": si, "student": self.student.state_dict(),
                    "old": self.old.state_dict() if self.old is not None else None,
                    "bank": self.bank.state_dict() if self.bank else None,
                    "buffer": self.buffer.state_dict() if self.buffer else None,
                    "cbank": self.cbank.state_dict() if self.cbank else None,
                    "R": self.R, "history": self.history, "components": self.comp}, self.out / "last.pt")

    def resume(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["components"] != self.comp:
            raise ValueError("Resume needs the same component switches as the checkpoint")
        self.student.load_state_dict(ck["student"])
        if ck["old"] is not None:
            self.old = deepcopy(self.student)
            self.old.load_state_dict(ck["old"])
            self._freeze(self.old)
        for obj, key in ((self.bank, "bank"), (self.buffer, "buffer"), (self.cbank, "cbank")):
            if obj is not None and ck[key] is not None:
                obj.load_state_dict(ck[key])
        self.R, self.history, self.start_stage = ck["R"], ck["history"], ck["stage"] + 1
        print(f"[resume] continuing at stage {self.start_stage + 1}/{len(self.stages)}")

    @staticmethod
    def _freeze(m):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ routing
    @torch.no_grad()
    def _calibrate_router(self, teacher, stage, loader):
        h = self.hp["routing"]
        router = TeacherRouter(teacher, self.hp["prototypes"]["layers"], h["z"], h["sharpness"])
        own = teacher_domains(self.cfg, stage["teacher"])
        rows = [r for d in own for r in self.splits[d]["train"]]
        src = f"teacher domains {own}"
        cal_loader = make_loader(self.cfg, rows, "distill", shuffle=True) if rows else loader
        if not rows:
            src = "distillation data (teacher domains unknown)"
        dists = []
        for i, b in enumerate(cal_loader):
            if i >= h["calib_batches"]:
                break
            with router.rec, amp_ctx(self.cfg["distill"]["amp"], self.device):
                teacher(b["image"].to(self.device))
            dists.append(router.distances(router.rec.take()).cpu())
        router.calibrate(dists, src)
        print(f"[router] calibrated on {src}: d={router.mu:.4f}±{router.sd:.4f}")
        return router

    # ------------------------------------------------------------------ one stage
    def run_stage(self, si, stage):
        cfg, c, h, dev = self.cfg, self.comp, self.hp, self.device
        dc = cfg["distill"]
        print(f"\n=== Stage {si + 1}/{len(self.stages)}: {stage['name']} | teacher={stage['teacher']} "
              f"| distill_on={stage['distill_on']} | external={stage['external']} ===")
        need_teacher = c["teacher_kd"] or c["aug_pseudo_label"] or c["teacher_echo"] or c["class_prototype_replay"] \
            or (c["exemplar_buffer"] and h["buffer"]["priority"] in ("vog", "entropy")) or c["auto_domain_routing"]
        teacher = load_teacher(cfg, stage["teacher"], dev) if need_teacher else None
        rows = [r for d in stage["distill_on"] for r in self.splits[d]["train"]]
        loader = make_loader(cfg, rows, "distill", shuffle=True, drop_last=True)
        params = [p for p in self.student.parameters() if p.requires_grad]
        opt = build_optimizer(params, dc)
        steps = stage["epochs"] * len(loader)
        sched = build_scheduler(opt, dc, steps)
        scaler = torch.amp.GradScaler(enabled=dc["amp"] and dev.type == "cuda")
        old_on = self.old is not None
        external_ids = {self.didx[d] for d in stage["external"]}
        stage_dom_ids = {self.didx[d] for d in stage["distill_on"]}
        router = self._calibrate_router(teacher, stage, loader) if (c["auto_domain_routing"] and c["old_student_kd"]
                                                                      and old_on) else None
        k_layers = h["prototypes"]["layers"]
        s_rec = BNStatRecorder(self.student, k_layers) if self.bank else None
        o_rec = BNStatRecorder(self.old, k_layers) if (self.bank and c["bn_layer_distill"] and old_on) else None
        restore = StochasticRestore(self.student, h["restore"]["p"]) if c["stochastic_restore"] else None
        W = h["weights"]
        route_log = {}
        self.student.train()
        step = 0
        for ep in range(stage["epochs"]):
            t0, agg = time.time(), {}
            last_epoch = ep + 1 == stage["epochs"]
            for b in loader:
                x = b["image"].to(dev, non_blocking=True)
                y = b["mask"].to(dev, non_blocking=True)
                dom = torch.as_tensor(b["domain"], device=dev)
                B = x.shape[0]
                rep = None
                if self.buffer is not None and len(self.buffer) and step % h["buffer"]["replay_every"] == 0:
                    rep = self.buffer.sample(h["buffer"]["replay_batch"], x.shape[-2:], dev, exclude=(stage["name"],))
                x_all = torch.cat([x, rep["image"]]) if rep is not None else x
                dom_all = torch.cat([dom, rep["domain"]]) if rep is not None else dom
                losses = {}
                with amp_ctx(dc["amp"], dev):
                    with torch.no_grad():
                        t_p1 = None
                        if teacher is not None:
                            if router is not None:
                                with router.rec:
                                    t_logits = teacher(x)
                                r_ext = router.score(router.distances(router.rec.take()))
                            else:
                                t_logits = teacher(x)
                            t_p1 = torch.softmax(t_logits.float(), 1)
                            if c["aug_pseudo_label"]:
                                t_flip = torch.softmax(teacher(x.flip(-1)).float(), 1).flip(-1)
                                t_aug = 0.5 * (t_p1 + t_flip)
                        o_p1 = None
                        if old_on and (c["old_student_kd"] or o_rec is not None):
                            if o_rec is not None:
                                with o_rec:
                                    o_logits = self.old(x_all)
                                o_stats = o_rec.take()
                            else:
                                o_logits = self.old(x_all)
                            o_p1 = torch.softmax(o_logits.float(), 1)
                    if s_rec is not None:
                        with s_rec:
                            s_logits, s_feats = self.student(x_all, return_features=True)
                        s_stats = s_rec.take()
                    else:
                        s_logits, s_feats = self.student(x_all, return_features=True)
                s_cur = s_logits[:B]

                # ---- current batch: teacher + old student (routed) -------------------------
                if router is None:
                    if h["old_student_on"] == "all":
                        r_ext = torch.ones(B, device=dev)
                    else:
                        r_ext = torch.tensor([float(int(d) in external_ids) for d in dom], device=dev)
                if old_on and c["old_student_kd"]:
                    for d in dom.unique().tolist():
                        route_log.setdefault(self.domains[d], []).append(float(r_ext[dom == d].mean()))
                sources = []
                if c["teacher_kd"]:
                    sources.append((t_p1, W["teacher_kd"]))
                if old_on and c["old_student_kd"]:
                    sources.append((o_p1[:B], W["old_student_kd"] * r_ext.view(-1, 1, 1, 1)))
                if sources:
                    losses["kd_current"] = multi_source_kd(s_cur, sources, self.T, c["entropy_gated_fusion"],
                                                           h["fusion"]["tau"])
                if c["aug_pseudo_label"]:
                    losses["aug_pseudo"] = W["aug_pseudo"] * pseudo_label_ce(s_cur, t_aug, h["aug_pseudo"]["threshold"], self.ign)
                if c["supervised_ce"]:
                    losses["ce"] = W["ce"] * supervised_ce(s_cur, y, self.ign)

                # ---- replay batch: teacher echoes of past stages + old student ------------
                if rep is not None:
                    s_rep = s_logits[B:]
                    rsrc = []
                    if c["teacher_echo"] and "echo" in rep:
                        rsrc.append((rep["echo"], W["echo"]))
                    if old_on and c["old_student_kd"]:
                        rsrc.append((o_p1[B:], W["old_student_kd"]))
                    if rsrc:
                        losses["kd_replay"] = W["replay"] * multi_source_kd(s_rep, rsrc, self.T, c["entropy_gated_fusion"],
                                                                            h["fusion"]["tau"])
                    if c["supervised_ce"] and "label" in rep:
                        losses["ce_replay"] = W["ce"] * W["replay"] * supervised_ce(s_rep, rep["label"], self.ign)

                # ---- BN prototypes ----------------------------------------------------------
                if self.bank is not None:
                    if c["bn_domain_prototypes"]:
                        terms = []
                        for d in dom_all.unique().tolist():
                            name = self.domains[d]
                            if self.bank.ready(name):
                                sel = dom_all == d
                                terms.append(self.bank.distance(name, [(mu[sel], lv[sel]) for mu, lv in s_stats]).sum())
                        if terms:
                            losses["bn_domain"] = W["bn_domain"] * torch.stack(terms).sum() / x_all.shape[0]
                    if o_rec is not None and rep is not None:
                        # Anchor feature statistics of *replayed old-domain* images to the old
                        # student; current-domain images stay free to adapt (plasticity).
                        terms = []
                        for d in rep["domain"].unique().tolist():
                            sel = torch.zeros_like(dom_all, dtype=torch.bool)
                            sel[B:] = dom_all[B:] == d
                            name = self.domains[d]
                            scales = self.bank.scale(name, dev) if (c["stats_aware_prototypes"] and self.bank.ready(name)) else None
                            terms.append(layer_distance([(mu[sel], lv[sel]) for mu, lv in s_stats],
                                                        [(mu[sel], lv[sel]) for mu, lv in o_stats], scales,
                                                        self.bank.margin * 0.5).sum())
                        losses["bn_layer"] = W["bn_layer"] * torch.stack(terms).sum() / max(x_all.shape[0] - B, 1)
                    cur = [(mu[:B].detach(), lv[:B].detach()) for mu, lv in s_stats]
                    for d in dom.unique().tolist():
                        sel = dom == d
                        self.bank.update(self.domains[d], [(mu[sel], lv[sel]) for mu, lv in cur])

                # ---- Gaussian class-prototype replay ---------------------------------------
                if self.cbank is not None:
                    old_doms = [d for d in self.cbank.domains() if d not in stage["distill_on"]]
                    z, zy = self.cbank.sample(old_doms, h["class_proto"]["samples_per_class"], dev)
                    if z is not None:
                        losses["class_proto"] = W["class_proto"] * F.cross_entropy(
                            self.student.classify_features(z).float(), zy)
                    if last_epoch:
                        if c["supervised_ce"]:
                            lab = y
                        else:
                            conf, lab = t_p1.max(1)
                            lab = lab.masked_fill(conf < h["class_proto"]["confidence"], self.ign)
                        for d in dom.unique().tolist():
                            sel = dom == d
                            self.cbank.update(self.domains[d], s_feats[:B][sel], lab[sel], self.ign)

                loss = sum(losses.values()) if losses else torch.zeros((), device=dev)
                opt.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, dc.get("grad_clip", 5.0))
                    scaler.step(opt)
                    scaler.update()
                sched.step()
                if restore is not None:
                    restore.step()

                # ---- buffer update (priority = difficulty) ---------------------------------
                bh = h["buffer"]
                if self.buffer is not None and step % bh["update_every"] == 0:
                    if bh["priority"] == "vog":
                        ref = teacher if teacher is not None else self.old
                        pri = vog_scores([self.student, ref], x).tolist() if ref is not None else torch.rand(B).tolist()
                    elif bh["priority"] == "entropy":
                        pri = entropy_scores(t_p1).tolist()
                    else:
                        pri = torch.rand(B).tolist()
                    self.buffer.offer(stage["name"], x, dom.tolist(), pri,
                                      t_p1 if c["teacher_echo"] else None, y if c["supervised_ce"] else None)

                for k, v in losses.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                agg["total"] = agg.get("total", 0.0) + float(loss)
                step += 1
            n = max(len(loader), 1)
            rec = {"stage": stage["name"], "epoch": ep + 1, "sec": round(time.time() - t0, 1),
                   **{k: v / n for k, v in agg.items()}}
            self.history.append(rec)
            print(f"[stage {stage['name']}] ep {ep + 1}/{stage['epochs']} " +
                  " ".join(f"{k}={v:.4f}" for k, v in rec.items() if isinstance(v, float)))
            if dc.get("eval_every_epoch"):
                m = {d: evaluate(self.student, ld, cfg, dev, cfg["eval"].get("max_batches"))["miou"]
                     for d, ld in self.eval_loaders.items()}
                print(f"[stage {stage['name']}] quick eval mIoU: " + ", ".join(f"{d}={v:.2f}" for d, v in m.items()))

        for r in (s_rec, o_rec, router):
            if r is not None:
                r.remove()
        diag = {"stage": stage["name"],
                "routing_mean_external_score": {d: sum(v) / len(v) for d, v in route_log.items()} if route_log else None,
                "router_calibrated_on": router.calibrated_on if router else None,
                "prototype_bank": self.bank.summary() if self.bank else None,
                "buffer": self.buffer.summary() if self.buffer else None}
        print(f"[stage {stage['name']}] diagnostics: {diag}")
        del teacher
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return diag

    # ------------------------------------------------------------------ full sequence
    def run(self):
        diags = []
        for si in range(self.start_stage, len(self.stages)):
            stage = self.stages[si]
            diags.append(self.run_stage(si, stage))
            self.old = deepcopy(self.student)
            self._freeze(self.old)
            row = {d: evaluate(self.student, ld, self.cfg, self.device) for d, ld in self.eval_loaders.items()}
            self.R.append({"stage": stage["name"], **{d: m["miou"] for d, m in row.items()}})
            write_json(self.out / f"eval_stage_{si + 1:02d}_{stage['name']}.json", row)
            print(f"[eval after {stage['name']}] " + ", ".join(f"{d}={m['miou']:.2f}" for d, m in row.items()))
            torch.save(self.student.state_dict(), self.out / f"student_stage_{si + 1:02d}.pt")
            self.save(si)
        write_json(self.out / "diagnostics.json", diags)
        return write_reports(self.cfg, self.R, self.history, self.stages, self.student, self.eval_loaders, self.device)
