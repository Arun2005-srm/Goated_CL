"""Shared utilities: seeding, device, optimisers, schedules, evaluation and metrics."""
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    from ..gpu import pin_single_gpu
    pin_single_gpu(cfg)  # one GPU only (no-op if train.py already pinned it)
    d = cfg.get("device", "auto")
    if d != "cpu" and torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else torch.device(d)


def build_optimizer(params, oc):
    name = oc.get("optimizer", "adamw")
    if name == "sgd":
        return torch.optim.SGD(params, lr=oc["lr"], momentum=oc.get("momentum", 0.9),
                               weight_decay=oc.get("weight_decay", 0.0), nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=oc["lr"], weight_decay=oc.get("weight_decay", 0.0))
    return torch.optim.AdamW(params, lr=oc["lr"], weight_decay=oc.get("weight_decay", 0.01))


def build_scheduler(opt, oc, total_steps):
    kind = oc.get("scheduler", "poly")
    warm = int(oc.get("warmup_steps", 0))
    if kind == "none":
        f = lambda s: 1.0
    elif kind == "cosine":
        f = lambda s: 0.5 * (1 + np.cos(np.pi * min(s, total_steps) / max(total_steps, 1)))
    else:
        f = lambda s: (1 - min(s, total_steps - 1) / max(total_steps, 1)) ** oc.get("poly_power", 0.9)
    g = lambda s: min(1.0, (s + 1) / warm) * f(s) if warm else f(s)
    return torch.optim.lr_scheduler.LambdaLR(opt, g)


def amp_ctx(enabled, device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def confusion(pred, target, C, ignore):
    k = target != ignore
    return torch.bincount(target[k] * C + pred[k], minlength=C * C).reshape(C, C)


def metrics_from_confusion(cm, classes, b_inter=None, b_union=None):
    """IoU / F1 (= Dice) per class from a confusion matrix; optional per-class boundary IoU.
    Classes absent from the ground truth are excluded from the macro averages (reported None)."""
    cm = cm.double()
    tp = cm.diag()
    gt, pr = cm.sum(1), cm.sum(0)
    union = gt + pr - tp
    present = gt > 0
    iou = tp / union.clamp_min(1)
    f1 = 2 * tp / (gt + pr).clamp_min(1)
    pc = lambda v, i: None if not present[i] else round(float(v[i] * 100), 2)
    out = {
        "miou": float(iou[present].mean() * 100) if present.any() else float("nan"),
        "mf1": float(f1[present].mean() * 100) if present.any() else float("nan"),
        "pixel_acc": float(tp.sum() / cm.sum().clamp_min(1) * 100),
        "per_class_iou": {c: pc(iou, i) for i, c in enumerate(classes)},
        "per_class_f1": {c: pc(f1, i) for i, c in enumerate(classes)},
    }
    if b_inter is not None:
        b_inter, b_union = b_inter.double(), b_union.double()
        biou = b_inter / b_union.clamp_min(1)
        bvalid = present & (b_union > 0)
        out["biou"] = float(biou[bvalid].mean() * 100) if bvalid.any() else float("nan")
        out["per_class_biou"] = {c: (round(float(biou[i] * 100), 2) if bvalid[i] else None)
                                 for i, c in enumerate(classes)}
    return out


def _boundary(mask, d):
    """Inner boundary band of width d of a boolean mask [B,H,W] (erosion via max-pool)."""
    m = mask.float()[:, None]
    eroded = -F.max_pool2d(-m, 2 * d + 1, stride=1, padding=d)
    return mask & (eroded[:, 0] < 0.5)


def boundary_stats(pred, gt, C, ignore, ratio):
    """Per-class boundary intersection/union (Boundary IoU, Cheng et al. 2021), ignore-masked."""
    H, W = gt.shape[-2:]
    d = max(1, round(ratio * (H ** 2 + W ** 2) ** 0.5))
    valid = gt != ignore
    inter = torch.zeros(C, device=gt.device, dtype=torch.long)
    union = torch.zeros(C, device=gt.device, dtype=torch.long)
    for c in torch.unique(torch.cat([gt[valid], pred[valid]])).tolist():
        if c >= C:
            continue
        bg, bp = _boundary(gt == c, d) & valid, _boundary(pred == c, d) & valid
        inter[c] += (bg & bp).sum()
        union[c] += (bg | bp).sum()
    return inter, union


@torch.no_grad()
def evaluate(model, loader, cfg, device, max_batches=None, boundary=True, return_counts=False):
    """Pixel accuracy, CE loss, per-class IoU/F1/Boundary-IoU and their macro means."""
    C, ign = len(cfg["classes"]), cfg["ignore_index"]
    ratio = cfg.get("metrics", {}).get("boundary_ratio", 0.02)
    was = model.training
    model.eval()
    cm = torch.zeros(C, C, dtype=torch.long, device=device)
    bi = torch.zeros(C, dtype=torch.long, device=device)
    bu = torch.zeros(C, dtype=torch.long, device=device)
    loss_sum, n_pix = 0.0, 0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
        with amp_ctx(cfg["eval"].get("amp", True), device):
            logits = model(x)
        logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
        k = int((y != ign).sum())
        if k:
            loss_sum += float(F.cross_entropy(logits, y, ignore_index=ign, reduction="sum"))
            n_pix += k
        pred = logits.argmax(1)
        cm += confusion(pred.flatten(), y.flatten(), C, ign)
        if boundary:
            a, u = boundary_stats(pred, y, C, ign, ratio)
            bi += a
            bu += u
    model.train(was)
    m = metrics_from_confusion(cm.cpu(), cfg["classes"], bi.cpu() if boundary else None, bu.cpu() if boundary else None)
    m["loss"] = loss_sum / max(n_pix, 1)
    if return_counts:
        m["_counts"] = (cm.cpu(), bi.cpu() if boundary else None, bu.cpu() if boundary else None, loss_sum, n_pix)
    return m


def combine_metrics(results, classes):
    """Pool confusion/boundary counts of several evaluate(..., return_counts=True) results."""
    if len(results) == 1:
        return results[0]
    cm = sum(r["_counts"][0] for r in results)
    has_b = results[0]["_counts"][1] is not None
    bi = sum(r["_counts"][1] for r in results) if has_b else None
    bu = sum(r["_counts"][2] for r in results) if has_b else None
    m = metrics_from_confusion(cm, classes, bi, bu)
    m["loss"] = sum(r["_counts"][3] for r in results) / max(sum(r["_counts"][4] for r in results), 1)
    return m


class Meter:
    """Online train metrics: loss, pixel accuracy vs GT (measurement only), teacher agreement."""

    def __init__(self, ignore):
        self.ign = ignore
        self.reset()

    def reset(self):
        self.loss = self.n = 0.0
        self.correct = self.valid = self.agree = self.agree_n = 0

    @torch.no_grad()
    def update(self, loss, logits, mask=None, teacher_p=None):
        self.loss += float(loss)
        self.n += 1
        pred = logits.argmax(1)
        if mask is not None:
            y = F.interpolate(mask[:, None].float(), size=pred.shape[-2:], mode="nearest")[:, 0].long()
            v = y != self.ign
            self.correct += int((pred[v] == y[v]).sum())
            self.valid += int(v.sum())
        if teacher_p is not None:
            t = F.interpolate(teacher_p, size=pred.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
            self.agree += int((pred == t).sum())
            self.agree_n += t.numel()

    def result(self):
        return {"train_loss": self.loss / max(self.n, 1),
                "train_acc": 100 * self.correct / self.valid if self.valid else float("nan"),
                "teacher_agreement": 100 * self.agree / self.agree_n if self.agree_n else float("nan")}


def format_epoch(tag, ep, n_ep, train, val, classes, extra=None, per_class=True):
    """Multi-line epoch report: headline numbers then a per-class IoU / F1 / bIoU table."""
    f = lambda v: "n/a" if v is None or v != v else f"{v:.2f}"
    head = (f"{tag} ep[{ep}/{n_ep}]: train loss {train['train_loss']:.4f} acc {f(train['train_acc'])}"
            + (f" (teacher-agree {f(train['teacher_agreement'])})" if train.get("teacher_agreement") is not None and train["teacher_agreement"] == train["teacher_agreement"] else "")
            )
    lines = [head]
    if val:
        lines.append(f"    val   loss {val['loss']:.4f} acc {f(val['pixel_acc'])} | mIoU {f(val['miou'])} "
                     f"| mF1 {f(val['mf1'])} | bIoU {f(val.get('biou'))}")
    for k, v in (extra or {}).items():
        lines.append(f"    {k}: {v}")
    if val and per_class:
        w = max(len(c) for c in classes) + 2
        lines.append(f"    {'class':<{w}}{'IoU':>8}{'F1':>8}{'bIoU':>8}")
        for c in classes:
            lines.append(f"    {c:<{w}}{f(val['per_class_iou'][c]):>8}{f(val['per_class_f1'][c]):>8}"
                         f"{f(val.get('per_class_biou', {}).get(c)):>8}")
    return "\n".join(lines)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
