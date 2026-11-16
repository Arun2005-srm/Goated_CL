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
    d = cfg.get("device", "auto")
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


def metrics_from_confusion(cm, classes):
    cm = cm.double()
    tp = cm.diag()
    gt, pr = cm.sum(1), cm.sum(0)
    union = gt + pr - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.full_like(tp, float("nan")))
    present = gt > 0  # classes absent from GT are excluded from mIoU
    valid = present & ~torch.isnan(iou)
    return {
        "miou": float(iou[valid].mean() * 100) if valid.any() else float("nan"),
        "pixel_acc": float(tp.sum() / cm.sum().clamp_min(1) * 100),
        "per_class_iou": {c: (None if not present[i] else round(float(iou[i] * 100), 2)) for i, c in enumerate(classes)},
    }


@torch.no_grad()
def evaluate(model, loader, cfg, device, max_batches=None):
    C, ign = len(cfg["classes"]), cfg["ignore_index"]
    was = model.training
    model.eval()
    cm = torch.zeros(C, C, dtype=torch.long, device=device)
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
        with amp_ctx(cfg["eval"].get("amp", True), device):
            logits = model(x)
        pred = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        cm += confusion(pred.flatten(), y.flatten(), C, ign)
    model.train(was)
    return metrics_from_confusion(cm.cpu(), cfg["classes"])


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
