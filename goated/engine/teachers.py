"""Phase 1: obtain one teacher per stage (load a checkpoint or train it supervised on its
domains). Teachers are only *used* by the stage they belong to; afterwards they are treated
as unavailable, exactly as in continual distillation."""
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from ..config import build_stages, teacher_domains
from ..data.dataset import make_loader
from ..models.segmenter import Segmenter
from .common import amp_ctx, build_optimizer, build_scheduler, evaluate, write_json


def teacher_path(cfg, name):
    spec = (cfg["teachers"].get("per_teacher") or {}).get(name) or {}
    return Path(spec.get("checkpoint") or Path(cfg["teachers"]["dir"]) / f"{name}.pt")


def load_teacher(cfg, name, device):
    path = teacher_path(cfg, name)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec = ck.get("model_spec", cfg["models"]["teacher"])
    spec = {**spec, "pretrained": False}
    model = Segmenter(spec, len(cfg["classes"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _train_one(cfg, name, splits, device):
    tc = cfg["teachers"]
    doms = teacher_domains(cfg, name)
    train_rows = [r for d in doms for r in splits[d]["train"]]
    val_rows = [r for d in doms for r in splits[d].get("val", [])]
    model = Segmenter(cfg["models"]["teacher"], len(cfg["classes"])).to(device)
    loader = make_loader(cfg, train_rows, "train", shuffle=True, batch_size=tc.get("batch_size"), drop_last=True)
    val_loader = make_loader(cfg, val_rows, "eval", shuffle=False) if val_rows else None
    opt = build_optimizer(model.parameters(), tc)
    sched = build_scheduler(opt, tc, tc["epochs"] * len(loader))
    scaler = torch.amp.GradScaler(enabled=tc.get("amp", True) and device.type == "cuda")
    ign = cfg["ignore_index"]
    cw = torch.tensor(tc["class_weights"], device=device) if tc.get("class_weights") else None
    best, best_state, log = -1.0, None, []
    print(f"[teacher {name}] training on {doms}: {len(train_rows)} images, {tc['epochs']} epochs")
    for ep in range(tc["epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in loader:
            x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
            if (y != ign).sum() == 0:
                continue
            with amp_ctx(tc.get("amp", True), device):
                logits = model(x)
            logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, y, weight=cw, ignore_index=ign)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 5.0))
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        rec = {"epoch": ep + 1, "loss": tot / max(n, 1), "sec": round(time.time() - t0, 1)}
        last = ep + 1 == tc["epochs"]
        if val_loader and ((ep + 1) % tc.get("eval_every", 1) == 0 or last):
            m = evaluate(model, val_loader, cfg, device)
            rec["val_miou"] = m["miou"]
            if m["miou"] > best:
                best, best_state = m["miou"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[teacher {name}] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        log.append(rec)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log, doms


def prepare_teachers(cfg, splits, device, eval_loaders):
    """Ensure every stage teacher has a checkpoint; report each teacher on every domain."""
    names = list(dict.fromkeys(s["teacher"] for s in build_stages(cfg)))
    report = {}
    for name in names:
        path = teacher_path(cfg, name)
        if path.exists() and not cfg["teachers"].get("retrain", False):
            print(f"[teacher {name}] using checkpoint {path}")
            model = load_teacher(cfg, name, device)
            log, doms = None, teacher_domains(cfg, name)
        else:
            if not cfg["teachers"].get("train", True):
                raise FileNotFoundError(f"Teacher checkpoint {path} missing and teachers.train is false")
            model, log, doms = _train_one(cfg, name, splits, device)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "model_spec": cfg["models"]["teacher"],
                        "train_on": doms, "log": log}, path)
        report[name] = {"train_on": doms, "checkpoint": str(path), "log": log,
                        "test_miou": {d: evaluate(model, ld, cfg, device)["miou"] for d, ld in eval_loaders.items()}}
        print(f"[teacher {name}] test mIoU: " + ", ".join(f"{d}={v:.2f}" for d, v in report[name]["test_miou"].items()))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(Path(cfg["output_dir"]) / "teachers_summary.json", report)
    return report
