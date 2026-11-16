"""End-to-end pipeline: validate -> discover/split -> (cache) -> EDA -> teachers -> distillation -> reports."""
from pathlib import Path
import yaml

from .config import build_stages, dump
from .data.discovery import discover_all, save_manifests
from .data.dataset import make_loader, maybe_cache
from .engine.common import get_device, seed_all


def eval_loaders(cfg, splits, split=None):
    split = split or cfg["data"].get("eval_split", "test")
    doms = cfg["protocol"].get("eval_on") or list(cfg["domains"])
    out = {}
    for d in doms:
        rows = [r for r in splits[d].get(split, []) if r.mask]
        if not rows:
            print(f"[eval] WARNING: domain {d} has no labelled '{split}' split; it is not evaluated")
            continue
        out[d] = make_loader(cfg, rows, "eval", shuffle=False, batch_size=cfg["eval"].get("batch_size"))
    return out


def run(cfg, only=None, resume=None):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    dump(cfg, out / "resolved_config.yaml")
    stages = build_stages(cfg)
    print("Stages:\n" + yaml.safe_dump(stages, sort_keys=False))
    print("Components:\n" + yaml.safe_dump(cfg["components"], sort_keys=False))
    if only == "validate":
        print("Configuration is valid.")
        return
    splits = discover_all(cfg)
    save_manifests(splits, out / "manifests")
    for d, s in splits.items():
        print(f"[data] {d}: " + ", ".join(f"{k}={len(v)}" for k, v in s.items()))
    if (cfg["eda"].get("enabled", True) and only in (None, "eda")) or only == "eda":
        from .data.eda import run_eda
        print(f"[eda] written to {run_eda(cfg, splits, out)}")
        if only == "eda":
            return
    splits = maybe_cache(cfg, splits)
    device = get_device(cfg)
    print(f"[device] {device}")
    loaders = eval_loaders(cfg, splits)
    from .engine.teachers import prepare_teachers
    prepare_teachers(cfg, splits, device, loaders)
    if only == "teachers":
        return
    from .engine.distill import GoatedDistiller
    seed_all(cfg["seed"])
    dist = GoatedDistiller(cfg, splits, device, loaders, eval_loaders(cfg, splits, "val"))
    if resume:
        dist.resume(resume)
    return dist.run()
