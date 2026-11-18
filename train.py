"""Entry point.

    python train.py --config configs/driving_cs_bdd_idd.yaml
    python train.py --config configs/smoke.yaml --set components.teacher_echo=false
    python train.py --config configs/driving_cs_bdd_idd.yaml --only eda
"""
import argparse
from datetime import datetime
from pathlib import Path

from goated.config import load_config, validate
from goated.pipeline import run


def run_dir(base, a):
    """Every run gets its own folder: <output_dir>/<YYYYmmdd-HHMMSS>[_name]. Resuming reuses the
    folder of the checkpoint; --exact-output-dir keeps output_dir as given."""
    if a.resume:
        return Path(a.resume).resolve().parent.as_posix()
    if a.exact_output_dir:
        return Path(base).as_posix()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{a.name}" if a.name else "")
    out, i = Path(base) / stamp, 1
    while out.exists():
        out, i = Path(base) / f"{stamp}-{i}", i + 1
    return out.as_posix()


def main():
    p = argparse.ArgumentParser(description="GOATED continual distillation pipeline")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key")
    p.add_argument("--only", choices=["validate", "eda", "teachers", "distill"], default=None,
                   help="stop after this phase (distill = full run)")
    p.add_argument("--resume", help="path to a last.pt of the same run (continues in that run's folder)")
    p.add_argument("--name", default="", help="optional tag appended to the run folder name")
    p.add_argument("--exact-output-dir", action="store_true",
                   help="write directly into output_dir (no timestamped sub-folder); used by run_ablations.py")
    a = p.parse_args()
    cfg = validate(load_config(a.config, a.set))
    cfg["output_dir"] = run_dir(cfg["output_dir"], a)
    print(f"[run] output folder: {cfg['output_dir']}")
    run(cfg, only=None if a.only == "distill" else a.only, resume=a.resume)


if __name__ == "__main__":
    main()
