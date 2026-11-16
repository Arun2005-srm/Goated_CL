"""Entry point.

    python train.py --config configs/driving_cs_bdd_idd.yaml
    python train.py --config configs/smoke.yaml --set components.teacher_echo=false
    python train.py --config configs/driving_cs_bdd_idd.yaml --only eda
"""
import argparse

from goated.config import load_config, validate
from goated.pipeline import run


def main():
    p = argparse.ArgumentParser(description="GOATED continual distillation pipeline")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key")
    p.add_argument("--only", choices=["validate", "eda", "teachers", "distill"], default=None,
                   help="stop after this phase (distill = full run)")
    p.add_argument("--resume", help="path to a last.pt of the same run")
    a = p.parse_args()
    cfg = validate(load_config(a.config, a.set))
    run(cfg, only=None if a.only == "distill" else a.only, resume=a.resume)


if __name__ == "__main__":
    main()
