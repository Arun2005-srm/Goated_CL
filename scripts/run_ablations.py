"""Run baselines and a leave-one-out sweep over GOATED components, then tabulate results.

    python scripts/run_ablations.py --config configs/driving_dil.yaml                 # everything
    python scripts/run_ablations.py --config configs/driving_dil.yaml --only se2d goated_full
    python scripts/run_ablations.py --config configs/smoke.yaml --set eda.enabled=false

Teachers are shared (teachers.dir), so they are trained once and reused by every run.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goated.config import COMPONENTS, load_config  # noqa: E402

OFF = {k: False for k in COMPONENTS}

PRESETS = {
    # --- baselines -------------------------------------------------------------------------
    "kl_only":          {**OFF, "teacher_kd": True},
    "self_distillation": {**OFF, "teacher_kd": True, "old_student_kd": True, "_old_student_on": "all"},
    "se2d":             {**OFF, "teacher_kd": True, "old_student_kd": True, "_old_student_on": "external"},
    "dose_style":       {**OFF, "aug_pseudo_label": True, "old_student_kd": True, "exemplar_buffer": True,
                         "bn_layer_distill": True, "bn_domain_prototypes": True, "stochastic_restore": True,
                         "_old_student_on": "all"},
    "finetune_supervised": {**OFF, "supervised_ce": True},
    # --- full model --------------------------------------------------------------------------
    "goated_full": {},
}
# Leave-one-out from the full model (dependencies switched off together).
LOO = {
    "teacher_echo": ["teacher_echo"],
    "exemplar_buffer": ["exemplar_buffer", "teacher_echo"],
    "old_student_kd": ["old_student_kd", "auto_domain_routing"],
    "stats_aware_prototypes": ["stats_aware_prototypes"],
    "bn_prototypes": ["bn_layer_distill", "bn_domain_prototypes", "stats_aware_prototypes"],
    "auto_domain_routing": ["auto_domain_routing"],
    "entropy_gated_fusion": ["entropy_gated_fusion"],
    "class_prototype_replay": ["class_prototype_replay"],
}
for name, comps in LOO.items():
    PRESETS[f"no_{name}"] = {c: False for c in comps}


def overrides(preset):
    out = []
    for k, v in preset.items():
        if k == "_old_student_on":
            out.append(f"hparams.old_student_on={v}")
        else:
            out.append(f"components.{k}={str(v).lower()}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    base = load_config(a.config, a.set)
    root = Path(base["output_dir"]).parent
    names = a.only or list(PRESETS)
    for n in names:
        out_dir = root / n
        if (out_dir / "results.json").exists():
            print(f"[skip] {n}: results exist")
            continue
        cmd = [sys.executable, str(ROOT / "train.py"), "--config", a.config, *sum([["--set", s] for s in a.set], []),
               "--set", f"output_dir={out_dir.as_posix()}", *sum([["--set", s] for s in overrides(PRESETS[n])], [])]
        print(f"\n######## {n} ########\n" + " ".join(cmd))
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            print(f"[fail] {n} exited with {r.returncode}")
    rows = []
    for n in names:
        f = root / n / "results.json"
        if f.exists():
            res = json.loads(f.read_text())
            rows.append((n, res["final_avg_miou"], res.get("avg_forgetting"), res.get("avg_bwt"), res["final_miou"]))
    if rows:
        doms = list(rows[0][4])
        lines = ["| run | " + " | ".join(doms) + " | avg mIoU | forgetting | BWT |",
                 "|---|" + "---|" * (len(doms) + 3)]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        for n, avg, fg, bwt, per in sorted(rows, key=lambda r: -r[1]):
            lines.append(f"| {n} | " + " | ".join(fmt(per[d]) for d in doms) + f" | {fmt(avg)} | {fmt(fg)} | {fmt(bwt)} |")
        (root / "ablation_table.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
