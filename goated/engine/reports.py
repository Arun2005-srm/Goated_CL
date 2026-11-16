"""Continual-learning metrics (accuracy matrix, forgetting, BWT) and plots."""
from pathlib import Path
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F

from ..config import teacher_domains
from ..data.eda import colorize
from .common import write_json


def cl_metrics(R, stages, domains, cfg):
    """R[i][d]: mIoU on domain d after stage i. A domain counts as 'learned' at the first stage
    whose teacher knows it or whose distillation data contains it."""
    A = np.array([[r[d] for d in domains] for r in R], dtype=float)
    first = {}
    for i, s in enumerate(stages[: len(R)]):
        for d in set(s["distill_on"]) | set(teacher_domains(cfg, s["teacher"])):
            first.setdefault(d, i)
    last = len(R) - 1
    forg, bwt = {}, {}
    for j, d in enumerate(domains):
        if d in first and first[d] < last:
            forg[d] = float(np.nanmax(A[first[d]:last, j]) - A[last, j])
            bwt[d] = float(A[last, j] - A[first[d], j])
    return {
        "final_miou": {d: float(A[last, j]) for j, d in enumerate(domains)},
        "final_avg_miou": float(np.nanmean(A[last])),
        "avg_miou_per_stage": [float(np.nanmean(a)) for a in A],
        "forgetting": forg, "avg_forgetting": float(np.mean(list(forg.values()))) if forg else None,
        "bwt": bwt, "avg_bwt": float(np.mean(list(bwt.values()))) if bwt else None,
        "first_learned_stage": {d: stages[i]["name"] for d, i in first.items()},
    }


def _plots(out, R, domains, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = np.array([[r[d] for d in domains] for r in R], dtype=float)
    names = [r["stage"] for r in R]
    fig, ax = plt.subplots(figsize=(1.2 * len(domains) + 3, 0.7 * len(names) + 2))
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=max(1, np.nanmax(A)))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.1f}", ha="center", va="center", color="w", fontsize=9)
    ax.set_xticks(range(len(domains))); ax.set_xticklabels(domains)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([f"after {n}" for n in names])
    ax.set_title("mIoU matrix R[stage, domain]"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(out / "miou_matrix.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, d in enumerate(domains):
        ax.plot(range(1, len(names) + 1), A[:, j], marker="o", label=d)
    ax.plot(range(1, len(names) + 1), np.nanmean(A, 1), "k--", marker="s", label="average")
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names)
    ax.set_ylabel("mIoU"); ax.legend(); ax.set_title("Per-domain mIoU across stages (forgetting curve)")
    fig.tight_layout(); fig.savefig(out / "forgetting_curve.png", dpi=120); plt.close(fig)
    keys = sorted({k for h in history for k in h if k not in ("stage", "epoch", "sec")})
    if keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in keys:
            ax.plot([h.get(k, np.nan) for h in history], marker=".", label=k)
        ax.set_xlabel("epoch (all stages)"); ax.set_ylabel("loss"); ax.set_yscale("log"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "loss_curves.png", dpi=120); plt.close(fig)


@torch.no_grad()
def _predictions(out, model, eval_loaders, cfg, device, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    for d, ld in eval_loaders.items():
        ds = ld.dataset
        idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).round().astype(int)
        fig, axes = plt.subplots(len(idx), 3, figsize=(10, 2.4 * len(idx)), squeeze=False)
        for ax, i in zip(axes, idx):
            s = ds[int(i)]
            img, mask = ds.raw(int(i))
            logits = model(s["image"][None].to(device)).float()
            pred = F.interpolate(logits, size=s["mask"].shape, mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
            ax[0].imshow(img.resize((pred.shape[1], pred.shape[0])))
            ax[1].imshow(colorize(s["mask"].numpy().astype(np.uint8), cfg["colors"], cfg["ignore_index"]))
            ax[2].imshow(colorize(pred.astype(np.uint8), cfg["colors"], cfg["ignore_index"]))
            for a, t in zip(ax, ("image", "ground truth", "student")):
                a.set_title(t, fontsize=8); a.axis("off")
        fig.tight_layout(); fig.savefig(out / f"predictions_{d}.png", dpi=100); plt.close(fig)


def write_reports(cfg, R, history, stages, model, eval_loaders, device):
    out = Path(cfg["output_dir"])
    domains = list(eval_loaders)
    m = cl_metrics(R, stages, domains, cfg)
    teachers = {}
    tpath = out / "teachers_summary.json"
    if tpath.exists():
        teachers = {k: v["test_miou"] for k, v in json.loads(tpath.read_text()).items()}
    m["teacher_upper_bound"] = {d: teachers[d][d] for d in domains if d in teachers and d in teachers[d]}
    write_json(out / "results.json", {"R": R, **m, "components": cfg["components"]})
    with (out / "miou_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["after_stage", *domains, "avg"])
        for r in R:
            w.writerow([r["stage"], *[f"{r[d]:.2f}" for d in domains], f"{np.nanmean([r[d] for d in domains]):.2f}"])
    with (out / "history.csv").open("w", newline="") as f:
        keys = sorted({k for h in history for k in h})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
    on = [k for k, v in cfg["components"].items() if v]
    lines = [f"# Results: {out.name}", "", f"Components on: {', '.join(on)}", "",
             "| after stage | " + " | ".join(domains) + " | avg |", "|---|" + "---|" * (len(domains) + 1)]
    for r in R:
        lines.append(f"| {r['stage']} | " + " | ".join(f"{r[d]:.2f}" for d in domains) +
                     f" | {np.nanmean([r[d] for d in domains]):.2f} |")
    if m["teacher_upper_bound"]:
        lines.append("| *teacher (own domain)* | " + " | ".join(
            f"{m['teacher_upper_bound'][d]:.2f}" if d in m["teacher_upper_bound"] else "-" for d in domains) + " | |")
    lines += ["", f"- Final average mIoU: **{m['final_avg_miou']:.2f}**",
              f"- Average forgetting: {m['avg_forgetting']:.2f}" if m["avg_forgetting"] is not None else "- Average forgetting: n/a",
              f"- Average BWT: {m['avg_bwt']:.2f}" if m["avg_bwt"] is not None else "- Average BWT: n/a"]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    _plots(out, R, domains, history)
    if cfg["reporting"].get("num_predictions"):
        _predictions(out, model, eval_loaders, cfg, device, cfg["reporting"]["num_predictions"])
    print("\n".join(lines))
    return m
