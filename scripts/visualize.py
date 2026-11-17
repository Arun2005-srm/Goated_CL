"""Research figures for a finished run. Part A redraws every training curve from the run's own
records (history.csv, per_class_val_history.csv, teachers_summary.json, results.json). Part B
reloads every saved checkpoint (the student after each stage and every teacher), runs it on
each domain's test split and draws confusion matrices, per-class IoU across stages, class-level
forgetting, calibration, confidence, model agreement, t-SNE of pixel and image features and
qualitative predictions.

Run from the repo root, after training, with the same data paths the run used:

    python scripts/visualize.py --run runs/driving_dil/goated_full
    python scripts/visualize.py --run runs/driving_dil/goated_full --no-inference      # curves only
    python scripts/visualize.py --run runs/smoke/goated_full --max-images 40 --formats png

Writes <run>/figures/ (PNG + PDF) and <run>/figures/tables/ (CSV, Markdown, LaTeX).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goated.config import build_stages, teacher_domains  # noqa: E402
from goated.data.dataset import make_loader, maybe_cache  # noqa: E402
from goated.data.discovery import discover_all  # noqa: E402
from goated.data.eda import colorize  # noqa: E402
from goated.engine.common import amp_ctx, boundary_stats, confusion, get_device, metrics_from_confusion  # noqa: E402
from goated.engine.teachers import load_teacher, teacher_path  # noqa: E402
from goated.models.segmenter import Segmenter  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 7,
                     "savefig.bbox": "tight"})
PAL = plt.get_cmap("tab10").colors
CAL_BINS, ENT_BINS = 15, 20


class Saver:
    def __init__(self, out, formats):
        self.out, self.formats = Path(out), formats
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "tables").mkdir(exist_ok=True)

    def fig(self, fig, name):
        for ext in self.formats:
            fig.savefig(self.out / f"{name}.{ext}", dpi=200)
        plt.close(fig)
        print(f"[viz] {name}")

    def table(self, name, rows, fields=None):
        if not rows:
            return
        fields = fields or list(rows[0])
        with (self.out / "tables" / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def dom_color(domains):
    return {d: PAL[i % len(PAL)] for i, d in enumerate(domains)}


# ============================================================================ Part A: training records
def shade_stages(ax, stages_of_x):
    """Alternate grey bands per stage and label each stage above the axes."""
    spans, start = [], 0
    for i in range(1, len(stages_of_x) + 1):
        if i == len(stages_of_x) or stages_of_x[i] != stages_of_x[start]:
            spans.append((stages_of_x[start], start + 1, i))
            start = i
    for k, (name, a, b) in enumerate(spans):
        if k % 2:
            ax.axvspan(a - 0.5, b + 0.5, color="0.93", zorder=0, lw=0)
        ax.text((a + b) / 2, 1.01, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)


def teacher_curves(sv, cfg, run):
    path = run / "teachers_summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    logs = {}
    for name in dict.fromkeys(s["teacher"] for s in build_stages(cfg)):
        log = (summary.get(name) or {}).get("log")
        if not log and teacher_path(cfg, name).exists():
            log = torch.load(teacher_path(cfg, name), map_location="cpu", weights_only=False).get("log")
        if log:
            logs[name] = log
    if not logs:
        print("[viz] no teacher training logs found; skipping teacher curves")
        return
    panels = [("Loss", [("train_loss", "train", "-"), ("val_loss", "val", "--")]),
              ("Pixel accuracy (%)", [("train_acc", "train", "-"), ("val_acc", "val", "--")]),
              ("Val mIoU (%)", [("val_miou", "mIoU", "-")]),
              ("Val boundary IoU (%)", [("val_biou", "bIoU", "-")])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (title, keys) in zip(axes, panels):
        for i, (name, log) in enumerate(logs.items()):
            ep = [r["epoch"] for r in log]
            for k, lab, ls in keys:
                ax.plot(ep, [num(r.get(k)) for r in log], ls=ls, color=PAL[i], label=f"{name} {lab}")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle("Teacher training", y=1.04)
    sv.fig(fig, "A1_teacher_training_overview")
    for i, (name, log) in enumerate(logs.items()):
        ep = [r["epoch"] for r in log]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        axes[0].plot(ep, [num(r["train_loss"]) for r in log], label="train")
        axes[0].plot(ep, [num(r.get("val_loss")) for r in log], "--", label="val")
        axes[0].set_title("Loss")
        axes[1].plot(ep, [num(r["train_acc"]) for r in log], label="train")
        axes[1].plot(ep, [num(r.get("val_acc")) for r in log], "--", label="val")
        axes[1].set_title("Pixel accuracy (%)")
        for k in ("val_miou", "val_mf1", "val_biou"):
            axes[2].plot(ep, [num(r.get(k)) for r in log], label=k[4:])
        axes[2].set_title("Validation metrics (%)")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.legend()
        fig.suptitle(f"Teacher {name}", y=1.04)
        sv.fig(fig, f"A2_teacher_{name}_curves")
    sv.table("teacher_training_log.csv", [{"teacher": n, **r} for n, log in logs.items() for r in log])


def student_curves(sv, cfg, run):
    hist = read_csv(run / "history.csv")
    if not hist:
        print("[viz] no history.csv; skipping student curves")
        return
    x = np.arange(1, len(hist) + 1)
    st = [h["stage"] for h in hist]
    col = lambda k: np.array([num(h.get(k)) for h in hist])
    domains = list(cfg["domains"])
    dc = dom_color(domains)

    def panel(ax, title, series, log=False, ylabel=None):
        for k, lab, kw in series:
            y = col(k)
            if np.isfinite(y).any():
                ax.plot(x, y, marker=".", ms=3, label=lab, **kw)
        shade_stages(ax, st)
        ax.set_xlim(0.5, len(hist) + 0.5)
        ax.set_title(title, pad=14)
        ax.set_xlabel("distillation epoch (all stages)")
        if ylabel:
            ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.legend()

    terms = sorted(k for k in hist[0] if k.startswith("loss_") and k != "loss_total")
    specs = {
        "B1_student_loss": ("Train vs validation loss", [("train_loss", "train", {}), ("val_loss", "val", {"ls": "--"})], False, "loss"),
        "B2_student_accuracy": ("Train vs validation pixel accuracy",
                                [("train_acc", "train acc (vs GT)", {}), ("val_acc", "val acc", {"ls": "--"}),
                                 ("teacher_agreement", "teacher agreement", {"ls": ":"})], False, "%"),
        "B3_student_val_metrics": ("Validation mIoU / mF1 / boundary IoU (current domains)",
                                   [("val_miou", "mIoU", {}), ("val_mf1", "mF1", {}), ("val_biou", "bIoU", {})], False, "%"),
        "B4_student_seen_domains": ("Validation mIoU per seen domain",
                                    [(f"val_miou_{d}", d, {"color": dc[d]}) for d in domains]
                                    + [("seen_avg_miou", "average", {"color": "k", "ls": "--"})], False, "mIoU (%)"),
        "B5_student_forgetting_bwt": ("Forgetting and backward transfer",
                                      [("forgetting", "forgetting", {}), ("bwt", "BWT", {})], False, "mIoU points"),
        "B6_student_loss_terms": ("Distillation loss terms", [(k, k[5:], {}) for k in terms], True, "loss"),
        "B7_student_lr": ("Learning rate", [("lr", "lr", {})], True, None),
    }
    for name, (title, series, log, yl) in specs.items():
        fig, ax = plt.subplots(figsize=(7, 3.6))
        panel(ax, title, series, log, yl)
        sv.fig(fig, name)
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, name in zip(axes.flat, list(specs)[:6]):
        title, series, log, yl = specs[name]
        panel(ax, title, series, log, yl)
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    sv.fig(fig, "B0_student_training_overview")


def per_class_heatmap(sv, cfg, run):
    rows = read_csv(run / "per_class_val_history.csv")
    if not rows:
        return
    classes = cfg["classes"]
    keys = list(dict.fromkeys((r["stage"], r["epoch"]) for r in rows))
    M = np.full((len(classes), len(keys)), np.nan)
    ki = {k: i for i, k in enumerate(keys)}
    for r in rows:
        M[classes.index(r["class"]), ki[(r["stage"], r["epoch"])]] = num(r["iou"])
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(keys) + 3), 0.32 * len(classes) + 1.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xticks(range(len(keys)), [f"{s[:4]}{e}" for s, e in keys], rotation=90, fontsize=6)
    for i in range(1, len(keys)):
        if keys[i][0] != keys[i - 1][0]:
            ax.axvline(i - 0.5, color="w", lw=1.5)
    ax.grid(False)
    ax.set_title("Per-class validation IoU over distillation epochs (current-stage domains)")
    fig.colorbar(im, label="IoU (%)")
    sv.fig(fig, "B8_per_class_val_iou_heatmap")


def cl_summary(sv, cfg, run):
    rp, tp = run / "results.json", run / "teachers_summary.json"
    if not rp.exists():
        return
    res = json.loads(rp.read_text())
    R, domains = res["R"], [d for d in cfg["domains"] if d in res["R"][0]]
    teachers = json.loads(tp.read_text()) if tp.exists() else {}
    rows = [(f"student after {r['stage']}", [r[d] for d in domains]) for r in R]
    rows += [(f"teacher {n}", [v["test_miou"].get(d, np.nan) for d in domains]) for n, v in teachers.items()]
    A = np.array([v for _, v in rows], float)
    fig, ax = plt.subplots(figsize=(1.3 * len(domains) + 4, 0.5 * len(rows) + 1.5))
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=100)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.1f}", ha="center", va="center", color="w" if A[i, j] < 60 else "k", fontsize=8)
    ax.axhline(len(R) - 0.5, color="w", lw=2)
    ax.set_xticks(range(len(domains)), domains)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.grid(False)
    ax.set_title("Test mIoU: student after each stage, and each teacher")
    fig.colorbar(im, label="mIoU (%)")
    sv.fig(fig, "C1_test_miou_matrix")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for j, d in enumerate(domains):
        ax.plot(range(1, len(R) + 1), A[: len(R), j], marker="o", color=PAL[j], label=d)
        ub = res.get("teacher_upper_bound", {}).get(d)
        if ub is not None:
            ax.axhline(ub, color=PAL[j], ls=":", lw=1)
    ax.plot(range(1, len(R) + 1), np.nanmean(A[: len(R)], 1), "k--", marker="s", label="average")
    ax.set_xticks(range(1, len(R) + 1), [f"after {r['stage']}" for r in R])
    ax.set_ylabel("test mIoU (%)")
    ax.set_title("Per-domain test mIoU across stages (dotted = domain teacher)")
    ax.legend()
    sv.fig(fig, "C2_forgetting_curve")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ub = res.get("teacher_upper_bound", {})
    w, xs = 0.38, np.arange(len(domains))
    axes[0].bar(xs - w / 2, A[len(R) - 1], w, label="final student")
    axes[0].bar(xs + w / 2, [ub.get(d, np.nan) for d in domains], w, label="domain teacher")
    for i, d in enumerate(domains):
        axes[0].text(i - w / 2, A[len(R) - 1, i] + 1, f"{A[len(R) - 1, i]:.1f}", ha="center", fontsize=7)
        if d in ub:
            axes[0].text(i + w / 2, ub[d] + 1, f"{ub[d]:.1f}", ha="center", fontsize=7)
    axes[0].set_xticks(xs, domains)
    axes[0].set_ylabel("test mIoU (%)")
    axes[0].set_title("Final student vs teacher upper bound")
    axes[0].legend()
    fd = [d for d in domains if d in res.get("forgetting", {})]
    xf = np.arange(len(fd))
    axes[1].bar(xf - w / 2, [res["forgetting"][d] for d in fd], w, label="forgetting")
    axes[1].bar(xf + w / 2, [res["bwt"][d] for d in fd], w, label="BWT")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(xf, fd)
    axes[1].set_ylabel("mIoU points")
    axes[1].set_title("Forgetting and BWT per earlier domain")
    axes[1].legend()
    sv.fig(fig, "C3_student_vs_teacher_and_forgetting")


# ============================================================================ Part B: checkpoint inference
class Stats:
    """Everything accumulated for one (model, domain) pass over a test split."""

    def __init__(self, C, quota):
        self.cm = torch.zeros(C, C, dtype=torch.long)
        self.bi = torch.zeros(C, dtype=torch.long)
        self.bu = torch.zeros(C, dtype=torch.long)
        self.loss = 0.0
        self.npix = 0
        self.cal = torch.zeros(3, CAL_BINS, dtype=torch.float64)  # count, sum conf, sum correct
        self.ent = torch.zeros(2, ENT_BINS, dtype=torch.float64)  # correct, wrong
        self.quota, self.have = quota, np.zeros(C, int)
        self.feats, self.flabels, self.img = [], [], []

    def metrics(self, classes):
        m = metrics_from_confusion(self.cm, classes, self.bi, self.bu)
        n, sc, sa = self.cal
        k = n > 0
        m["ece"] = float((n[k] / n.sum() * (sa[k] / n[k] - sc[k] / n[k]).abs()).sum() * 100)
        m["mean_conf"] = float(sc.sum() / n.sum().clamp_min(1) * 100)
        m["loss"] = self.loss / max(self.npix, 1)
        return m


def build_models(cfg, run, device):
    stages = build_stages(cfg)
    C = len(cfg["classes"])
    models = {}
    for i, s in enumerate(stages):
        p = run / f"student_stage_{i + 1:02d}.pt"
        if p.exists():
            m = Segmenter({**cfg["models"]["student"], "pretrained": False}, C)
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            models[f"student@{s['name']}"] = m.to(device).eval()
        else:
            print(f"[viz] missing {p}; skipped")
    for name in dict.fromkeys(s["teacher"] for s in stages):
        if teacher_path(cfg, name).exists():
            models[f"teacher:{name}"] = load_teacher(cfg, name, device)
        else:
            print(f"[viz] missing teacher checkpoint {teacher_path(cfg, name)}; skipped")
    return models


def denorm(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def run_inference(cfg, models, loaders, device, args):
    C, ign = len(cfg["classes"]), cfg["ignore_index"]
    ratio = cfg.get("metrics", {}).get("boundary_ratio", 0.02)
    names = list(models)
    stats = {(m, d): Stats(C, args.tsne_points // C) for m in names for d in loaders}
    agree = {d: torch.zeros(len(names), len(names), dtype=torch.float64) for d in loaders}
    gallery = {d: [] for d in loaders}
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))
    for d, ld in loaders.items():
        n = len(ld.dataset)
        pick = set(np.linspace(0, n - 1, min(args.gallery, n)).round().astype(int).tolist()) if args.gallery else set()
        print(f"[viz] inference on {d}: {n} images, {len(names)} models")
        for b in ld:
            x, y = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
            valid = y != ign
            preds, ents = {}, {}
            for mn in names:
                s = stats[(mn, d)]
                with amp_ctx(cfg["eval"].get("amp", True), device):
                    logits, feats = models[mn](x, return_features=True)
                logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
                p = logits.softmax(1)
                conf, pred = p.max(1)
                ent = -(p * p.clamp_min(1e-12).log()).sum(1) / np.log(C)
                preds[mn], ents[mn] = pred, ent
                s.cm += confusion(pred.flatten(), y.flatten(), C, ign).cpu()
                bi, bu = boundary_stats(pred, y, C, ign, ratio)
                s.bi += bi.cpu()
                s.bu += bu.cpu()
                if valid.any():
                    s.loss += float(F.cross_entropy(logits, y, ignore_index=ign, reduction="sum"))
                    s.npix += int(valid.sum())
                    cv, ok, ev = conf[valid].double(), (pred[valid] == y[valid]).double(), ent[valid].double()
                    bins = (cv * CAL_BINS).long().clamp_max(CAL_BINS - 1)
                    s.cal[0] += torch.bincount(bins, minlength=CAL_BINS).cpu()
                    s.cal[1] += torch.bincount(bins, cv, minlength=CAL_BINS).cpu()
                    s.cal[2] += torch.bincount(bins, ok, minlength=CAL_BINS).cpu()
                    eb = (ev * ENT_BINS).long().clamp(0, ENT_BINS - 1)
                    s.ent[0] += torch.bincount(eb[ok > 0], minlength=ENT_BINS).cpu()
                    s.ent[1] += torch.bincount(eb[ok == 0], minlength=ENT_BINS).cpu()
                feats = feats.float()
                s.img.append(feats.mean((2, 3)).cpu())
                if not args.no_tsne:
                    ys = F.interpolate(y[:, None].float(), size=feats.shape[-2:], mode="nearest")[:, 0].long()
                    fz = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
                    yz = ys.flatten()
                    per_batch = max(1, s.quota // 8)
                    for c in torch.unique(yz[yz != ign]).tolist():
                        need = min(s.quota - s.have[c], per_batch)
                        if need <= 0:
                            continue
                        idx = (yz == c).nonzero()[:, 0]
                        idx = idx[torch.randperm(len(idx), generator=gen)[:need].to(idx.device)]
                        s.feats.append(fz[idx].cpu())
                        s.flabels.append(torch.full((len(idx),), c))
                        s.have[c] += len(idx)
            for i, a in enumerate(names):
                for j, bb in enumerate(names):
                    agree[d][i, j] += float((preds[a][valid] == preds[bb][valid]).sum())
            for k, idx in enumerate(b["index"].tolist()):
                if idx in pick:
                    gallery[d].append({"image": denorm(b["image"][k]), "gt": b["mask"][k].numpy(),
                                       "pred": {mn: preds[mn][k].cpu().numpy() for mn in names},
                                       "ent": {mn: ents[mn][k].float().cpu().numpy() for mn in names}})
        for i in range(len(names)):  # agreement as % of valid pixels
            tot = agree[d][i, i]
            agree[d][i] = agree[d][i] / max(float(tot), 1) * 100
    return stats, agree, gallery


def own_teacher(cfg, d, names):
    for s in build_stages(cfg):
        if d in teacher_domains(cfg, s["teacher"]) and f"teacher:{s['teacher']}" in names:
            return f"teacher:{s['teacher']}"
    return None


def learned_stage(cfg, d):
    for s in build_stages(cfg):
        if d in s["distill_on"] or d in teacher_domains(cfg, s["teacher"]):
            return s["name"]
    return None


def inference_figures(sv, cfg, stats, agree, gallery, names, domains, args):
    classes, C = cfg["classes"], len(cfg["classes"])
    students = [n for n in names if n.startswith("student@")]
    final = students[-1] if students else names[0]
    M = {k: v.metrics(classes) for k, v in stats.items()}
    rows, pcs = [], []
    for (mn, d), m in M.items():
        rows.append({"model": mn, "domain": d, **{k: round(m[k], 3) for k in
                     ("miou", "mf1", "pixel_acc", "biou", "ece", "mean_conf", "loss") if m.get(k) is not None}})
        for c in classes:
            pcs.append({"model": mn, "domain": d, "class": c, "iou": m["per_class_iou"][c],
                        "f1": m["per_class_f1"][c], "biou": m.get("per_class_biou", {}).get(c)})
    sv.table("test_metrics_all_models.csv", rows)
    sv.table("test_per_class_all_models.csv", pcs)
    md = ["| model | " + " | ".join(domains) + " | avg |", "|---|" + "---|" * (len(domains) + 1)]
    tex = [r"\begin{tabular}{l" + "c" * (len(domains) + 1) + "}", r"\toprule",
           "Model & " + " & ".join(domains) + r" & Avg \\", r"\midrule"]
    for mn in names:
        v = [M[(mn, d)]["miou"] for d in domains]
        md.append(f"| {mn} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {np.mean(v):.2f} |")
        tex.append(mn.replace("_", r"\_") + " & " + " & ".join(f"{x:.2f}" for x in v) + f" & {np.mean(v):.2f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (sv.out / "tables" / "test_miou_table.md").write_text("\n".join(md) + "\n")
    (sv.out / "tables" / "test_miou_table.tex").write_text("\n".join(tex) + "\n")

    # confusion matrices: domain teacher vs final student
    for d in domains:
        pair = [n for n in (own_teacher(cfg, d, names), final) if n]
        fig, axes = plt.subplots(1, len(pair), figsize=(7.2 * len(pair), 6.4), squeeze=False)
        for ax, mn in zip(axes[0], pair):
            cm = stats[(mn, d)].cm.double()
            cmn = (cm / cm.sum(1, keepdim=True).clamp_min(1)).numpy()
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(C):
                for j in range(C):
                    if cmn[i, j] >= 0.05:
                        ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=5,
                                color="w" if cmn[i, j] > 0.6 else "k")
            ax.set_xticks(range(C), classes, rotation=90, fontsize=7)
            ax.set_yticks(range(C), classes, fontsize=7)
            ax.set_xlabel("predicted")
            ax.set_ylabel("ground truth")
            ax.grid(False)
            ax.set_title(f"{mn} on {d} (row-normalised)")
            fig.colorbar(im, ax=ax, fraction=0.046)
        sv.fig(fig, f"D1_confusion_{d}")

    # per-class IoU across stages, with the domain teacher, and class-level forgetting
    for d in domains:
        cols = students + [n for n in [own_teacher(cfg, d, names)] if n]
        A = np.array([[num(M[(mn, d)]["per_class_iou"][c]) for mn in cols] for c in classes])
        fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 4, 0.32 * C + 1.5))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("0.85")
        im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(C):
            for j in range(len(cols)):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=6, color="w" if A[i, j] < 60 else "k")
        ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
        ax.set_yticks(range(C), classes)
        ax.grid(False)
        ax.set_title(f"Per-class test IoU on {d} across stages")
        fig.colorbar(im, label="IoU (%)")
        sv.fig(fig, f"D2_per_class_iou_across_stages_{d}")
        ls = learned_stage(cfg, d)
        if ls and f"student@{ls}" != final and f"student@{ls}" in names:
            a = np.array([num(M[(f"student@{ls}", d)]["per_class_iou"][c]) for c in classes])
            f_ = np.array([num(M[(final, d)]["per_class_iou"][c]) for c in classes])
            delta = f_ - a
            order = np.argsort(np.nan_to_num(delta, nan=0))
            fig, ax = plt.subplots(figsize=(8, 0.28 * C + 1.2))
            ax.barh(range(C), delta[order], color=["#c0392b" if v < 0 else "#27ae60" for v in delta[order]])
            ax.set_yticks(range(C), [classes[i] for i in order])
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel(f"IoU change: {final} minus student@{ls} (points)")
            ax.set_title(f"Class-level forgetting on {d}")
            sv.fig(fig, f"D3_class_forgetting_{d}")
    # final student vs domain teacher per class, every domain in one figure
    fig, axes = plt.subplots(len(domains), 1, figsize=(12, 2.6 * len(domains)), squeeze=False)
    for ax, d in zip(axes[:, 0], domains):
        t = own_teacher(cfg, d, names)
        xs, w = np.arange(C), 0.4
        ax.bar(xs - w / 2, [num(M[(final, d)]["per_class_iou"][c]) for c in classes], w, label=final)
        if t:
            ax.bar(xs + w / 2, [num(M[(t, d)]["per_class_iou"][c]) for c in classes], w, label=t)
        ax.set_xticks(xs, classes, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("IoU (%)")
        ax.set_title(d)
        ax.legend()
    sv.fig(fig, "D4_per_class_iou_student_vs_teacher")

    # calibration and confidence
    fig, axes = plt.subplots(1, len(domains), figsize=(4.2 * len(domains), 4), squeeze=False)
    for ax, d in zip(axes[0], domains):
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        for i, mn in enumerate([final] + [n for n in names if n.startswith("teacher:")]):
            n_, sc, sa = stats[(mn, d)].cal
            k = n_ > 0
            ax.plot((sc[k] / n_[k]).numpy(), (sa[k] / n_[k]).numpy(), marker="o", ms=3, color=PAL[i],
                    label=f"{mn} (ECE {M[(mn, d)]['ece']:.1f})")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Reliability on {d}")
        ax.legend()
    sv.fig(fig, "E1_reliability_diagrams")
    fig, axes = plt.subplots(1, len(domains), figsize=(4.2 * len(domains), 3.4), squeeze=False)
    edges = np.linspace(0, 1, ENT_BINS + 1)
    for ax, d in zip(axes[0], domains):
        ok, wr = stats[(final, d)].ent.numpy()
        ax.stairs(ok / max(ok.sum(), 1), edges, fill=True, alpha=0.5, label="correct pixels")
        ax.stairs(wr / max(wr.sum(), 1), edges, fill=True, alpha=0.5, label="wrong pixels")
        ax.set_xlabel("normalised predictive entropy")
        ax.set_ylabel("fraction of pixels")
        ax.set_title(f"{final} uncertainty on {d}")
        ax.legend()
    sv.fig(fig, "E2_entropy_correct_vs_wrong")
    fig, axes = plt.subplots(1, len(domains), figsize=(4.6 * len(domains), 4.2), squeeze=False)
    for ax, d in zip(axes[0], domains):
        A = agree[d].numpy()
        im = ax.imshow(A, cmap="magma", vmin=max(0, A.min() - 5), vmax=100)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=6, color="w" if A[i, j] < 85 else "k")
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.grid(False)
        ax.set_title(f"Prediction agreement on {d} (%)")
    sv.fig(fig, "E3_model_agreement")
    return M, final


def tsne_figures(sv, cfg, stats, names, domains, args):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    classes, colors = cfg["classes"], np.array(cfg["colors"]) / 255.0
    dc = dom_color(domains)
    seed = cfg.get("seed", 0)
    models = names if args.tsne_models == "all" else [n for n in names if n.startswith("student@")][-1:] + \
        [n for n in names if n.startswith("teacher:")]
    sil = []

    def embed(X):
        X = PCA(n_components=min(50, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X)
        Z = TSNE(n_components=2, perplexity=min(30, max(5, len(X) // 4)), init="pca", learning_rate="auto",
                 random_state=seed).fit_transform(X)
        return X, Z

    pix, img = {}, {}
    for mn in models:
        X = [torch.cat(stats[(mn, d)].feats) for d in domains if stats[(mn, d)].feats]
        if X:
            ys = np.concatenate([torch.cat(stats[(mn, d)].flabels).numpy() for d in domains if stats[(mn, d)].feats])
            ds = np.concatenate([np.full(sum(len(f) for f in stats[(mn, d)].feats), i)
                                 for i, d in enumerate(domains) if stats[(mn, d)].feats])
            print(f"[viz] t-SNE of {len(ys)} pixel features for {mn}")
            P, Z = embed(torch.cat(X).numpy())
            pix[mn] = (Z, ys, ds)
            k = np.random.default_rng(seed).permutation(len(P))[:5000]
            sil.append({"model": mn, "features": "pixel",
                        "class_silhouette": round(float(silhouette_score(P[k], ys[k])), 4) if len(set(ys[k])) > 1 else None,
                        "domain_silhouette": round(float(silhouette_score(P[k], ds[k])), 4) if len(set(ds[k])) > 1 else None})
        Xi = [torch.cat(stats[(mn, d)].img) for d in domains]
        di = np.concatenate([np.full(len(x), i) for i, x in enumerate(Xi)])
        if len(di) > 10:
            print(f"[viz] t-SNE of {len(di)} image features for {mn}")
            P, Z = embed(torch.cat(Xi).numpy())
            img[mn] = (Z, di)
            sil.append({"model": mn, "features": "image", "class_silhouette": None,
                        "domain_silhouette": round(float(silhouette_score(P, di)), 4) if len(set(di)) > 1 else None})
    sv.table("feature_silhouette.csv", sil)

    def legend_classes(ax, present):
        h = [plt.Line2D([], [], marker="o", ls="", color=colors[c], label=classes[c]) for c in sorted(present)]
        ax.legend(handles=h, fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1, 0.5))

    def legend_domains(ax):
        h = [plt.Line2D([], [], marker="o", ls="", color=dc[d], label=d) for d in domains]
        ax.legend(handles=h, loc="best")

    for mn, (Z, ys, ds) in pix.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        axes[0].scatter(Z[:, 0], Z[:, 1], c=colors[ys], s=3, lw=0)
        axes[0].set_title(f"{mn}: pixel features by class")
        legend_classes(axes[0], set(ys.tolist()))
        axes[1].scatter(Z[:, 0], Z[:, 1], c=[dc[domains[i]] for i in ds], s=3, lw=0, alpha=0.7)
        axes[1].set_title(f"{mn}: pixel features by domain")
        legend_domains(axes[1])
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        sv.fig(fig, f"F1_tsne_pixels_{mn.replace(':', '_').replace('@', '_')}")
    studs = [n for n in pix if n.startswith("student@")]
    if len(studs) > 1:
        fig, axes = plt.subplots(2, len(studs), figsize=(4.6 * len(studs), 9), squeeze=False)
        for j, mn in enumerate(studs):
            Z, ys, ds = pix[mn]
            axes[0, j].scatter(Z[:, 0], Z[:, 1], c=colors[ys], s=2, lw=0)
            axes[1, j].scatter(Z[:, 0], Z[:, 1], c=[dc[domains[i]] for i in ds], s=2, lw=0, alpha=0.7)
            axes[0, j].set_title(f"{mn}\nby class")
            axes[1, j].set_title("by domain")
        legend_domains(axes[1, -1])
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        sv.fig(fig, "F2_tsne_pixels_student_across_stages")
    if img:
        fig, axes = plt.subplots(1, len(img), figsize=(4.2 * len(img), 4.2), squeeze=False)
        for ax, (mn, (Z, di)) in zip(axes[0], img.items()):
            ax.scatter(Z[:, 0], Z[:, 1], c=[dc[domains[i]] for i in di], s=6, lw=0, alpha=0.8)
            ax.set_title(mn, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        legend_domains(axes[0, -1])
        fig.suptitle("t-SNE of image-level (pooled decoder) features, coloured by domain", y=1.03)
        sv.fig(fig, "F3_tsne_images_by_domain")


def gallery_figures(sv, cfg, gallery, names, domains, final):
    classes, ign = cfg["classes"], cfg["ignore_index"]
    students = [n for n in names if n.startswith("student@")]
    for d in domains:
        items = gallery[d]
        if not items:
            continue
        t = own_teacher(cfg, d, names)
        cols = ["image", "ground truth"] + ([t] if t else []) + students + ["error (final)", "entropy (final)"]
        fig, axes = plt.subplots(len(items), len(cols), figsize=(2.6 * len(cols), 1.5 * len(items) + 0.6), squeeze=False)
        for r, it in enumerate(items):
            gt = it["gt"].astype(np.uint8)
            panels = [it["image"], colorize(gt, cfg["colors"], ign)]
            panels += [colorize(it["pred"][m].astype(np.uint8), cfg["colors"], ign) for m in ([t] if t else []) + students]
            err = np.zeros(gt.shape + (3,))
            wrong = (it["pred"][final] != gt) & (gt != ign)
            err[gt != ign] = 0.85
            err[wrong] = (0.85, 0.15, 0.15)
            panels.append(err)
            for c, (ax, im) in enumerate(zip(axes[r], panels)):
                ax.imshow(im)
                ax.axis("off")
                if r == 0:
                    ax.set_title(cols[c], fontsize=7)
            axes[r, -1].imshow(it["ent"][final], cmap="inferno", vmin=0, vmax=1)
            axes[r, -1].axis("off")
            if r == 0:
                axes[r, -1].set_title(cols[-1], fontsize=7)
        sv.fig(fig, f"G1_predictions_{d}")
    h = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in cfg["colors"]]
    fig = plt.figure(figsize=(8, 1.2))
    fig.legend(h, classes, ncol=7, loc="center", fontsize=7, frameon=False)
    sv.fig(fig, "G0_class_colour_legend")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True, help="output_dir of a finished run (holds resolved_config.yaml)")
    p.add_argument("--out", help="default: <run>/figures")
    p.add_argument("--split", default="test", help="split used for inference (test or val)")
    p.add_argument("--max-images", type=int, help="cap images per domain (evenly spaced) for faster inference")
    p.add_argument("--batch-size", type=int, help="inference batch size (default: eval.batch_size of the run)")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--gallery", type=int, default=4, help="qualitative examples per domain")
    p.add_argument("--tsne-points", type=int, default=1900, help="pixel features per model and domain (class-balanced)")
    p.add_argument("--tsne-models", choices=["all", "final"], default="all", help="final = final student + teachers")
    p.add_argument("--no-tsne", action="store_true")
    p.add_argument("--no-inference", action="store_true", help="only redraw curves from the run's records")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--device", help="cuda / cpu (default: the run's device setting)")
    a = p.parse_args()

    run = Path(a.run)
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    if a.device:
        cfg["device"] = a.device
    if a.num_workers is not None:
        cfg["loader"]["num_workers"] = a.num_workers
    sv = Saver(a.out or run / "figures", a.formats)

    teacher_curves(sv, cfg, run)
    student_curves(sv, cfg, run)
    per_class_heatmap(sv, cfg, run)
    cl_summary(sv, cfg, run)
    if a.no_inference:
        print(f"[viz] done: {sv.out}")
        return

    device = get_device(cfg)
    splits = maybe_cache(cfg, discover_all(cfg))
    doms = cfg["protocol"].get("eval_on") or list(cfg["domains"])
    loaders = {}
    for d in doms:
        rows = [r for r in splits[d].get(a.split, []) if r.mask]
        if a.max_images and len(rows) > a.max_images:
            rows = [rows[i] for i in np.linspace(0, len(rows) - 1, a.max_images).round().astype(int)]
        if rows:
            loaders[d] = make_loader(cfg, rows, "eval", shuffle=False, batch_size=a.batch_size or cfg["eval"].get("batch_size"))
    models = build_models(cfg, run, device)
    if not models or not loaders:
        print("[viz] no checkpoints or no labelled split to run on; done")
        return
    names, domains = list(models), list(loaders)
    print(f"[viz] device {device}; models: {', '.join(names)}")
    stats, agree, gallery = run_inference(cfg, models, loaders, device, a)
    M, final = inference_figures(sv, cfg, stats, agree, gallery, names, domains, a)
    gallery_figures(sv, cfg, gallery, names, domains, final)
    if not a.no_tsne:
        tsne_figures(sv, cfg, stats, names, domains, a)
    print(f"[viz] done: {sv.out}")


if __name__ == "__main__":
    main()
