"""Exploratory data analysis per domain/split plus cross-domain shift plots.
Descriptive only: nothing computed here feeds training."""
from collections import Counter
from pathlib import Path
import csv
import json
import random
import numpy as np
from PIL import Image

from .dataset import build_lut, load_mask


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def colorize(mask, colors, ignore):
    pal = np.zeros((256, 3), dtype=np.uint8)
    pal[: len(colors)] = np.array(colors, dtype=np.uint8)
    pal[ignore] = (0, 0, 0)
    return pal[mask]


def analyse_split(rows, spec, cfg, max_images, rng):
    C, ign = len(cfg["classes"]), cfg["ignore_index"]
    lut = build_lut(spec, C, ign)
    rows = rows if not max_images or len(rows) <= max_images else rng.sample(rows, max_images)
    pix = np.zeros(C + 1, dtype=np.int64)  # last bin = ignore
    presence = np.zeros(C, dtype=np.int64)
    raw_counts, sizes, corrupt = Counter(), Counter(), []
    rgb_sum, rgb_sq, n_rgb = np.zeros(3), np.zeros(3), 0
    bright_hist = np.zeros(32, dtype=np.int64)
    labelled = 0
    for r in rows:
        try:
            img = Image.open(r.image).convert("RGB")
            sizes[f"{img.size[1]}x{img.size[0]}"] += 1
            small = np.asarray(img.resize((max(1, img.size[0] // 8), max(1, img.size[1] // 8))), dtype=np.float64) / 255
            flat = small.reshape(-1, 3)
            rgb_sum += flat.sum(0)
            rgb_sq += (flat ** 2).sum(0)
            n_rgb += len(flat)
            bright_hist += np.histogram(flat.mean(1), bins=32, range=(0, 1))[0]
            if r.mask:
                raw = np.asarray(Image.open(r.mask))
                raw = raw[..., 0] if raw.ndim == 3 and not spec.get("rgb_label_map") else raw
                if not spec.get("rgb_label_map"):
                    vals, cnts = np.unique(raw, return_counts=True)
                    raw_counts.update(dict(zip(vals.tolist(), cnts.tolist())))
                m = load_mask(r.mask, spec, lut, ign)
                if m.shape != (img.size[1], img.size[0]):
                    raise ValueError("image/mask size mismatch")
                cnt = np.bincount(np.where(m == ign, C, m).ravel(), minlength=C + 1)
                pix += cnt
                presence += cnt[:C] > 0
                labelled += 1
        except Exception as e:  # report, keep scanning
            corrupt.append({"image": r.image, "mask": r.mask, "error": repr(e)})
    mean = rgb_sum / max(n_rgb, 1)
    std = np.sqrt(np.maximum(rgb_sq / max(n_rgb, 1) - mean ** 2, 0))
    total = max(pix[:C].sum(), 1)
    lmap = spec.get("label_map") or {}
    unmapped = {int(k): int(v) for k, v in raw_counts.items() if lmap and int(k) not in {int(a) for a in lmap}}
    return {
        "images_scanned": len(rows), "labelled_scanned": labelled,
        "groups": len({r.group for r in rows}),
        "sizes": dict(sizes.most_common()),
        "rgb_mean": mean.round(4).tolist(), "rgb_std": std.round(4).tolist(),
        "brightness_hist": bright_hist.tolist(),
        "class_pixel_fraction": {c: float(pix[i] / total) for i, c in enumerate(cfg["classes"])},
        "class_image_presence": {c: int(presence[i]) for i, c in enumerate(cfg["classes"])},
        "ignore_pixel_fraction": float(pix[C] / max(pix.sum(), 1)),
        "raw_label_values": {int(k): int(v) for k, v in sorted(raw_counts.items())},
        "raw_values_mapped_to_ignore": unmapped,
        "corrupt": corrupt,
    }


def _examples(rows, spec, cfg, out, n, rng):
    plt = _plt()
    rows = [r for r in rows if r.mask][:200]
    rows = rng.sample(rows, min(n, len(rows)))
    if not rows:
        return
    lut = build_lut(spec, len(cfg["classes"]), cfg["ignore_index"])
    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 2.4 * len(rows)), squeeze=False)
    for ax, r in zip(axes, rows):
        img = Image.open(r.image).convert("RGB")
        m = load_mask(r.mask, spec, lut, cfg["ignore_index"])
        ax[0].imshow(img); ax[1].imshow(colorize(m, cfg["colors"], cfg["ignore_index"]))
        ax[0].set_title(Path(r.image).name, fontsize=7); ax[1].set_title("mapped GT", fontsize=7)
        for a in ax:
            a.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def run_eda(cfg, all_splits, out_dir):
    plt = _plt()
    out_dir = Path(out_dir) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    e = cfg["eda"]
    rng = random.Random(cfg["seed"])
    summary = {}
    for dom, splits in all_splits.items():
        spec = cfg["domains"][dom]
        for split, rows in splits.items():
            print(f"[eda] {dom}/{split}: {len(rows)} images")
            s = analyse_split(rows, spec, cfg, e.get("max_images"), rng)
            s["images_total"] = len(rows)
            summary[f"{dom}/{split}"] = s
            d = out_dir / dom / split
            d.mkdir(parents=True, exist_ok=True)
            (d / "stats.json").write_text(json.dumps(s, indent=2))
            with (d / "class_stats.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["class", "pixel_fraction", "images_present"])
                for c in cfg["classes"]:
                    w.writerow([c, s["class_pixel_fraction"][c], s["class_image_presence"][c]])
            if s["corrupt"]:
                print(f"[eda] WARNING {dom}/{split}: {len(s['corrupt'])} unreadable/mismatched samples (see stats.json)")
            if s["raw_values_mapped_to_ignore"]:
                print(f"[eda] {dom}/{split}: raw label values mapped to ignore: {sorted(s['raw_values_mapped_to_ignore'])}")
            if e.get("example_images", 0):
                _examples(rows, spec, cfg, d / "examples.png", e["example_images"], rng)
    # Cross-domain comparison: class distribution + brightness (domain shift at a glance).
    keys = [k for k in summary if k.endswith("/train")] or list(summary)
    if keys:
        C = cfg["classes"]
        fig, ax = plt.subplots(figsize=(max(8, len(C) * 0.6), 4))
        wdt = 0.8 / len(keys)
        for i, k in enumerate(keys):
            ax.bar(np.arange(len(C)) + i * wdt, [summary[k]["class_pixel_fraction"][c] for c in C], wdt, label=k)
        ax.set_xticks(np.arange(len(C)) + 0.4 - wdt / 2); ax.set_xticklabels(C, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("pixel fraction"); ax.set_yscale("log"); ax.legend(fontsize=8); ax.set_title("Class distribution per domain")
        fig.tight_layout(); fig.savefig(out_dir / "class_distribution_by_domain.png", dpi=120); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in keys:
            h = np.array(summary[k]["brightness_hist"], dtype=float)
            ax.plot(np.linspace(0, 1, 32), h / max(h.sum(), 1), label=k)
        ax.set_xlabel("pixel brightness"); ax.set_ylabel("density"); ax.legend(fontsize=8); ax.set_title("Brightness distribution per domain")
        fig.tight_layout(); fig.savefig(out_dir / "brightness_by_domain.png", dpi=120); plt.close(fig)
    lean = {k: {kk: vv for kk, vv in v.items() if kk not in ("corrupt", "brightness_hist", "raw_label_values")}
            for k, v in summary.items()}
    (out_dir / "summary.json").write_text(json.dumps(lean, indent=2))
    lines = ["| domain/split | images | groups | sizes | ignore % | RGB mean |", "|---|---|---|---|---|---|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {v['images_total']} | {v['groups']} | {', '.join(list(v['sizes'])[:2])} | "
                     f"{100 * v['ignore_pixel_fraction']:.1f} | {v['rgb_mean']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return str(out_dir)
