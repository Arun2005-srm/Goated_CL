"""Dataset-agnostic discovery of image/mask pairs (glob templates or CSV manifests),
seeded group-aware hold-out splits and cross-split leakage checks."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import random


@dataclass(frozen=True)
class Record:
    image: str
    mask: str | None
    group: str
    sample_id: str
    domain: str


def _from_manifest(root, spec, domain):
    rows = []
    with (root / spec["manifest"]).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            img = str((root / r["image"]).resolve())
            mask = str((root / r["mask"]).resolve()) if r.get("mask") else None
            rows.append(Record(img, mask, r.get("group", "") or "", r.get("id") or r["image"], domain))
    return rows


def _from_glob(root, spec, domain, group_by):
    base = root / spec.get("image_base", "")
    rows = []
    for image in sorted(root.glob(spec["images"])):
        rel = image.relative_to(base)
        stem = image.stem
        suffix = spec.get("strip_suffix", "")
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        mask = spec.get("masks")
        if mask:
            mask = str((root / mask.format(relative=rel.as_posix(), parent=rel.parent.as_posix(),
                                           stem=stem, name=image.name)).resolve())
        group = rel.parent.as_posix() if group_by == "parent" else image.relative_to(root).as_posix()
        rows.append(Record(str(image.resolve()), mask, group, image.relative_to(root).as_posix(), domain))
    return rows


def discover_domain(name, spec, seed, max_per_split=None):
    """Return {split_name: [Record]} for one domain."""
    if not spec.get("root") or not Path(spec["root"]).is_dir():
        raise ValueError(f"Domain '{name}': set root to an existing directory (got {spec.get('root')!r})")
    root = Path(spec["root"]).resolve()
    group_by = spec.get("group_by", "parent")
    splits = {}
    for split, sspec in spec["splits"].items():
        if sspec is None:
            continue
        rows = _from_manifest(root, sspec, name) if "manifest" in sspec else _from_glob(root, sspec, name, group_by)
        missing = [r for r in rows if r.mask and not Path(r.mask).is_file()]
        if missing:
            policy = spec.get("missing_masks", "error")
            if policy == "error":
                raise FileNotFoundError(f"{name}/{split}: {len(missing)} masks missing, e.g. {missing[0].mask}. "
                                        f"Fix the template or set missing_masks: drop")
            missing = set(missing)
            rows = [r for r in rows if r not in missing]
        if not rows:
            raise ValueError(f"Domain '{name}' split '{split}' matched no images under {root}")
        splits[split] = rows
    hold = spec.get("holdout")
    if hold:
        src = splits.pop(hold["source"])
        key = (lambda r: r.group) if hold.get("by_group", True) else (lambda r: r.sample_id)
        groups = sorted({key(r) for r in src})
        random.Random(seed).shuffle(groups)
        n_hold = max(1, round(len(groups) * hold["ratio"]))
        if n_hold >= len(groups):
            raise ValueError(f"Domain '{name}': hold-out would consume every group")
        held = set(groups[:n_hold])
        splits[hold["target"]] = [r for r in src if key(r) in held]
        splits[hold["source"]] = [r for r in src if key(r) not in held]
    if max_per_split:
        rng = random.Random(seed)
        splits = {k: (sorted(rng.sample(v, max_per_split), key=lambda r: r.sample_id) if len(v) > max_per_split else v)
                  for k, v in splits.items()}
    seen = {}
    for split, rows in splits.items():
        for r in rows:
            for v in (r.image, r.mask):
                if v and seen.setdefault(v, split) != split:
                    raise ValueError(f"Split leakage in '{name}': {v} is in {seen[v]} and {split}")
    return splits


def discover_all(cfg):
    lim = cfg["data"].get("max_images_per_split")
    return {name: discover_domain(name, spec, cfg["seed"], lim) for name, spec in cfg["domains"].items()}


def save_manifests(all_splits, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dom, splits in all_splits.items():
        for split, rows in splits.items():
            with (out_dir / f"{dom}_{split}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
                w.writeheader()
                w.writerows(asdict(r) for r in rows)
