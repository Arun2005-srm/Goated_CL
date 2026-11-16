"""Multi-domain segmentation dataset with per-domain label mapping and optional on-disk resize cache."""
from dataclasses import replace
from pathlib import Path
import hashlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .transforms import Compose


def build_lut(spec, num_classes, ignore):
    """256-entry lookup table: raw mask value -> class index (or ignore)."""
    lmap = spec.get("label_map") or {}
    if lmap:
        lut = np.full(256, ignore, dtype=np.uint8)
        for raw, tgt in lmap.items():
            lut[int(raw)] = tgt
    else:  # masks already hold contiguous class ids
        lut = np.arange(256, dtype=np.int64)
        lut[lut >= num_classes] = ignore
        lut = lut.astype(np.uint8)
    lut[ignore] = ignore
    return lut


def load_mask(path, spec, lut, ignore):
    m = Image.open(path)
    rgb_map = spec.get("rgb_label_map") or {}
    if rgb_map:
        arr = np.asarray(m.convert("RGB"), dtype=np.int64)
        code = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
        out = np.full(code.shape, ignore, dtype=np.uint8)
        for key, tgt in rgb_map.items():
            r, g, b = (int(v) for v in str(key).split(","))
            out[code == ((r << 16) | (g << 8) | b)] = tgt
        return out
    arr = np.asarray(m)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max(initial=0) > 255:
        raise ValueError(f"Mask {path} has values > 255; provide an 8-bit mask or an rgb_label_map")
    return lut[arr.astype(np.uint8)]


class SegDataset(Dataset):
    def __init__(self, records, domain_specs, domain_index, transform, num_classes, ignore):
        self.records = list(records)
        self.specs = domain_specs
        self.domain_index = domain_index
        self.transform = transform
        self.ignore = ignore
        self.luts = {d: build_lut(s, num_classes, ignore) for d, s in domain_specs.items()}

    def __len__(self):
        return len(self.records)

    def raw(self, i):
        r = self.records[i]
        img = Image.open(r.image).convert("RGB")
        if r.mask:
            spec = self.specs[r.domain]
            mapped = getattr(r, "sample_id", "").startswith("cache::")
            arr = np.asarray(Image.open(r.mask)) if mapped else load_mask(r.mask, spec, self.luts[r.domain], self.ignore)
            if arr.shape[:2] != (img.size[1], img.size[0]):
                raise ValueError(f"Image/mask size mismatch: {r.image}")
            mask = Image.fromarray(arr.astype(np.uint8), mode="L")
        else:
            mask = Image.new("L", img.size, self.ignore)
        return img, mask

    def __getitem__(self, i):
        img, mask = self.raw(i)
        x, y = self.transform(img, mask)
        return {"image": x, "mask": y, "domain": self.domain_index[self.records[i].domain], "index": i}


def cache_records(records, spec, lut, size, cache_dir, ignore):
    """Resize images (bilinear) and mapped masks (nearest) once to `size`=[H,W]; return cached records.
    Cached masks already contain class ids (marked by a `cache::` sample-id prefix)."""
    h, w = size
    out = []
    cache_dir = Path(cache_dir)
    for r in records:
        key = hashlib.md5(f"{r.image}|{h}x{w}".encode()).hexdigest()[:16]
        img_p = cache_dir / "images" / f"{key}.jpg"
        mask_p = cache_dir / "masks" / f"{key}.png"
        if not img_p.exists() or (r.mask and not mask_p.exists()):
            img_p.parent.mkdir(parents=True, exist_ok=True)
            mask_p.parent.mkdir(parents=True, exist_ok=True)
            Image.open(r.image).convert("RGB").resize((w, h), Image.BILINEAR).save(img_p, quality=95)
            if r.mask:
                m = load_mask(r.mask, spec, lut, ignore)
                Image.fromarray(m, mode="L").resize((w, h), Image.NEAREST).save(mask_p)
        out.append(replace(r, image=str(img_p), mask=str(mask_p) if r.mask else None,
                           sample_id="cache::" + r.sample_id))
    return out


def maybe_cache(cfg, all_splits):
    pre = cfg["data"].get("cache")
    if not pre or not pre.get("enabled"):
        return all_splits
    out = {}
    for dom, splits in all_splits.items():
        spec = cfg["domains"][dom]
        lut = build_lut(spec, len(cfg["classes"]), cfg["ignore_index"])
        out[dom] = {s: cache_records(rows, spec, lut, pre["size"], Path(pre["dir"]) / dom, cfg["ignore_index"])
                    for s, rows in splits.items()}
        print(f"[cache] {dom}: resized copies under {Path(pre['dir']) / dom}")
    return out


def make_loader(cfg, records, transform_key, shuffle, batch_size=None, drop_last=False):
    names = list(cfg["domains"])
    ds = SegDataset(records, cfg["domains"], {n: i for i, n in enumerate(names)},
                    Compose(cfg["transforms"][transform_key], cfg["ignore_index"]),
                    len(cfg["classes"]), cfg["ignore_index"])
    lc = cfg["loader"]
    return DataLoader(ds, batch_size=batch_size or lc["batch_size"], shuffle=shuffle,
                      num_workers=lc["num_workers"], pin_memory=lc.get("pin_memory", False) and torch.cuda.is_available(),
                      drop_last=drop_last and len(ds) > (batch_size or lc["batch_size"]),
                      persistent_workers=lc["num_workers"] > 0)
