"""Paired image/mask transforms configured by name. Geometry is shared; masks use nearest."""
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Compose:
    def __init__(self, specs, ignore_index):
        self.ops = [_build(dict(s), ignore_index) for s in specs]
        norm = [s for s in specs if s["name"] == "normalize"]
        self.mean = norm[-1].get("mean", IMAGENET_MEAN) if norm else None
        self.std = norm[-1].get("std", IMAGENET_STD) if norm else None
        self.ignore_index = ignore_index

    def __call__(self, image, mask):
        for op in self.ops:
            image, mask = op(image, mask)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        if self.mean is not None:
            x = (x - torch.tensor(self.mean)[:, None, None]) / torch.tensor(self.std)[:, None, None]
        y = torch.from_numpy(np.asarray(mask, dtype=np.int64)) if mask is not None else None
        return x, y


def _resize(size):
    h, w = size

    def op(img, m):
        img = img.resize((w, h), Image.BILINEAR)
        return img, (m.resize((w, h), Image.NEAREST) if m is not None else None)
    return op


def _resize_short(short):
    def op(img, m):
        w, h = img.size
        s = short / min(w, h)
        size = (max(1, round(h * s)), max(1, round(w * s)))
        return _resize(size)(img, m)
    return op


def _random_scale(scale):
    lo, hi = scale

    def op(img, m):
        s = random.uniform(lo, hi)
        w, h = img.size
        return _resize((max(1, round(h * s)), max(1, round(w * s))))(img, m)
    return op


def _crop(size, ignore, center):
    ch, cw = size

    def op(img, m):
        w, h = img.size
        if w < cw or h < ch:  # pad: image with zeros, mask with ignore
            nw, nh = max(w, cw), max(h, ch)
            pad = Image.new("RGB", (nw, nh))
            pad.paste(img, (0, 0))
            img = pad
            if m is not None:
                mp = Image.new("L", (nw, nh), ignore)
                mp.paste(m, (0, 0))
                m = mp
            w, h = nw, nh
        x = (w - cw) // 2 if center else random.randint(0, w - cw)
        y = (h - ch) // 2 if center else random.randint(0, h - ch)
        box = (x, y, x + cw, y + ch)
        return img.crop(box), (m.crop(box) if m is not None else None)
    return op


def _hflip(p):
    def op(img, m):
        if random.random() < p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            m = m.transpose(Image.FLIP_LEFT_RIGHT) if m is not None else None
        return img, m
    return op


def _jitter(brightness=0.0, contrast=0.0, saturation=0.0):
    def op(img, m):
        for enh, amt in ((ImageEnhance.Brightness, brightness), (ImageEnhance.Contrast, contrast),
                         (ImageEnhance.Color, saturation)):
            if amt:
                img = enh(img).enhance(random.uniform(1 - amt, 1 + amt))
        return img, m
    return op


def _build(s, ignore):
    name = s.pop("name")
    if name == "resize":
        return _resize(s["size"])
    if name == "resize_short":
        return _resize_short(s["size"])
    if name == "random_scale":
        return _random_scale(s["scale"])
    if name in ("random_crop", "center_crop"):
        return _crop(s["size"], ignore, center=name == "center_crop")
    if name == "random_horizontal_flip":
        return _hflip(s.get("p", 0.5))
    if name == "color_jitter":
        return _jitter(**s)
    if name == "normalize":
        return lambda img, m: (img, m)  # applied after tensor conversion
    raise ValueError(f"Unknown transform {name}")
