"""Synthetic 3-domain 'driving-like' dataset for offline smoke tests. Same 4 classes in every
domain (road, building, sky, car); domains differ in colour palette, noise and brightness.
Validates execution only; numbers from it say nothing about the method."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

DOMAINS = {  # class colours per domain + global brightness + noise
    "sunny": dict(cols=[(110, 110, 110), (180, 120, 90), (120, 170, 235), (200, 30, 30)], bright=1.0, noise=8),
    "foggy": dict(cols=[(150, 150, 155), (170, 160, 150), (200, 200, 205), (170, 90, 90)], bright=1.05, noise=4),
    "night": dict(cols=[(40, 40, 50), (70, 50, 40), (15, 15, 35), (120, 20, 20)], bright=0.7, noise=14),
}


def scene(rng, h, w, d):
    m = np.full((h, w), 2, np.uint8)                       # sky
    horizon = rng.integers(h // 3, h // 2)
    m[horizon:] = 0                                         # road
    for _ in range(rng.integers(1, 4)):                     # buildings
        x0, bw = rng.integers(0, w - 8), rng.integers(8, w // 3)
        top = rng.integers(h // 8, horizon)
        m[top:horizon, x0:x0 + bw] = 1
    for _ in range(rng.integers(1, 3)):                     # cars on road
        cy, cx = rng.integers(horizon + 3, h - 6), rng.integers(4, w - 16)
        m[cy:cy + rng.integers(4, 8), cx:cx + rng.integers(8, 16)] = 3
    img = np.array(d["cols"], np.float32)[m]
    img = img * d["bright"] + rng.normal(0, d["noise"], img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/toy_data")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--size", type=int, nargs=2, default=[64, 128])
    a = p.parse_args()
    rng = np.random.default_rng(0)
    for name, d in DOMAINS.items():
        for split, n in (("train", a.n), ("test", a.n // 3)):
            for i in range(n):
                img, m = scene(rng, *a.size, d)
                seq = f"seq{i % 4}"
                ip = Path(a.out) / name / "images" / split / seq / f"{i:04d}.png"
                mp = Path(a.out) / name / "masks" / split / seq / f"{i:04d}.png"
                ip.parent.mkdir(parents=True, exist_ok=True)
                mp.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(ip)
                Image.fromarray(m).save(mp)
    print(f"toy dataset written to {a.out}")


if __name__ == "__main__":
    main()
