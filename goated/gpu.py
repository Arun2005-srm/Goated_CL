"""Pin the process to exactly ONE GPU (shared university server: one GPU per user).

Must run before CUDA is initialised, so this module does not import torch. It sets
CUDA_VISIBLE_DEVICES to a single index; inside the process that GPU is then `cuda:0`.

Config (`gpu:` in base.yaml):
  id: auto        auto = least-used GPU from nvidia-smi | an integer index, e.g. 2
  min_free_gb: 16 auto only: refuse to start if even the freest GPU has less free memory

If CUDA_VISIBLE_DEVICES is already set (by you or a job scheduler), it wins, but only its
first entry is kept, so the run can never use more than one GPU."""
import os
import subprocess

_DONE = {}


def _query():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        i, used, total, util = [v.strip() for v in line.split(",")]
        rows.append({"index": int(i), "used": float(used), "total": float(total), "util": float(util)})
    return rows


def pin_single_gpu(cfg):
    """Restrict this process to one GPU. Returns a description; idempotent."""
    if "msg" in _DONE:
        return _DONE["msg"]
    g = cfg.get("gpu") or {}
    if cfg.get("device") == "cpu":
        msg = "cpu (no GPU used)"
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        first = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = first
        msg = f"GPU {first} (from CUDA_VISIBLE_DEVICES; limited to one GPU)"
    elif g.get("id", "auto") != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(g["id"]))
        msg = f"GPU {int(g['id'])} (gpu.id in config)"
    else:
        try:
            rows = _query()
        except (OSError, subprocess.CalledProcessError):
            rows = []
        if not rows:
            msg = "no nvidia-smi found; CUDA left as is"
        else:
            best = min(rows, key=lambda r: (r["used"], r["util"]))
            free_gb = (best["total"] - best["used"]) / 1024
            need = float(g.get("min_free_gb", 0) or 0)
            if free_gb < need:
                table = ", ".join(f"{r['index']}:{(r['total'] - r['used']) / 1024:.1f}GB free/{r['util']:.0f}%"
                                  for r in rows)
                raise RuntimeError(f"No GPU has {need} GB free (freest is GPU {best['index']} with "
                                   f"{free_gb:.1f} GB). GPUs: {table}. Wait, or set gpu.id / lower gpu.min_free_gb.")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best["index"])
            msg = f"GPU {best['index']} (auto: least used, {free_gb:.1f} GB free, {best['util']:.0f}% util)"
    _DONE["msg"] = msg
    print(f"[gpu] using {msg}")
    return msg
