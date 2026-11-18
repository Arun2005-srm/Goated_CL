"""YAML configuration: inheritance (`extends`), file includes, CLI overrides, validation."""
from copy import deepcopy
from pathlib import Path
import os
import yaml

COMPONENTS = (
    "teacher_kd", "old_student_kd", "exemplar_buffer", "teacher_echo",
    "bn_layer_distill", "bn_domain_prototypes", "stats_aware_prototypes",
    "auto_domain_routing", "entropy_gated_fusion", "class_prototype_replay",
    "stochastic_restore", "aug_pseudo_label", "supervised_ce",
)


def merge(base, override):
    out = deepcopy(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else deepcopy(v)
    return out


def _read(path):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(os.path.expandvars(f.read())) or {}


def load_config(path, overrides=(), _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    seen.add(path)
    raw = _read(path)
    parent = raw.pop("extends", None)
    cfg = merge(load_config(path.parent / parent, _seen=seen), raw) if parent else raw
    # Domain specs may be given as `{include: datasets/x.yaml, ...overrides}`.
    for name, spec in list((cfg.get("domains") or {}).items()):
        if isinstance(spec, str):
            spec = {"include": spec}
        if isinstance(spec, dict) and "include" in spec:
            inc = Path(spec["include"])
            inc = inc if inc.is_absolute() else path.parent / inc
            local = {k: v for k, v in spec.items() if k != "include"}
            cfg["domains"][name] = merge(_read(inc), local)
    for item in overrides:
        key, value = item.split("=", 1)
        cur = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(value)
    return cfg


def build_stages(cfg):
    """Explicit `protocol.stages`, or one stage per domain in `domain_order`."""
    stages = cfg["protocol"].get("stages") or [
        {"name": d, "teacher": d, "distill_on": [d]} for d in cfg["domain_order"]
    ]
    out = []
    for s in stages:
        s = dict(s)
        s.setdefault("name", s["teacher"])
        s.setdefault("distill_on", [s["teacher"]])
        s.setdefault("external", [d for d in s["distill_on"] if d not in teacher_domains(cfg, s["teacher"])])
        s.setdefault("epochs", cfg["distill"]["epochs"])
        out.append(s)
    return out


def teacher_domains(cfg, teacher):
    spec = (cfg["teachers"].get("per_teacher") or {}).get(teacher, {}) or {}
    shared = list(cfg["teachers"].get("shared_domains") or [])
    return list(dict.fromkeys(spec.get("train_on", shared + [teacher])))


def validate(cfg):
    c = len(cfg["classes"])
    if c < 2:
        raise ValueError("At least two classes are required")
    if cfg.get("colors") and len(cfg["colors"]) != c:
        raise ValueError("colors must have one RGB triple per class")
    if cfg["ignore_index"] in range(c):
        raise ValueError("ignore_index must be outside class indices")
    comps = cfg["components"]
    if set(comps) != set(COMPONENTS) or any(type(v) is not bool for v in comps.values()):
        raise ValueError(f"components must contain exactly these booleans: {list(COMPONENTS)}")
    if not (comps["teacher_kd"] or comps["supervised_ce"] or comps["aug_pseudo_label"]):
        raise ValueError("Enable at least one current-stage learning signal: teacher_kd, aug_pseudo_label or supervised_ce")
    if comps["teacher_echo"] and not comps["exemplar_buffer"]:
        raise ValueError("teacher_echo stores soft labels on buffer exemplars: enable exemplar_buffer")
    if comps["stats_aware_prototypes"] and not (comps["bn_domain_prototypes"] or comps["bn_layer_distill"]):
        raise ValueError("stats_aware_prototypes changes the BN prototype losses: enable bn_domain_prototypes or bn_layer_distill")
    if comps["exemplar_buffer"] and not (comps["teacher_echo"] or comps["old_student_kd"] or comps["supervised_ce"]
                                         or comps["bn_domain_prototypes"]):
        raise ValueError("exemplar_buffer is unused: replayed images need echo, old-student KD, CE or BN prototypes")
    if cfg["teachers"].get("loss", "ce") not in ("ce", "ohem"):
        raise ValueError("teachers.loss must be ce or ohem")
    if cfg["protocol"].get("strict_teacher_access") and comps["teacher_echo"]:
        raise ValueError("protocol.strict_teacher_access forbids teacher_echo (stored outputs of past teachers)")
    if cfg["hparams"]["old_student_on"] not in ("external", "all"):
        raise ValueError("hparams.old_student_on must be 'external' or 'all'")
    if cfg["hparams"]["buffer"]["priority"] not in ("vog", "entropy", "random"):
        raise ValueError("buffer.priority must be vog, entropy or random")
    if not cfg["domain_order"] and not cfg["protocol"].get("stages"):
        raise ValueError("Set domain_order or protocol.stages")
    for split in ("eval",):
        for t in cfg["transforms"][split]:
            if t["name"].startswith("random") or t["name"] == "color_jitter":
                raise ValueError(f"Stochastic {split} transform is forbidden: {t['name']}")
    stages = build_stages(cfg)
    names = [s["name"] for s in stages]
    if len(set(names)) != len(names):
        raise ValueError("Stage names must be unique")
    for s in stages:
        for d in [*s["distill_on"], *s["external"], *teacher_domains(cfg, s["teacher"])]:
            if d not in cfg["domains"]:
                raise ValueError(f"Stage {s['name']} references unknown domain '{d}'")
    for name, d in cfg["domains"].items():
        if "splits" not in d:
            raise ValueError(f"Domain {name} has no splits")
    if cfg["distill"]["temperature"] <= 0:
        raise ValueError("distill.temperature must be positive")
    return cfg


def dump(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
