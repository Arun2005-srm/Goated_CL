from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from goated.config import COMPONENTS, load_config, validate, build_stages  # noqa: E402
from goated.components.losses import fuse, kd_loss, soften  # noqa: E402
from goated.components.memory import ExemplarBuffer  # noqa: E402
from goated.components.prototypes import DomainPrototypeBank, layer_distance  # noqa: E402
from goated.engine.common import confusion, metrics_from_confusion  # noqa: E402
from goated.engine.distill import GoatedDistiller  # noqa: E402

SMOKE = ROOT / "configs" / "smoke.yaml"


def cfg(*sets):
    return load_config(SMOKE, list(sets))


def test_all_configs_validate():
    for name in ("smoke.yaml", "driving_dil.yaml", "driving_cd.yaml", "driving_dil_strong.yaml"):
        validate(load_config(ROOT / "configs" / name))


def test_cd_protocol_derives_external():
    st = build_stages(load_config(ROOT / "configs" / "driving_cd.yaml"))
    assert st[0]["external"] == ["idd"] and st[1]["external"] == ["bdd100k"]


@pytest.mark.parametrize("sets", [
    ["components.exemplar_buffer=false"],                                   # echo needs buffer
    ["components.teacher_kd=false"],                                        # no current signal
    ["components.bn_domain_prototypes=false", "components.bn_layer_distill=false"],  # stats-aware unused
    ["protocol.strict_teacher_access=true"],                                # strict forbids echoes
    ["teachers.loss=focal"],
])
def test_invalid_combinations_rejected(sets):
    with pytest.raises(ValueError):
        validate(cfg(*sets))


def test_disabled_components_build_nothing(tmp_path):
    off = [f"components.{k}=false" for k in COMPONENTS if k != "teacher_kd"]
    c = validate(cfg(*off, f"output_dir={tmp_path.as_posix()}"))
    d = GoatedDistiller(c, {}, torch.device("cpu"), {})
    assert d.bank is None and d.buffer is None and d.cbank is None
    full = GoatedDistiller(validate(cfg(f"output_dir={tmp_path.as_posix()}")), {}, torch.device("cpu"), {})
    assert full.bank is not None and full.buffer is not None and full.cbank is not None


def test_stats_aware_gain_is_lower_for_noisy_channels():
    torch.manual_seed(0)
    bank = DomainPrototypeBank("stats_aware", warmup=8)
    noise = torch.tensor([0.01, 1.0])  # mu-half: [stable, noisy]; lv-half constant
    for _ in range(200):
        mu = (torch.tensor([1.0, 1.0]) + noise * torch.randn(2))[None]
        bank.update("d", [(mu, torch.zeros(1, 2))])
    s = bank.state["d"][0]
    R = bank._R(s)
    K = (s["P"] + bank.q * R) / (s["P"] + bank.q * R + R)
    assert abs(float(s["m"][0]) - 1.0) < 0.01
    assert torch.allclose(K[:2], K[:2].mean(), rtol=0.5)  # gain relative to own noise, both converge
    # outlier gating: a 100-sigma observation barely moves the stable channel
    before = s["m"][0].clone()
    bank.update("d", [(torch.tensor([[100.0, 1.0]]), torch.zeros(1, 2))])
    assert abs(float(s["m"][0] - before)) < 1.0 and s["rejected"] >= 1


def test_ema_mode_matches_dose_rule():
    bank = DomainPrototypeBank("ema", ema_momentum=0.9)
    bank.update("d", [(torch.tensor([[0.0]]), torch.tensor([[0.0]]))])
    bank.update("d", [(torch.tensor([[1.0]]), torch.tensor([[0.0]]))])
    assert torch.isclose(bank.state["d"][0]["m"][0], torch.tensor(0.1))


def test_stats_aware_distance_ignores_natural_spread():
    bank = DomainPrototypeBank("stats_aware", warmup=2, margin=2.0)
    for v in (0.0, 1.0, -1.0, 0.5, -0.5):
        bank.update("d", [(torch.tensor([[v]]), torch.tensor([[v]]))])
    near = bank.distance("d", [(torch.tensor([[0.3]]), torch.tensor([[0.3]]))])
    far = bank.distance("d", [(torch.tensor([[20.0]]), torch.tensor([[20.0]]))])
    assert float(near) == 0.0 and float(far) > 0


def test_layer_distance_zero_for_identical_stats():
    s = [(torch.randn(3, 4), torch.randn(3, 4))]
    assert torch.allclose(layer_distance(s, s), torch.zeros(3))


def test_echo_roundtrip_exact_when_topk_covers_classes():
    buf = ExemplarBuffer(per_stage=4, topk=5, stride=1, store_labels=False)
    p = torch.softmax(torch.randn(2, 5, 4, 4), 1)
    buf.offer("s", torch.randn(2, 3, 4, 4), [0, 0], [1.0, 2.0], teacher_p1=p)
    out = buf.sample(2, (4, 4), "cpu")
    assert out["echo"].shape == (2, 5, 4, 4)
    assert torch.allclose(out["echo"].sum(1), torch.ones(2, 4, 4), atol=1e-3)


def test_buffer_keeps_highest_priority():
    buf = ExemplarBuffer(per_stage=2, topk=3, stride=1, store_labels=False)
    buf.offer("s", torch.randn(4, 3, 4, 4), [0] * 4, [0.1, 0.9, 0.5, 0.2])
    assert sorted(e["priority"] for e in buf.buckets["s"]) == [0.5, 0.9]


def test_fusion_prefers_confident_source():
    C = 4
    confident = torch.zeros(1, C, 2, 2); confident[:, 1] = 1.0
    confident = confident * 0.97 + 0.01
    uncertain = torch.full((1, C, 2, 2), 1.0 / C)
    target, w = fuse([(confident, 1.0), (uncertain, 1.0)], T=1.0, tau=0.5)
    assert (target.argmax(1) == 1).all() and target[:, 1].min() > 0.8


def test_kd_zero_when_student_matches():
    logits = torch.randn(2, 3, 4, 4)
    assert float(kd_loss(logits, soften(torch.softmax(logits, 1), 2.0), 2.0)) < 1e-5


def test_miou_known_values():
    pred = torch.tensor([0, 0, 1, 1])
    gt = torch.tensor([0, 1, 1, 255])
    m = metrics_from_confusion(confusion(pred, gt, 2, 255), ["a", "b"])
    assert abs(m["miou"] - 50.0) < 1e-6  # IoU(a)=1/2, IoU(b)=1/2


# ---------------------------------------------------------------------- teacher access guard
def _guard_cfg(tmp_path, *sets):
    from goated.models.segmenter import Segmenter
    c = validate(cfg(f"output_dir={tmp_path.as_posix()}", f"teachers.dir={(tmp_path / 't').as_posix()}", *sets))
    (tmp_path / "t").mkdir()
    for d in c["domain_order"]:
        m = Segmenter(c["models"]["teacher"], len(c["classes"]))
        torch.save({"model": m.state_dict(), "model_spec": c["models"]["teacher"]}, tmp_path / "t" / f"{d}.pt")
    return c


def test_guard_refuses_other_stage_teacher(tmp_path):
    from goated.engine.access import TeacherAccessGuard, TeacherAccessError
    c = _guard_cfg(tmp_path)
    st = build_stages(c)
    g = TeacherAccessGuard(c, st, tmp_path)
    with pytest.raises(TeacherAccessError):
        g.load(1, st[0], torch.device("cpu"))          # stage 2 asking for stage-1 teacher


def test_guard_detects_teacher_kept_alive(tmp_path):
    from goated.engine.access import TeacherAccessGuard, TeacherAccessError
    c = _guard_cfg(tmp_path)
    st = build_stages(c)
    g = TeacherAccessGuard(c, st, tmp_path)
    t = g.load(0, st[0], torch.device("cpu"))
    assert all(not p.requires_grad for p in t.parameters())
    with pytest.raises(TeacherAccessError):
        g.release()                                     # `t` still referenced here
    del t
    g2 = TeacherAccessGuard(c, st, tmp_path)
    t = g2.load(0, st[0], torch.device("cpu"))
    del t
    g2.release()
    t = g2.load(1, st[1], torch.device("cpu"))         # next stage loads fine after release
    del t
    g2.release()
    summ = g2.write()
    assert summ["only_current_teacher_each_stage"] and summ["all_teachers_released"]
    assert not summ["gt_labels_used_in_any_loss"]


def test_guard_strict_rejects_echo(tmp_path):
    from goated.engine.access import TeacherAccessGuard, TeacherAccessError
    c = _guard_cfg(tmp_path, "components.teacher_echo=false", "protocol.strict_teacher_access=true")
    g = TeacherAccessGuard(c, build_stages(c), tmp_path)
    g.begin_without_teacher(1, build_stages(c)[1])
    with pytest.raises(TeacherAccessError):
        g.note_echo(["sunny"])


def test_ohem_focuses_on_hard_pixels():
    from goated.engine.teachers import seg_loss
    y = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 0] = 10.0                                 # all easy ...
    logits[0, 0, 0, 0], logits[0, 1, 0, 0] = 0.0, 10.0  # ... except one hard pixel
    ce = seg_loss(logits, y, 255, kind="ce")
    oh = seg_loss(logits, y, 255, kind="ohem", ohem_min_kept=0.0)
    assert oh > 5 * ce


# ---------------------------------------------------------------------- one GPU per run
SMI = [  # the shared 8-GPU server: GPUs 2 and 7 idle
    {"index": 0, "used": 32488, "total": 40960, "util": 100}, {"index": 1, "used": 25653, "total": 40960, "util": 100},
    {"index": 2, "used": 721, "total": 40960, "util": 0}, {"index": 3, "used": 36799, "total": 40960, "util": 100},
    {"index": 7, "used": 4, "total": 40960, "util": 0}]


@pytest.fixture
def gpu_env(monkeypatch):
    import goated.gpu as g
    monkeypatch.setattr(g, "_DONE", {})
    monkeypatch.setattr(g, "_query", lambda: SMI)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return g


def test_gpu_auto_picks_single_least_used(gpu_env):
    import os
    gpu_env.pin_single_gpu({"gpu": {"id": "auto", "min_free_gb": 16}})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_gpu_fixed_id_and_env_limited_to_one(gpu_env, monkeypatch):
    import os
    gpu_env.pin_single_gpu({"gpu": {"id": 2}})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    monkeypatch.setattr(gpu_env, "_DONE", {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4,5")
    gpu_env.pin_single_gpu({"gpu": {"id": "auto"}})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_gpu_refuses_when_all_busy(gpu_env, monkeypatch):
    monkeypatch.setattr(gpu_env, "_query", lambda: [r for r in SMI if r["util"] == 100])
    with pytest.raises(RuntimeError):
        gpu_env.pin_single_gpu({"gpu": {"id": "auto", "min_free_gb": 16}})


def test_gpu_id_validated():
    with pytest.raises(ValueError):
        validate(cfg("gpu.id=-1"))
