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
    for name in ("smoke.yaml", "driving_dil.yaml", "driving_cd.yaml"):
        validate(load_config(ROOT / "configs" / name))


def test_cd_protocol_derives_external():
    st = build_stages(load_config(ROOT / "configs" / "driving_cd.yaml"))
    assert st[0]["external"] == ["idd"] and st[1]["external"] == ["bdd100k"]


@pytest.mark.parametrize("sets", [
    ["components.exemplar_buffer=false"],                                   # echo needs buffer
    ["components.teacher_kd=false"],                                        # no current signal
    ["components.bn_domain_prototypes=false", "components.bn_layer_distill=false"],  # stats-aware unused
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
