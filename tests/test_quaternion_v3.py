"""V3 (MRQ-LVCG, revised): the magnitude / orientation / rotation decomposition."""

import importlib.util
import os

import pytest
import torch

from lvcg.quaternion.features import QuaternionDynamicFeatures
from lvcg.quaternion.mrq import LEGACY_NAMES, VARIANTS, build_probe
from lvcg.quaternion.qdf import QDFProbe

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _batch(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 640, generator=generator), torch.randn(n, 3, 1000, generator=generator)


# --- the decomposition ------------------------------------------------------------


def test_orientation_is_the_unit_vector_in_the_fixed_frame():
    """u_t = P_t / ||P_t||: the absolute direction q_t discards, with no per-record rotation."""
    features = QuaternionDynamicFeatures(("direction",), dt=0.01, with_mask=False)
    vcg = torch.randn(2, 3, 200)
    unit, _ = features(vcg)
    assert torch.allclose(unit.pow(2).sum(1).sqrt(), torch.ones(2, 199), atol=1e-5)
    assert torch.allclose(unit, features(vcg * 7.5)[0], atol=1e-5), "orientation ignores amplitude"
    # A rotation of the record changes it: the frame is fixed, not normalised away.
    angle = torch.tensor(0.7)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                             [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    assert not torch.allclose(unit, features(rotation @ vcg)[0], atol=1e-3)


def test_magnitude_and_orientation_reconstruct_the_cardiac_vector():
    features = QuaternionDynamicFeatures(("magnitude", "direction"), dt=0.01, with_mask=False)
    vcg = torch.randn(2, 3, 200)
    x, _ = features(vcg)
    assert torch.allclose(x[:, :1] * x[:, 1:], vcg[..., :-1], atol=1e-4)


def test_rotation_ignores_amplitude_and_magnitude_ignores_direction():
    rotation = QuaternionDynamicFeatures(VARIANTS["q"], dt=0.01, with_mask=False)
    magnitude = QuaternionDynamicFeatures(VARIANTS["m"], dt=0.01, with_mask=False)
    vcg = torch.randn(2, 3, 200)
    assert torch.allclose(rotation(vcg)[0], rotation(vcg * 4.0)[0], atol=1e-4)
    angle = torch.tensor(0.7)
    spin = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0],
                         [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(magnitude(vcg)[0], magnitude(spin @ vcg)[0], atol=1e-4)


# --- the variants -----------------------------------------------------------------


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_variant_runs_and_stays_finite(variant):
    logits = build_probe(variant).eval()(*_batch())
    assert logits.shape == (6, 5) and torch.isfinite(logits).all()


def test_mq_and_cdf_are_v1_channel_for_channel():
    """Plan 12: V1's locked results must carry over, so these two sets cannot drift."""
    spec = importlib.util.spec_from_file_location(
        "train_qdf", os.path.join(REPO_ROOT, "scripts", "train_qdf.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert VARIANTS["mq"] == module.FEATURE_SETS["qdf"]
    assert VARIANTS["cdf"] == module.FEATURE_SETS["control"] == module.FEATURE_SETS["cdf"]

    base, vcg = _batch()
    torch.manual_seed(0)
    ours = build_probe("mq").eval()
    torch.manual_seed(0)
    v1 = QDFProbe(features=module.FEATURE_SETS["qdf"]).eval()
    assert torch.equal(ours(base, vcg), v1(base, vcg))


def test_legacy_names_still_resolve():
    assert LEGACY_NAMES == {"qdf": "mq", "control": "cdf"}
    assert build_probe("qdf").dynamics.features == VARIANTS["mq"]
    assert build_probe("control").dynamics.features == VARIANTS["cdf"]


def test_v0_reproduces_the_frozen_probe_bitwise():
    """Plan 7: scale = 0 must equal V0's logits exactly."""
    torch.manual_seed(0)
    model = build_probe("v0").eval()
    base, vcg = _batch()
    assert torch.equal(model(base, vcg), model.head(torch.cat((base, torch.zeros(6, 128)), dim=-1)))
    model(base, vcg).sum().backward()
    assert model.encoder.net[1].weight.grad.abs().max() == 0


def test_scale_one_gives_the_branch_gradient():
    for variant in ("o", "oq", "moq"):
        model = build_probe(variant)
        model(*_batch()).sum().backward()
        assert model.encoder.net[1].weight.grad.abs().max() > 0, variant


def test_parameter_counts_stay_within_a_few_percent():
    """Capacity must not be what distinguishes the variants (plan 6, 7)."""
    counts = {name: build_probe(name).parameter_counts()["trainable_total"] for name in VARIANTS}
    smallest, largest = min(counts.values()), max(counts.values())
    assert (largest - smallest) / smallest < 0.03, counts


def test_unknown_variant_is_refused():
    with pytest.raises(ValueError):
        build_probe("curvature")


def test_near_zero_stretches_stay_finite_in_every_variant():
    vcg = torch.randn(2, 3, 300)
    vcg[:, :, 100:150] *= 1e-9
    for variant in VARIANTS:
        logits = build_probe(variant).eval()(torch.randn(2, 640), vcg)
        assert torch.isfinite(logits).all(), variant


def test_the_v1_loop_trains_an_orientation_only_signal():
    spec = importlib.util.spec_from_file_location(
        "train_qdf", os.path.join(REPO_ROOT, "scripts", "train_qdf.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = torch.Generator().manual_seed(0)
    def make(n):
        labels = (torch.rand(n, 5, generator=generator) > 0.5).float()
        vcg = torch.randn(n, 3, 200, generator=generator) * 0.05
        # Label 0 tilts the mean cardiac vector: absolute orientation, constant magnitude.
        vcg[:, 0] += 1.0 - labels[:, 0:1]
        vcg[:, 1] += labels[:, 0:1]
        return torch.zeros(n, 8), vcg, labels

    data = {"train": make(512), "val": make(128), "test": make(128)}
    model = build_probe("o", base_dim=8, embedding_dim=16, hidden=8)
    module.train(model, data, torch.device("cpu"), {"batch_size": 32, "max_epochs": 8, "patience": 8}, seed=0)
    scores = module.evaluate(model, data["val"], torch.device("cpu"), 64)
    assert scores["per_label_auroc"][0] > 0.85, scores["per_label_auroc"]
