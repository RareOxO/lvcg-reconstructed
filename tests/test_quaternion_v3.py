"""V3 (MRQ-LVCG): the magnitude/rotation split, the component switches and the direction
diagnostic."""

import importlib.util
import os

import pytest
import torch

from lvcg.quaternion.features import QuaternionDynamicFeatures
from lvcg.quaternion.mrq import COMPONENTS, MRQProbe

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _batch(n=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 640, generator=generator), torch.randn(n, 3, 1000, generator=generator)


def test_direction_is_the_unit_cardiac_vector_and_ignores_magnitude():
    """u_t = P_t / ||P_t||: what the shortest-arc quaternion discards."""
    features = QuaternionDynamicFeatures(("direction",), dt=0.01, with_mask=False)
    vcg = torch.randn(2, 3, 200)
    unit, _ = features(vcg)
    assert torch.allclose(unit.pow(2).sum(1).sqrt(), torch.ones(2, 199), atol=1e-5)
    scaled, _ = features(vcg * 7.5)  # a different amplitude, the same orientation
    assert torch.allclose(unit, scaled, atol=1e-5)


def test_the_factorisation_reconstructs_the_cardiac_vector():
    """P_t = r_t u_t, so the two branches together carry the whole vector."""
    features = QuaternionDynamicFeatures(("magnitude", "direction"), dt=0.01, with_mask=False)
    vcg = torch.randn(2, 3, 200)
    x, _ = features(vcg)
    magnitude, direction = x[:, :1], x[:, 1:]
    assert torch.allclose(magnitude * direction, vcg[..., :-1], atol=1e-4)


@pytest.mark.parametrize("components", [
    ("vcg",), ("vcg", "magnitude", "rotation"), ("magnitude", "rotation"),
    ("vcg", "rotation"), ("vcg", "magnitude"), ("rotation",), ("magnitude",),
    ("vcg", "control"), ("vcg", "rotation", "direction"),
])
def test_every_ablation_of_the_plan_runs_and_stays_finite(components):
    model = MRQProbe(components=components).eval()
    logits = model(*_batch())
    assert logits.shape == (6, 5) and torch.isfinite(logits).all()


def test_vcg_only_is_the_v0_linear_probe():
    """Plan 7.1 Level 0: with every quaternion branch off, the model is V0."""
    model = MRQProbe(components=("vcg",)).eval()
    base, vcg = _batch()
    assert torch.equal(model(base, vcg), model.head(base))
    assert len(model.encoders) == 0


def test_dropping_the_pretrained_embedding_drops_it_from_the_head():
    with_base = MRQProbe(components=("vcg", "rotation"))
    without = MRQProbe(components=("rotation",)).eval()  # eval: no dropout between the two calls
    assert with_base.head.in_features - without.head.in_features == 640
    base, vcg = _batch()
    # The branch output is what the two share; only the head input differs.
    assert torch.equal(without(torch.randn_like(base), vcg), without(torch.zeros_like(base), vcg))


def test_each_branch_sees_only_its_own_factor():
    """Magnitude must not react to a pure rotation, rotation must not react to a rescaling."""
    torch.manual_seed(0)
    model = MRQProbe(components=("magnitude", "rotation")).eval()
    base, vcg = _batch(n=2)

    rescaled = vcg * 3.0  # same directions, different amplitude
    magnitude, rotation = model.embeddings(base, vcg)
    magnitude_rescaled, rotation_rescaled = model.embeddings(base, rescaled)
    assert torch.allclose(rotation, rotation_rescaled, atol=1e-4), "rotation ignores amplitude"
    assert not torch.allclose(magnitude, magnitude_rescaled, atol=1e-3), "magnitude follows it"


def test_component_order_is_the_order_the_head_reads():
    model = MRQProbe(components=("vcg", "rotation", "magnitude"))
    parts = model.embeddings(*_batch())
    assert [p.shape[-1] for p in parts] == [640, 128, 128]
    assert model.head.in_features == 640 + 128 + 128


def test_unknown_or_empty_components_are_refused():
    with pytest.raises(ValueError):
        MRQProbe(components=("vcg", "curvature"))
    with pytest.raises(ValueError):
        MRQProbe(components=())


def test_every_enabled_branch_receives_gradient():
    model = MRQProbe(components=("vcg", "magnitude", "rotation", "direction"))
    model(*_batch()).sum().backward()
    for name, encoder in model.encoders.items():
        assert encoder.net[1].weight.grad.abs().max() > 0, name


def test_the_v1_training_loop_drives_this_model_unchanged():
    """V3 reuses V1's loop and cache, so the two stages are scored identically."""
    spec = importlib.util.spec_from_file_location(
        "train_qdf", os.path.join(REPO_ROOT, "scripts", "train_qdf.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = torch.Generator().manual_seed(0)
    def make(n):
        labels = (torch.rand(n, 5, generator=generator) > 0.5).float()
        vcg = torch.randn(n, 3, 200, generator=generator) * 0.2
        vcg *= 1.0 + 4.0 * labels[:, :1].unsqueeze(-1)  # label 0 lives in the magnitude
        return torch.zeros(n, 8), vcg, labels

    data = {"train": make(256), "val": make(64), "test": make(64)}
    model = MRQProbe(components=("magnitude",), base_dim=8, embedding_dim=16, hidden=8)
    module.train(model, data, torch.device("cpu"), {"batch_size": 32, "max_epochs": 8, "patience": 8}, seed=0)
    scores = module.evaluate(model, data["val"], torch.device("cpu"), 64)
    assert scores["per_label_auroc"][0] > 0.85, scores["per_label_auroc"]


def test_components_survive_a_shell_that_does_not_split_words(tmp_path, monkeypatch):
    """zsh passes "vcg rotation" as a single argument; the script must still split it."""
    spec = importlib.util.spec_from_file_location(
        "train_mrq", os.path.join(REPO_ROOT, "scripts", "train_mrq.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser_components = lambda argv: tuple(  # noqa: E731
        part for argument in argv for part in __import__("re").split(r"[\s,+]+", argument.strip()) if part
    )
    assert parser_components(["vcg rotation direction"]) == ("vcg", "rotation", "direction")
    assert parser_components(["vcg", "control"]) == ("vcg", "control")
    assert parser_components(["vcg+magnitude,rotation"]) == ("vcg", "magnitude", "rotation")
