"""V8 (LQA-LVCG): label-conditioned attention over the dynamics sequence."""

import pytest
import torch

from lvcg.quaternion.attention import MODES, LabelAttentionProbe
from lvcg.quaternion.mrq import build_probe


def _batch(n=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 640, generator=generator), torch.randn(n, 3, 1000, generator=generator)


# --- the attention ----------------------------------------------------------------


@pytest.mark.parametrize("mode", list(MODES))
def test_every_mode_runs_and_its_weights_are_a_distribution(mode):
    model = LabelAttentionProbe(mode=mode).eval()
    base, vcg = _batch()
    logits = model(base, vcg)
    assert logits.shape == (4, 5) and torch.isfinite(logits).all()
    weights, h, mask = model.attention(vcg)
    assert weights.shape[0] == 4 and weights.shape[-1] == h.shape[1]
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-5)
    assert (weights >= 0).all()


def test_label_mode_has_one_query_per_class_and_shared_mode_one_in_total():
    assert LabelAttentionProbe(mode="label").queries.shape == (5, 128)
    assert LabelAttentionProbe(mode="shared").queries.shape == (1, 128)
    assert not hasattr(LabelAttentionProbe(mode="mean"), "queries")


def test_unreliable_steps_receive_no_attention():
    """The reliability mask of V1 is carried into the attention, not only the pooling."""
    model = LabelAttentionProbe(mode="label").eval()
    vcg = torch.randn(2, 3, 1000)
    vcg[:, :, 400:700] *= 1e-9  # a stretch with no reliable direction
    weights, _, mask = model.attention(vcg)
    assert not mask.all(), "the quiet stretch must be masked"
    assert float(weights[:, :, ~mask[0]].abs().max()) == 0.0


def test_a_record_without_any_reliable_step_stays_finite():
    model = LabelAttentionProbe(mode="label").eval()
    base, vcg = _batch(n=2)
    logits = model(base, vcg * 1e-9)
    assert torch.isfinite(logits).all()


def test_labels_attend_differently_once_their_queries_differ():
    model = LabelAttentionProbe(mode="label").eval()
    with torch.no_grad():  # two deliberately opposite queries
        model.queries[0] = torch.randn(128)
        model.queries[1] = -model.queries[0]
    weights, _, _ = model.attention(_batch()[1])
    assert not torch.allclose(weights[:, 0], weights[:, 1], atol=1e-3)


def test_shared_mode_gives_every_label_the_same_weights():
    model = LabelAttentionProbe(mode="shared").eval()
    weights, _, _ = model.attention(_batch()[1])
    assert weights.shape[1] == 1, "one distribution, reused by every head"


# --- equivalences and structure ---------------------------------------------------


def test_mean_mode_reproduces_v3_to_float_rounding():
    """Uniform masked averaging is exactly what V1 and V3 do."""
    base, vcg = _batch()
    for variant in ("mo", "mq"):
        torch.manual_seed(0)
        ours = LabelAttentionProbe(variant=variant, mode="mean").eval()
        torch.manual_seed(0)
        v3 = build_probe(variant).eval()
        assert torch.allclose(ours(base, vcg), v3(base, vcg), atol=1e-6), variant


def test_scale_zero_reproduces_v0():
    model = LabelAttentionProbe(mode="label", scale=0.0).eval()
    base, vcg = _batch()
    assert torch.equal(model(base, vcg), model(base, torch.randn_like(vcg)))
    pooled = torch.zeros(4, 5, 128)
    joint = torch.cat((base.unsqueeze(1).expand(-1, 5, -1), pooled), dim=-1)
    assert torch.allclose(model(base, vcg), (joint * model.label_heads).sum(-1) + model.label_bias, atol=1e-6)


def test_each_label_head_reads_only_its_own_query():
    torch.manual_seed(0)
    model = LabelAttentionProbe(mode="label").eval()
    base, vcg = _batch()
    before = model(base, vcg)
    with torch.no_grad():
        # A random direction, not a constant: the scores are taken against a
        # zero-mean normalised feature map, so adding a constant to every element of a
        # query leaves them unchanged.
        model.queries[2] = torch.randn(128) * 3.0
    after = model(base, vcg)
    changed = (after - before).abs().max(dim=0).values
    assert changed[2] > 1e-4
    assert float(changed[[0, 1, 3, 4]].max()) < 1e-6


def test_the_label_heads_are_small_not_five_mlps():
    """Plan: one small head per label, not five large ones."""
    model = LabelAttentionProbe(mode="label")
    assert model.label_heads.shape == (5, 640 + 128)
    counts = model.parameter_counts()
    assert counts["queries"] == 5 * 128
    # The branch dominates; heads and queries together are a few percent of it.
    assert (counts["trainable_total"] - counts["branch"]) / counts["branch"] < 0.05


def test_attention_costs_little_over_the_mean_baseline():
    mean = LabelAttentionProbe(mode="mean").parameter_counts()["trainable_total"]
    label = LabelAttentionProbe(mode="label").parameter_counts()["trainable_total"]
    assert (label - mean) / mean < 0.01


def test_gradients_reach_the_queries_and_the_encoder():
    model = LabelAttentionProbe(mode="label")
    model(*_batch()).sum().backward()
    assert model.queries.grad.abs().max() > 0
    assert model.encoder.net[1].weight.grad.abs().max() > 0


def test_unknown_mode_or_variant_is_refused():
    with pytest.raises(ValueError):
        LabelAttentionProbe(mode="softmax")
    with pytest.raises(ValueError):
        LabelAttentionProbe(variant="curvature")
