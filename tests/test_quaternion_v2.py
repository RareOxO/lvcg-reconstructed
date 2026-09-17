"""V2 (QDT-LVCG): beat quaternion tokens, the pretrained-preserving fusion, and the
structural limit that only beat token 1 reaches the output."""

import torch

from lvcg.models.lvcg import StateGRU
from lvcg.quaternion.features import BeatQuaternionFeatures
from lvcg.quaternion.qdt import QDTProbe, TokenFusion, rollout_hidden

TOKEN_DIM, BEATS, PATCH = 256, 6, 128


def _probe(**kwargs):
    torch.manual_seed(0)
    generator = StateGRU(state_dim=TOKEN_DIM, hidden_dim=TOKEN_DIM, num_layers=2, dropout=0.0)
    return QDTProbe(
        state_generator=generator,
        norm_struct=torch.nn.LayerNorm(TOKEN_DIM),
        norm_dynamic=torch.nn.LayerNorm(TOKEN_DIM),
        **kwargs,
    ).eval()


def _batch(batch=4, beats=BEATS, seed=1):
    generator = torch.Generator().manual_seed(seed)
    patches = torch.randn(batch, beats, 3, PATCH, generator=generator)
    rr = torch.full((batch, beats), 85.0)
    mask = torch.ones(batch, beats)
    reference = patches.norm(dim=2).reshape(batch, -1).quantile(0.99, dim=-1)
    tokens = torch.randn(batch, beats, TOKEN_DIM, generator=generator)
    steps = torch.full((batch,), beats - 1, dtype=torch.long)
    rhythm = torch.randn(batch, 128, generator=generator)
    return patches, rr, mask, reference, tokens, steps, rhythm


def test_beat_features_have_one_transition_per_patch_step_and_mask_padding():
    features = BeatQuaternionFeatures(("q", "theta", "omega", "magnitude"), fs=100)
    patches, rr, mask, reference, *_ = _batch()
    mask[:, -2:] = 0  # two padding beats
    x, valid = features(patches, rr, mask, reference)
    assert x.shape == (4, BEATS, features.channels, PATCH - 1)
    assert torch.isfinite(x).all()
    assert not valid[:, -2:].any(), "padding beats must be masked everywhere"


def test_both_fusions_start_as_the_identity_on_the_pretrained_token():
    z = torch.randn(3, BEATS, TOKEN_DIM)
    q = torch.randn(3, BEATS, 128)
    for mode in ("gated", "concat"):
        fusion = TokenFusion(TOKEN_DIM, 128, mode)
        assert torch.allclose(fusion(z, q), z, atol=1e-6), mode


def test_the_zero_block_learns_first_and_the_branch_only_after_it():
    """Zero initialisation means the branch is gradient-free on step 0, the fusion is not."""
    for mode in ("gated", "concat"):
        fusion = TokenFusion(TOKEN_DIM, 128, mode)
        z, q = torch.randn(3, BEATS, TOKEN_DIM), torch.randn(3, BEATS, 128, requires_grad=True)
        fusion(z, q).sum().backward()
        assert q.grad.abs().max() == 0, f"{mode}: nothing reaches the branch yet"
        block = fusion.value.weight if mode == "gated" else fusion.projection.weight
        assert block.grad.abs().max() > 0, f"{mode}: the fusion itself must learn"

        with torch.no_grad():  # one update moves the block off zero
            block -= 0.1 * block.grad
        q.grad = None
        fusion(z, q).sum().backward()
        assert q.grad.abs().max() > 0, f"{mode}: the branch learns once the block is non-zero"


def test_near_zero_initialisation_lets_gradient_reach_the_branch_immediately():
    for mode in ("gated", "concat"):
        fusion = TokenFusion(TOKEN_DIM, 128, mode, init_std=0.01)
        z, q = torch.randn(3, BEATS, TOKEN_DIM), torch.randn(3, BEATS, 128, requires_grad=True)
        out = fusion(z, q)
        drift = (out - z).norm() / z.norm()
        assert drift < 0.2, f"{mode}: still close to the pretrained token, drift {drift:.3f}"
        out.sum().backward()
        assert q.grad.abs().max() > 0, mode


def test_rollout_matches_the_released_stategru():
    torch.manual_seed(0)
    generator = StateGRU(state_dim=TOKEN_DIM, hidden_dim=TOKEN_DIM, num_layers=2, dropout=0.0).eval()
    state_0 = torch.randn(5, TOKEN_DIM)
    for num_steps in (1, 2, 5, 9):
        expected = generator(state_0, num_steps)[1]
        ours = rollout_hidden(generator, state_0, torch.full((5,), num_steps, dtype=torch.long))
        assert torch.allclose(ours, expected, atol=1e-6), num_steps


def test_rollout_gives_each_record_its_own_step_count():
    torch.manual_seed(0)
    generator = StateGRU(state_dim=TOKEN_DIM, hidden_dim=TOKEN_DIM, num_layers=2, dropout=0.0).eval()
    state_0 = torch.randn(3, TOKEN_DIM)
    steps = torch.tensor([2, 5, 1])
    ours = rollout_hidden(generator, state_0, steps)
    for row, count in enumerate(steps.tolist()):
        expected = generator(state_0[row:row + 1], count)[1]
        assert torch.allclose(ours[row:row + 1], expected, atol=1e-6), count


def test_scale_zero_bypasses_the_fusion_and_reproduces_v0():
    """Plan 7.1 Level 0: with the quaternion contribution off, the pretrained path is untouched."""
    model = _probe(scale=0.0)
    patches, rr, mask, reference, tokens, steps, rhythm = _batch()
    logits = model(patches, rr, mask, reference, tokens, steps, rhythm)

    anchor = tokens[:, 1]
    v0 = model.head(torch.cat((
        model.norm_struct(anchor),
        model.norm_dynamic(rollout_hidden(model.state_generator, anchor, steps)),
        rhythm,
    ), dim=-1))
    assert torch.equal(logits, v0)
    # The quaternion branch is unreachable, so it collects no gradient.
    logits.sum().backward()
    assert model.encoder.net[1].weight.grad is None or model.encoder.net[1].weight.grad.abs().max() == 0


def test_only_beat_token_one_can_reach_the_output():
    """The released architecture reads token 1 alone; V2 may not change that.

    ``emb_struct`` is beat token 1 and the GRU is unrolled from it, so altering any other
    beat leaves the logits untouched. This is the ceiling on what V2 can do.
    """
    model = _probe(scale=1.0)
    patches, rr, mask, reference, tokens, steps, rhythm = _batch()
    before = model(patches, rr, mask, reference, tokens, steps, rhythm)

    other = tokens.clone()
    other[:, 2:] = torch.randn_like(other[:, 2:]) * 10  # every beat except 0 and 1
    after = model(patches, rr, mask, reference, other, steps, rhythm)
    assert torch.equal(before, after)

    anchor_changed = tokens.clone()
    anchor_changed[:, 1] += 1.0
    assert not torch.allclose(before, model(patches, rr, mask, reference, anchor_changed, steps, rhythm))


def test_forward_is_finite_with_the_ptbxl_shape_and_trains_the_right_parameters():
    model = _probe(scale=1.0)
    logits = model(*_batch())
    assert logits.shape == (4, 5) and torch.isfinite(logits).all()

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert not any(name.startswith(("state_generator", "norm_struct", "norm_dynamic")) for name in trainable)
    assert any(name.startswith("encoder") for name in trainable)
    assert any(name.startswith("fusion") for name in trainable)

    model.train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    for _ in range(2):  # the branch starts learning once the fusion's zero block moves
        optimizer.zero_grad()
        model(*_batch()).sum().backward()
        optimizer.step()
    assert model.encoder.net[1].weight.grad.abs().max() > 0
    assert model.state_generator.gru.weight_ih_l0.grad is None


def test_pretrained_pieces_stay_in_evaluation_mode():
    model = _probe(scale=1.0).train()
    assert not model.state_generator.training and not model.norm_struct.training
    assert model.encoder.training
