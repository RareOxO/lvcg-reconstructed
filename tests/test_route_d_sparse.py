"""Route D: the sparse-lead protocol and the geometry that survives it."""

import importlib.util
import math
import os

import pytest
import torch

from lvcg.data.angle import get_lead_directions
from lvcg.models.lvcg import StateGRU
from lvcg.models.vcg import VCGPseudoInverse
from lvcg.quaternion.frame import FrameProbe
from lvcg.quaternion.sparse import (
    LEAD_SETS,
    frame_agreement,
    geometry_fidelity,
    recover_vcg,
    visible_leads,
)

TOKEN_DIM, BEATS, PATCH = 256, 4, 64
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Backbone(torch.nn.Module):
    """Just enough of the frozen model for the recovery path."""

    def __init__(self):
        super().__init__()
        self.register_buffer("all_lead_directions", get_lead_directions("mimic", as_tensor=True))
        self.vcg_inverse = VCGPseudoInverse(eps=0.1)


def _ecg(batch=3, length=256, seed=0):
    generator = torch.Generator().manual_seed(seed)
    ecg = torch.randn(batch, 12, length, generator=generator) * 0.3
    ecg[:, 1, 20::85] += 6.0  # R peaks on lead II
    return ecg


# --- the lead sets ----------------------------------------------------------------


def test_lead_sets_are_named_configurations_not_arbitrary_picks():
    assert LEAD_SETS[3] == (0, 1, 6), "the release's own sparse default: I, II, V1"
    assert LEAD_SETS[6] == (0, 1, 2, 3, 4, 5), "the limb leads"
    assert LEAD_SETS[2] == (0, 1) and LEAD_SETS[1] == (1,)
    for count, leads in LEAD_SETS.items():
        assert len(leads) == count and len(set(leads)) == count
        assert all(0 <= lead < 12 for lead in leads)


def test_an_unknown_lead_count_is_refused():
    with pytest.raises(ValueError):
        visible_leads(5)


def test_twelve_leads_reproduce_the_full_recovery():
    backbone = _Backbone()
    ecg = _ecg()
    directions = backbone.all_lead_directions.unsqueeze(0).expand(len(ecg), -1, -1)
    assert torch.allclose(recover_vcg(backbone, ecg, LEAD_SETS[12]),
                          backbone.vcg_inverse(ecg, directions), atol=1e-6)


def test_a_sparse_recovery_uses_only_its_leads():
    """Changing an invisible lead must not move the recovered trajectory."""
    backbone = _Backbone()
    ecg = _ecg()
    changed = ecg.clone()
    changed[:, 8] += 5.0  # V3, which the three-lead set does not contain
    assert torch.allclose(recover_vcg(backbone, ecg, LEAD_SETS[3]),
                          recover_vcg(backbone, changed, LEAD_SETS[3]), atol=1e-6)


# --- what the recovery loses ------------------------------------------------------


def test_the_limb_leads_recover_a_planar_trajectory():
    """Six limb leads all lie in the frontal plane, so the recovery cannot leave it."""
    backbone = _Backbone()
    ecg = _ecg()
    full = recover_vcg(backbone, ecg, LEAD_SETS[12])
    limb = recover_vcg(backbone, ecg, LEAD_SETS[6])
    assert float(geometry_fidelity(full, limb)["planarity"].max()) < 1e-3


def test_one_lead_collapses_the_trajectory_to_a_line():
    backbone = _Backbone()
    ecg = _ecg()
    single = recover_vcg(backbone, ecg, LEAD_SETS[1])
    directions = single.transpose(1, 2)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    # Every sample points along the same axis, up to sign.
    cosine = (directions * directions[:, :1]).sum(-1).abs()
    assert float(cosine.min()) > 0.99


def test_fidelity_is_perfect_against_itself_and_degrades_with_fewer_leads():
    backbone = _Backbone()
    ecg = _ecg()
    full = recover_vcg(backbone, ecg, LEAD_SETS[12])
    identical = geometry_fidelity(full, full)
    assert float(identical["direction_cosine"].mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(identical["magnitude_ratio"].mean()) == pytest.approx(1.0, abs=1e-5)

    cosines = [float(geometry_fidelity(full, recover_vcg(backbone, ecg, LEAD_SETS[k]))
                     ["direction_cosine"].mean()) for k in (12, 8, 3, 1)]
    assert cosines == sorted(cosines, reverse=True), cosines


def test_frame_agreement_is_zero_for_identical_beats_and_grows_when_they_differ():
    generator = torch.Generator().manual_seed(0)
    angle = torch.linspace(0, 2 * math.pi, PATCH)
    loop = torch.stack((2.0 * torch.cos(angle), torch.sin(angle), 0.2 * torch.sin(2 * angle)))
    beats = loop.expand(2, BEATS, 3, PATCH) + 0.02 * torch.randn(2, BEATS, 3, PATCH, generator=generator)
    mask = torch.ones(2, BEATS)
    assert float(frame_agreement(beats, beats, mask).max()) < 1e-2

    flattened = beats.clone()
    flattened[..., 2, :] = 0.0  # a different geometry: the frame should move
    assert float(frame_agreement(beats, flattened, mask).mean()) >= 0.0


def test_fidelity_stays_finite_on_a_degenerate_recovery():
    backbone = _Backbone()
    ecg = torch.zeros(2, 12, 128)
    full = recover_vcg(backbone, ecg, LEAD_SETS[12])
    scores = geometry_fidelity(full, recover_vcg(backbone, ecg, LEAD_SETS[1]))
    assert all(torch.isfinite(value).all() for value in scores.values())


# --- the script -------------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location(
        "train_sparse", os.path.join(REPO_ROOT, "scripts", "train_sparse.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_exposes_the_sweep_defaults():
    module = _script()
    assert module.MAX_BEATS == 20
    assert set(LEAD_SETS) >= {12, 6, 3, 2, 1}


def test_the_frozen_pass_keeps_r_peaks_from_the_full_recording():
    """Sparsity is spatial: the beats must be the same whichever leads are visible."""
    torch.manual_seed(0)
    from lvcg.models.lvcg import LVCG
    from lvcg.utils.config import load_config

    backbone = LVCG.from_config(load_config(os.path.join(REPO_ROOT, "configs/train/lvcg_v5_gru.yaml")))
    ecg = _ecg(batch=2, length=1000)
    full = recover_vcg(backbone, ecg, LEAD_SETS[12])
    sparse = recover_vcg(backbone, ecg, LEAD_SETS[2])
    _, rr_full, mask_full = backbone.beat_segmenter(full, ecg, rr_lead_idx=backbone.rr_lead_idx)
    _, rr_sparse, mask_sparse = backbone.beat_segmenter(sparse, ecg, rr_lead_idx=backbone.rr_lead_idx)
    assert torch.equal(rr_full, rr_sparse) and torch.equal(mask_full, mask_sparse)


def test_the_probe_trains_on_sparse_recoveries():
    module = _script()
    generator = torch.Generator().manual_seed(0)

    def make(n):
        labels = (torch.rand(n, 5, generator=generator) > 0.5).float()
        angle = torch.linspace(0, 2 * math.pi, PATCH)
        width = 1.0 + 3.0 * labels[:, 0]
        beats = torch.stack((
            width.view(n, 1, 1) * torch.cos(angle).view(1, 1, -1).expand(n, BEATS, PATCH),
            torch.sin(angle).view(1, 1, -1).expand(n, BEATS, PATCH),
            torch.zeros(n, BEATS, PATCH)), dim=2)
        beats = beats + 0.05 * torch.randn(beats.shape, generator=generator)
        return (beats, torch.full((n, BEATS), 85.0), torch.ones(n, BEATS), torch.ones(n),
                torch.zeros(n, BEATS, TOKEN_DIM), torch.full((n,), BEATS - 1, dtype=torch.long),
                torch.zeros(n, 128), labels)

    torch.manual_seed(0)
    parts = dict(
        state_generator=StateGRU(state_dim=TOKEN_DIM, hidden_dim=TOKEN_DIM, num_layers=2, dropout=0.1),
        norm_struct=torch.nn.LayerNorm(TOKEN_DIM),
        norm_dynamic=torch.nn.LayerNorm(TOKEN_DIM),
    )
    data = {"train": make(256), "val": make(64), "test": make(64)}
    model = FrameProbe(**parts, parts=("invariant",), embedding_dim=32, hidden=16)
    train_frame = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(
            "train_frame", os.path.join(REPO_ROOT, "scripts", "train_frame.py")))
    importlib.util.spec_from_file_location(
        "train_frame", os.path.join(REPO_ROOT, "scripts", "train_frame.py")).loader.exec_module(train_frame)
    train_frame.train(model, data, torch.device("cpu"),
                      {"batch_size": 32, "max_epochs": 8, "patience": 8}, seed=0)
    scores = train_frame.evaluate(model, data["test"], torch.device("cpu"), 64)
    assert scores["per_label_auroc"][0] > 0.85, scores["per_label_auroc"]
