"""Route D: what survives when the spatial observation is reduced.

LVCG's premise is that a latent 3D trajectory can be recovered from a few leads. The
fewer leads, the less determined that trajectory is -- with two leads the least-squares
recovery can only span a plane, with one a line -- so a representation that genuinely
uses rotational geometry should hold up better as leads are removed. Route D measures
that, in the release's own visible-lead protocol.

Two things stay fixed as the lead count drops, and both follow the released model:

* **R peaks still come from lead II of the full recording.** In the release,
  ``forward_inference(use_all_leads=False)`` hands the whole ECG to the segmenter and
  only restricts the *VCG recovery*. Sparsity here is therefore spatial, not temporal:
  the beats are the same, their 3D shape is not.
* **The pseudo-inverse keeps its regularisation** (eps = 0.1), so a rank-deficient
  recovery returns the minimum-norm solution rather than failing. With one or two leads
  the trajectory collapses onto a line or plane; ``geometry_fidelity`` measures exactly
  how far it has collapsed, which is what makes the low end of the sweep interpretable
  instead of merely bad.

The lead subsets are named configurations, not arbitrary picks: the three-lead set is the
release's own default, and the six-lead set is the limb leads, which is what a real
reduced-lead recording gives you.
"""

import torch

from .frame import intrinsic_frame
from .utils import _safe_norm, quaternion_angle, vectors_to_quaternion

# Indices in the model's lead order (mimic): I, II, III, aVR, aVF, aVL, V1..V6.
LEAD_SETS = {
    12: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    8: (0, 1, 6, 7, 8, 9, 10, 11),      # the two independent limb leads plus the chest leads
    6: (0, 1, 2, 3, 4, 5),              # the limb leads, a real reduced-lead recording
    3: (0, 1, 6),                       # I, II, V1 -- the release's own sparse default
    2: (0, 1),                          # I and II, the two independent limb leads
    1: (1,),                            # II alone, the lead the R peaks come from
}


def visible_leads(count):
    if count not in LEAD_SETS:
        raise ValueError(f"No lead set for {count} leads; expected one of {sorted(LEAD_SETS)}")
    return LEAD_SETS[count]


def recover_vcg(backbone, ecg, leads):
    """The latent VCG the frozen model would recover from those leads only: [B, 3, T]."""
    index = torch.as_tensor(leads, device=ecg.device, dtype=torch.long)
    directions = backbone.all_lead_directions[index].unsqueeze(0).expand(ecg.shape[0], -1, -1)
    return backbone.vcg_inverse(ecg[:, index], directions)


def geometry_fidelity(reference, recovered, eps=1e-8):
    """How much of the 12-lead geometry survives a sparse recovery.

    Returns, per record:

    * ``direction_cosine`` -- mean |cos| between the two trajectories' unit directions,
      1.0 when every instantaneous direction is preserved;
    * ``magnitude_ratio``  -- the recovered trajectory's scale relative to the reference;
    * ``planarity``        -- the share of the recovered trajectory's variance outside its
      own best-fit plane, which falls to 0 once the recovery has collapsed.
    """
    a = reference.transpose(1, 2)  # [B, T, 3]
    b = recovered.transpose(1, 2)
    cosine = (a * b).sum(-1) / (_safe_norm(a) * _safe_norm(b) + eps)
    magnitude = _safe_norm(b).mean(-1) / (_safe_norm(a).mean(-1) + eps)

    centred = b - b.mean(1, keepdim=True)
    covariance = torch.einsum("bti,btj->bij", centred, centred)
    eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(0)
    planarity = (eigenvalues[..., 0] / eigenvalues.sum(-1).clamp_min(eps)).to(b.dtype)
    return {"direction_cosine": cosine.abs().mean(-1), "magnitude_ratio": magnitude,
            "planarity": planarity}


def frame_agreement(reference_beats, sparse_beats, beat_mask, degrees=True):
    """Angle between the intrinsic frames of the two recoveries: [B].

    Route C's frame is what route D expects to degrade gracefully -- if the geometry is
    still readable from few leads, the frame built from it should still point the same
    way. Reported as the rotation angle taking one frame's x axis to the other's.
    """
    reference, _ = intrinsic_frame(reference_beats, beat_mask)
    sparse, _ = intrinsic_frame(sparse_beats, beat_mask)
    angle = quaternion_angle(vectors_to_quaternion(reference[..., 0], sparse[..., 0]))
    return angle * 180.0 / torch.pi if degrees else angle
