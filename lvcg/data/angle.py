"""Lead direction vectors and reorder utilities for ECG multi-lead reconstruction.

Each lead is a linear view of the cardiac vector, e_l(t) = u_l^T v(t), and stacking the
twelve unit directions gives the lead direction matrix A in R^{12x3} (paper Eq. 2).

The vectors are the paper's Table 7, verbatim. Coordinates (Appendix A.3): x runs right
to left, y superior to inferior, z anterior to posterior. The limb leads lie in the
frontal plane at the Einthoven/hexaxial angles I 0, II 60, III 120, aVR -150, aVL -30,
aVF 90 degrees; the precordial leads sit on a transverse ring tilted 10 degrees inferior,
at 110, 70, 40, 10, -20, -50 degrees for V1-V6. Those angles reproduce every published
entry to five decimals, which the tests check.

Directions are stored by lead *name*, so any declared order is physically correct by
construction.

Lead orders. ``"ptbxl"`` is PTB-XL's native order (I, II, III, aVR, aVL, aVF, V1-V6).
``"mimic"`` is MIMIC-IV-ECG's native order, which puts aVF before aVL: every downloaded
MIMIC-IV-ECG header lists I, II, III, aVR, aVF, aVL, V1-V6, and the release's probing
loader swaps aVL and aVF on PTB-XL records "to match MIMIC", which only makes sense if
the model saw MIMIC's native order during pretraining. The paper's Table 7 caption calls
the aVL-before-aVF order "MIMIC order"; the data files and the code disagree with it, and
the table's *vectors* are unaffected because they are looked up here by lead name.
"""

from typing import Sequence, Tuple, Union

import numpy as np
import torch

# Paper Table 7, rows u_l^T = [u_x, u_y, u_z].
_DIRECTIONS_BY_NAME = {
    "I": (1.00000, 0.00000, 0.00000),
    "II": (0.50000, 0.86603, 0.00000),
    "III": (-0.50000, 0.86603, 0.00000),
    "aVR": (-0.86603, -0.50000, 0.00000),
    "aVL": (0.86603, -0.50000, 0.00000),
    "aVF": (0.00000, 1.00000, 0.00000),
    "V1": (-0.33682, 0.17365, 0.92542),
    "V2": (0.33682, 0.17365, 0.92542),
    "V3": (0.75441, 0.17365, 0.63302),
    "V4": (0.96985, 0.17365, 0.17101),
    "V5": (0.92542, 0.17365, -0.33682),
    "V6": (0.63302, 0.17365, -0.75441),
}

LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MIMIC_LEAD_NAMES = ("I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6")
LEAD_ORDERS = {"ptbxl": LEAD_NAMES, "mimic": MIMIC_LEAD_NAMES}

LEAD_DIRECTIONS_PTBXL = np.array(
    [_DIRECTIONS_BY_NAME[name] for name in LEAD_ORDERS["ptbxl"]], dtype=np.float32
)
LEAD_DIRECTIONS_MIMIC = np.array(
    [_DIRECTIONS_BY_NAME[name] for name in LEAD_ORDERS["mimic"]], dtype=np.float32
)


def _order(order: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    if isinstance(order, str):
        if order not in LEAD_ORDERS:
            raise ValueError(f"Unknown lead order {order!r}; expected one of {list(LEAD_ORDERS)}")
        return LEAD_ORDERS[order]
    names = tuple(order)
    unknown = [name for name in names if name not in _DIRECTIONS_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown lead names {unknown}")
    return names


def get_lead_directions(
    order: Union[str, Sequence[str]] = "mimic", as_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Get lead direction vectors in specified order.

    These are the standard 12-lead directions of paper Table 7, returned as [L, 3].
    ``order`` is a named order or an explicit sequence of lead names.
    """
    directions = np.array([_DIRECTIONS_BY_NAME[name] for name in _order(order)], np.float32)
    return torch.from_numpy(directions) if as_tensor else directions


def compute_lead_directions(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Compute 3D direction vectors from lead angles (for dynamic/batched angles).

    theta is the angle inside the frontal (x-y) plane measured from +x, phi the
    elevation out of that plane towards +z, both in radians:

        u = [cos(phi) cos(theta), cos(phi) sin(theta), sin(phi)]

    Any leading batch shape is kept; the result has a trailing dimension of 3.
    """
    return torch.stack(
        (torch.cos(phi) * torch.cos(theta), torch.cos(phi) * torch.sin(theta), torch.sin(phi)),
        dim=-1,
    )


def compute_lead_directions_np(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """NumPy version of compute_lead_directions.

    Args:
        theta: Frontal-plane angle from +x, radians.
        phi: Elevation towards +z, radians.
    """
    theta, phi = np.asarray(theta, dtype=np.float64), np.asarray(phi, dtype=np.float64)
    return np.stack(
        (np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)), axis=-1
    )


def _permutation(source: Sequence[str], target: Sequence[str]) -> list:
    source, target = _order(source), _order(target)
    if sorted(source) != sorted(target):
        raise ValueError("Source and target orders must contain the same leads")
    return [source.index(name) for name in target]


def reorder_leads(
    ecg: Union[np.ndarray, torch.Tensor],
    source_order: Union[str, Sequence[str]],
    target_order: Union[str, Sequence[str]],
    lead_axis: int = -2,
) -> Union[np.ndarray, torch.Tensor]:
    """Reorder ECG leads from source dataset order to target order.

    Pipeline Role: make a record's channel order agree with the rows of the lead
    direction matrix before lifting. ``lead_axis`` defaults to -2 for [..., L, T].
    """
    index = _permutation(source_order, target_order)
    if isinstance(ecg, torch.Tensor):
        return ecg.index_select(lead_axis, torch.as_tensor(index, device=ecg.device))
    return np.take(ecg, index, axis=lead_axis)


def reorder_directions(
    directions: Union[np.ndarray, torch.Tensor],
    source_order: Union[str, Sequence[str]],
    target_order: Union[str, Sequence[str]],
) -> Union[np.ndarray, torch.Tensor]:
    """Reorder lead direction vectors from source order to target order.

    Args:
        directions: [L, 3] rows in ``source_order``.
    """
    return reorder_leads(directions, source_order, target_order, lead_axis=0)


def directions_to_angles(directions: Union[np.ndarray, torch.Tensor]):
    """Convert direction vectors to (theta, phi) angles.

    theta: angle in XY plane from +x; phi: elevation towards +z. Exact inverse of
    ``compute_lead_directions`` for unit vectors.
    """
    if isinstance(directions, torch.Tensor):
        unit = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        theta = torch.atan2(unit[..., 1], unit[..., 0])
        phi = torch.asin(unit[..., 2].clamp(-1.0, 1.0))
        return theta, phi
    directions = np.asarray(directions, dtype=np.float64)
    unit = directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-12)
    return np.arctan2(unit[..., 1], unit[..., 0]), np.arcsin(np.clip(unit[..., 2], -1.0, 1.0))


def get_lead_angles(
    order: Union[str, Sequence[str]] = "mimic", as_tensor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Get lead angles in specified order.

    NOTE: These angles are derived from the Table 7 direction vectors, so they are
    exact for the vectors and not an independent source. Returns [L, 2] = (theta, phi).
    """
    theta, phi = directions_to_angles(get_lead_directions(order))
    angles = np.stack((theta, phi), axis=-1).astype(np.float32)
    return torch.from_numpy(angles) if as_tensor else angles


def reorder_angles(
    angles: Union[np.ndarray, torch.Tensor],
    source_order: Union[str, Sequence[str]],
    target_order: Union[str, Sequence[str]],
) -> Union[np.ndarray, torch.Tensor]:
    """Reorder lead angles from source order to target order.

    Args:
        angles: [L, 2] rows in ``source_order``.
    """
    return reorder_leads(angles, source_order, target_order, lead_axis=0)


def get_visible_lead_angles(lead_angles: torch.Tensor, visible_indices: torch.Tensor):
    """Extract angles for visible leads given indices.

    Args:
        lead_angles: [L, 2]
        visible_indices: [B, K]
    Returns:
        theta, phi: [B, K] each
    """
    selected = lead_angles[visible_indices]
    return selected[..., 0], selected[..., 1]


def get_visible_lead_directions(
    lead_directions: torch.Tensor, visible_indices: torch.Tensor
) -> torch.Tensor:
    """Extract direction vectors for visible leads given indices.

    Args:
        lead_directions: [L, 3]
        visible_indices: [B, K]
    Returns:
        [B, K, 3]
    """
    return lead_directions[visible_indices]
