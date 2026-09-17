"""MIMIC-IV ECG data loading utilities.

Reconstructed. The release records the functions below and that
``preprocess_mimic_record`` "reorder[s] leads, resample[s], bandpass filter[s], and
normalize[s]" in that order; the paper (Appendix B.1, Table 8) gives the targets: 100 Hz,
band-pass 0.67-40 Hz, per-lead z-score. What neither records, and is therefore chosen
here and stated:

* Leads are reordered by *name* to the model's lead order, so a record whose header
  lists the leads differently is still fed consistently.
* Missing samples (NaN) are set to 0 before filtering; a filter would otherwise spread
  one NaN across the whole lead.
* Resampling uses ``scipy.signal.resample_poly`` at the exact rational ratio, whose FIR
  low-pass removes content above the new Nyquist before decimating.
* The band-pass is a 4th-order Butterworth applied forwards and backwards
  (``sosfiltfilt``), so it adds no phase shift and the fiducial timing is kept.
* The signal is cut or zero-padded at the end to ``time_len`` samples. MIMIC-IV-ECG
  records are all 10 s at 500 Hz, so at 100 Hz this is a no-op on real data.
"""

import json
import os
from math import gcd
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

from .angle import LEAD_ORDERS

DEFAULT_BANDPASS = (0.67, 40.0)
FILTER_ORDER = 4


def load_mimic_manifest(path: str) -> List[Dict]:
    """Load the MIMIC manifest (JSONL format): one object per line with ``ecg_path``."""
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "ecg_path" not in row:
                raise ValueError(f"{path}:{number} has no ecg_path")
            records.append(row)
    return records


def load_wfdb_record(ecg_path: str) -> Tuple[np.ndarray, int, List[str]]:
    """Load a single WFDB record and return (data, fs, sig_names).

    Data shape: [T, L], physical units as stored in the header (mV for MIMIC-IV-ECG).
    ``ecg_path`` is the record stem, without ``.hea`` / ``.dat``.
    """
    stem = ecg_path[:-4] if ecg_path.endswith((".hea", ".dat")) else ecg_path
    if not os.path.exists(stem + ".hea"):
        raise FileNotFoundError(f"No WFDB header at {stem}.hea")
    signal, fields = wfdb.rdsamp(stem)
    return signal.astype(np.float32), int(fields["fs"]), list(fields["sig_name"])


def normalize_ecg(data: np.ndarray, scale_method: str = "zscore", eps: float = 1e-8) -> np.ndarray:
    """Normalize ECG data using specified scaling method.

    Args:
        data: [L, T]; every method works per lead, along time.
        scale_method: ``zscore`` (the paper's per-lead z-score), ``minmax`` (per lead to
            [0, 1]) or ``none``.
    """
    data = np.asarray(data, dtype=np.float32)
    if scale_method == "zscore":
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        return (data - mean) / (std + eps)
    if scale_method == "minmax":
        low = data.min(axis=-1, keepdims=True)
        high = data.max(axis=-1, keepdims=True)
        return (data - low) / (high - low + eps)
    if scale_method in (None, "none"):
        return data
    raise ValueError(f"Unknown scale_method: {scale_method}")


def _reorder_by_name(data: np.ndarray, sig_names: Sequence[str], target: Sequence[str]) -> np.ndarray:
    lookup = {name.lower(): index for index, name in enumerate(sig_names)}
    missing = [name for name in target if name.lower() not in lookup]
    if missing:
        raise ValueError(f"Record lacks leads {missing}; has {list(sig_names)}")
    return data[:, [lookup[name.lower()] for name in target]]


def preprocess_mimic_record(
    data: np.ndarray,
    fs: int,
    sig_names: Sequence[str],
    target_fs: int = 100,
    time_len: int = 1000,
    bandpass: Optional[Tuple[float, float]] = DEFAULT_BANDPASS,
    scale_method: str = "zscore",
    lead_order: str = "mimic",
) -> np.ndarray:
    """Preprocess raw ECG data: reorder leads, resample, bandpass filter, and normalize.

    Args:
        data: [T, L] raw signal as returned by ``load_wfdb_record``.
        fs: its sampling rate.
    Returns:
        float32 [L, time_len] in ``lead_order``.
    """
    signal = _reorder_by_name(np.asarray(data, dtype=np.float64), sig_names, LEAD_ORDERS[lead_order])
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).T  # [L, T]
    if fs != target_fs:
        factor = gcd(int(fs), int(target_fs))
        signal = resample_poly(signal, int(target_fs) // factor, int(fs) // factor, axis=-1)
    if bandpass is not None:
        low, high = bandpass
        if not 0 < low < high < target_fs / 2:
            raise ValueError(f"Band-pass {bandpass} must lie inside (0, {target_fs / 2}) Hz")
        sos = butter(FILTER_ORDER, [low, high], btype="bandpass", fs=target_fs, output="sos")
        signal = sosfiltfilt(sos, signal, axis=-1)
    length = signal.shape[-1]
    if length >= time_len:
        signal = signal[:, :time_len]
    else:
        signal = np.pad(signal, ((0, 0), (0, time_len - length)))
    return normalize_ecg(signal, scale_method)


class MimicLoader:
    """Convenience class to iterate over MIMIC records from a manifest."""

    def __init__(
        self,
        manifest_path: str,
        target_fs: int = 100,
        time_len: int = 1000,
        bandpass: Optional[Tuple[float, float]] = DEFAULT_BANDPASS,
        scale_method: str = "zscore",
        lead_order: str = "mimic",
    ):
        """
        Args:
            manifest_path: Path to MIMIC manifest (JSONL format)
            target_fs, time_len, bandpass, scale_method, lead_order: see
                ``preprocess_mimic_record``.
        """
        self.records = load_mimic_manifest(manifest_path)
        self.target_fs = target_fs
        self.time_len = time_len
        self.bandpass = bandpass
        self.scale_method = scale_method
        self.lead_order = lead_order

    def __len__(self) -> int:
        return len(self.records)

    def get_record(self, index: int) -> np.ndarray:
        data, fs, names = load_wfdb_record(self.records[index]["ecg_path"])
        return preprocess_mimic_record(
            data,
            fs,
            names,
            target_fs=self.target_fs,
            time_len=self.time_len,
            bandpass=self.bandpass,
            scale_method=self.scale_method,
            lead_order=self.lead_order,
        )
