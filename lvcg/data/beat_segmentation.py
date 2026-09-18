"""Beat Segmentation for LVCG.

Pipeline Role: VCG [B, 3, T] + ECG [B, 12, T] -> V_beats [B, N, 3, P], RR [B, N], mask [B, N]

Reconstructed from paper Appendix A.4 and from how ``lvcg/models`` consumes the result.
The paper fixes the algorithm; the call sites fix three contract details that the
paper leaves implicit:

* Beats are R-to-R intervals, not windows centred on the peak. Index 0 is the partial
  interval [0, r_0) and index 1 the first complete beat, which is why the model takes
  ``states[:, 1]`` as its anchor and the TTT path strips ``[1:-1]``.
* ``BeatStitcher`` lays the valid beats end to end, each resampled back to its
  ``rr_intervals`` length, starting at sample 0. The valid intervals must therefore
  partition [0, T) in order, and ``rr_intervals`` must be each interval's own length
  in samples -- including the two partial boundary intervals.
* ``GlobalRREmbedding`` and ``masked_fill(~mask)`` need a boolean mask and float RRs.

Numerical choices the paper does not state, fixed here:

* R-peak search on the reference lead z-scored as in A.4 (mean removed, divided by the
  standard deviation when it exceeds 1e-6), with a height of 2.0 standard deviations and
  a minimum distance of ``min_rr_sec``. "Too few peaks" means fewer than three -- the
  minimum that yields the four intervals ``LVCG.forward_train`` asserts -- and triggers
  one retry at 1.0 standard deviation. The height was calibrated on 484 training-fold
  PTB-XL records (100 Hz) against an independent detector on the 500 Hz VCG magnitude, never
  against labels: heights 0.5 / 1.0 / 1.5 / 2.0 / 2.5 gave peak F1 80.7 / 84.7 / 86.7 /
  87.2 / 86.9 (precision 91.1%, recall 83.7% at 2.0). The remaining misses are what a
  single reference lead cannot see -- an inverted or low-amplitude QRS in lead II -- and
  belong to the paper's method, not to this threshold.
* Physiological R-R bounds 0.3-2.0 s. A peak closer than the lower bound to the
  previous kept peak is dropped, merging the two intervals. An interval longer than
  the upper bound is kept and stays valid: removing it would break the partition the
  stitcher depends on.
* More intervals than ``max_beats``: the first ``max_beats - 1`` are kept and the rest
  merge into one final interval ending at T, so coverage is preserved.
* Each interval [s, e) is resampled to P points at s + k (e - s - 1) / (P - 1), with
  linear interpolation, so the first and last samples of the interval are kept exactly.

Peak detection runs on the detached reference lead on the CPU, as the SciPy routine
requires. The resampling itself gathers from ``vcg`` on its own device, so gradients
flow back into the VCG -- which matters as soon as the ECG-to-VCG lift is learnable.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks


class BeatSegmenter(nn.Module):
    """Segment VCG into individual beats using R-peak detection.

    Input: VCG [B, 3, T] and ECG [B, L, T]; peaks are found on ECG lead ``rr_lead_idx``.
    Output: V_beats [B, N, 3, P], rr_intervals [B, N] (float, samples), beat_mask [B, N]
    (bool), with N the largest interval count in the batch and padding marked invalid.
    """

    def __init__(
        self,
        beat_len: int = 128,
        fs: int = 100,
        max_beats: int = 20,
        min_rr_sec: float = 0.3,
        max_rr_sec: float = 2.0,
        height: float = 2.0,
        fallback_height: float = 1.0,
        min_peaks: int = 3,
    ):
        """
        Args:
            beat_len: Output beat length P (after resampling)
            fs: Sampling rate of the input, Hz
            max_beats: Cap on intervals per record, boundary intervals included
            min_rr_sec, max_rr_sec: Physiological R-R bounds, seconds
            height: Peak height on the z-scored reference lead
            fallback_height: Lower height used once when too few peaks are found
            min_peaks: Fewer peaks than this counts as "too few"
        """
        super().__init__()
        if beat_len < 2 or max_beats < 1:
            raise ValueError("beat_len must be >= 2 and max_beats >= 1")
        self.beat_len = int(beat_len)
        self.fs = int(fs)
        self.max_beats = int(max_beats)
        self.min_rr = max(1, int(round(min_rr_sec * fs)))
        self.max_rr = int(round(max_rr_sec * fs))
        self.height = float(height)
        self.fallback_height = float(fallback_height)
        self.min_peaks = int(min_peaks)

    def _detect_r_peaks(self, lead_signal: np.ndarray) -> np.ndarray:
        """Detect R-peaks in a single lead signal.

        Args:
            lead_signal: [T] reference lead
        Returns:
            Sorted peak indices
        """
        signal = np.asarray(lead_signal, dtype=np.float64)
        signal = signal - signal.mean()
        std = signal.std()
        if std > 1e-6:
            signal = signal / std
        peaks, _ = find_peaks(signal, height=self.height, distance=self.min_rr)
        if len(peaks) < self.min_peaks:
            peaks, _ = find_peaks(signal, height=self.fallback_height, distance=self.min_rr)
        return peaks.astype(np.int64)

    def _get_beat_boundaries(self, peaks: np.ndarray, length: int) -> List[Tuple[int, int]]:
        """Get beat boundaries from R-peaks, ensuring full coverage of [0, T]."""
        kept: List[int] = []
        for peak in sorted(int(p) for p in peaks if 0 < p < length):
            if not kept or peak - kept[-1] >= self.min_rr:
                kept.append(peak)
        edges = [0, *kept, length]
        intervals = [(s, e) for s, e in zip(edges[:-1], edges[1:]) if e > s]
        if len(intervals) > self.max_beats:
            head = intervals[: self.max_beats - 1]
            start = intervals[self.max_beats - 1][0]
            intervals = [*head, (start, length)]
        return intervals

    def r_peaks(
        self, ecg: torch.Tensor, rr_lead_idx: int = 1, max_peaks: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """R-peak sample positions per record: (peaks [B, max_peaks], mask [B, max_peaks]).

        The same detection ``forward`` segments with, exposed for the phase windows of
        V4. Padding positions are 0 and masked; a record with more peaks than
        ``max_peaks`` keeps the first ones, which is what segmentation does as well.
        """
        reference = ecg[:, rr_lead_idx].detach().float().cpu().numpy()
        records = [self._detect_r_peaks(lead) for lead in reference]
        batch = len(records)
        peaks = torch.zeros(batch, max_peaks, dtype=torch.long)
        mask = torch.zeros(batch, max_peaks, dtype=torch.bool)
        for row, found in enumerate(records):
            kept = found[:max_peaks]
            peaks[row, : len(kept)] = torch.as_tensor(kept, dtype=torch.long)
            mask[row, : len(kept)] = True
        return peaks.to(ecg.device), mask.to(ecg.device)

    def forward(
        self, vcg: torch.Tensor, ecg: torch.Tensor, rr_lead_idx: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Segment VCG into beats.

        Args:
            vcg: [B, 3, T] - VCG trajectory
            ecg: [B, L, T] - ECG used only to locate R-peaks
            rr_lead_idx: reference lead (1 = lead II)

        Returns:
            V_beats: [B, N, 3, P]
            rr_intervals: [B, N] interval lengths in samples (0 for padding)
            beat_mask: [B, N] True for real intervals
        """
        if vcg.ndim != 3 or vcg.shape[1] != 3:
            raise ValueError(f"vcg must be [B, 3, T], got {tuple(vcg.shape)}")
        batch, _, length = vcg.shape
        if ecg.shape[0] != batch or ecg.shape[-1] != length:
            raise ValueError("vcg and ecg must share batch size and length")
        reference = ecg[:, rr_lead_idx].detach().float().cpu().numpy()
        records = [
            self._get_beat_boundaries(self._detect_r_peaks(lead), length) for lead in reference
        ]
        count = max(len(intervals) for intervals in records)
        starts = np.zeros((batch, count), dtype=np.float64)
        spans = np.zeros((batch, count), dtype=np.float64)
        mask = np.zeros((batch, count), dtype=bool)
        for row, intervals in enumerate(records):
            for column, (start, end) in enumerate(intervals):
                starts[row, column] = start
                spans[row, column] = end - start
                mask[row, column] = True

        device = vcg.device
        steps = torch.linspace(0.0, 1.0, self.beat_len, device=device, dtype=torch.float64)
        starts_t = torch.as_tensor(starts, device=device)
        spans_t = torch.as_tensor(spans, device=device)
        # [B, N, P] fractional sample positions inside each interval.
        position = starts_t[..., None] + (spans_t[..., None] - 1.0).clamp_min(0.0) * steps
        lower = position.floor().long().clamp(0, length - 1)
        upper = (lower + 1).clamp(max=length - 1)
        weight = (position - lower.to(position.dtype)).to(vcg.dtype)

        def gather(index: torch.Tensor) -> torch.Tensor:
            flat = index.reshape(batch, 1, -1).expand(-1, 3, -1)
            return vcg.gather(2, flat).reshape(batch, 3, count, self.beat_len)

        beats = gather(lower) * (1 - weight[:, None]) + gather(upper) * weight[:, None]
        beats = beats.permute(0, 2, 1, 3).contiguous()  # [B, N, 3, P]
        mask_t = torch.as_tensor(mask, device=device)
        beats = beats * mask_t[:, :, None, None].to(beats.dtype)
        rr_intervals = torch.as_tensor(spans, device=device, dtype=vcg.dtype)
        return beats, rr_intervals, mask_t
