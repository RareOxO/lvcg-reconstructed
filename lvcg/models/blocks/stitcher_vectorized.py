"""A vectorised drop-in for ``BeatStitcher``, numerically equivalent to it.

``BeatStitcher`` walks every record and every beat in Python, resampling each beat with
its own ``F.interpolate`` call and reading each R-R length with ``.item()``: at batch 64
and 20 beats that is over a thousand small kernel launches and GPU synchronisations per
step, and it is the CPU bound of pretraining. This module computes the same output with
a fixed handful of tensor operations. It has no parameters or buffers, so checkpoints are
interchangeable between the two, and ``LVCG(vectorized_stitcher=...)`` chooses between
them; the author's loop stays the default.

Everything the loop does is reproduced, including the behaviour that looks accidental:

* Beats are laid end to end from sample 0 in index order, each resampled to
  ``int(rr)`` samples. A valid beat with ``int(rr) <= 0`` is skipped without moving the
  write position, but still counts as a valid beat for the cross-fade rules below.
* Resampling is ``F.interpolate(mode="linear", align_corners=False)``: the source
  coordinate of output sample k is (k + 0.5) * P / L - 0.5, clamped at 0, and the upper
  neighbour is clamped to the last input sample.
* Cross-fade windows are computed for the beat's full length L, with
  f = max(1, int(L * crossfade_ratio)). A beat that is not the first valid beat gets
  ``linspace(0, 1, f)`` over its first f samples; one that is not the last valid beat gets
  ``linspace(1, 0, f)`` over its last f samples, written second, so it wins where the two
  overlap. With f = 1 those are [0] and [1]: a one-sample fade-in zeroes the sample, a
  one-sample fade-out leaves it at 1.
* Writing stops at ``target_len``; a beat that crosses it keeps the window of its full
  length. Samples no beat reaches stay 0.
* The output is (value * window) / clamp(window, 1e-6). Beats never overlap, so this is
  the beat's value wherever the window is non-zero and 0 where it is zero -- the
  first and last sample of every inner beat -- exactly as in the loop.

Agreement with ``BeatStitcher`` in outputs and gradients is tested on random and real
beats; the remaining difference is float rounding, of order 1e-7.
"""

import torch
import torch.nn as nn


class VectorizedBeatStitcher(nn.Module):
    """Stitch decoded beats back to a continuous waveform using RR intervals.

    Input: V_hat_beats [B, N, 3, P] + rr_intervals [B, N] + beat_mask [B, N]
    Output: V_hat [B, 3, target_len]
    """

    def __init__(self, beat_len: int = 128, target_len: int = 1024, crossfade_ratio: float = 0.1):
        super().__init__()
        self.beat_len = beat_len
        self.target_len = target_len
        self.crossfade_ratio = crossfade_ratio

    def forward(
        self,
        V_hat_beats: torch.Tensor,
        rr_intervals: torch.Tensor,
        beat_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C, P = V_hat_beats.shape
        T = self.target_len
        device, dtype = V_hat_beats.device, V_hat_beats.dtype

        # int(rr) truncates toward zero, as .long() does.
        lengths = rr_intervals.detach().to(torch.long)
        valid = beat_mask.to(torch.bool)
        placed = valid & (lengths > 0)
        span = torch.where(placed, lengths, torch.zeros_like(lengths))
        ends = span.cumsum(dim=1)  # write position after each beat
        starts = ends - span
        rank = valid.to(torch.long).cumsum(dim=1) - 1  # beat_idx among valid beats
        count = valid.to(torch.long).sum(dim=1, keepdim=True)  # num_valid

        # For every output sample, the first beat whose end lies beyond it. Skipped and
        # padded beats have zero width, so they are never selected.
        t = torch.arange(T, device=device).expand(B, T).contiguous()
        beat = torch.searchsorted(ends.contiguous(), t, right=True)
        covered = beat < N
        beat = beat.clamp(max=N - 1)
        length = lengths.gather(1, beat)
        k = t - starts.gather(1, beat)  # position inside the beat
        safe_length = length.clamp(min=1)

        # F.interpolate, linear, align_corners=False, size=length.
        scale = P / safe_length.to(dtype)
        source = ((k.to(dtype) + 0.5) * scale - 0.5).clamp(min=0)
        lower = source.to(torch.long).clamp(max=P - 1)
        upper = lower + (lower < P - 1).to(torch.long)
        upper_weight = (source - lower.to(dtype)).unsqueeze(-1)
        lower_weight = 1 - upper_weight
        flat = V_hat_beats.permute(0, 1, 3, 2).reshape(B, N * P, C)

        def take(index):
            return flat.gather(1, (beat * P + index).unsqueeze(-1).expand(B, T, C))

        values = lower_weight * take(lower) + upper_weight * take(upper)  # [B, T, C]

        # Cross-fade windows.
        fade = (length.to(torch.float64) * self.crossfade_ratio).to(torch.long).clamp(min=1)
        denominator = (fade - 1).clamp(min=1).to(dtype)
        window = torch.ones(B, T, device=device, dtype=dtype)
        fade_in = (rank.gather(1, beat) > 0) & (k < fade)
        window = torch.where(
            fade_in,
            torch.where(fade == 1, torch.zeros_like(window), k.to(dtype) / denominator),
            window,
        )
        fade_out = (rank.gather(1, beat) < count - 1) & (k >= length - fade)
        position = (k - (length - fade)).to(dtype)
        window = torch.where(
            fade_out,
            torch.where(fade == 1, torch.ones_like(window), 1 - position / denominator),
            window,
        )
        window = torch.where(covered, window, torch.zeros_like(window)).unsqueeze(-1)

        V_hat = (values * window) / window.clamp(min=1e-6)
        return V_hat.permute(0, 2, 1).contiguous()
