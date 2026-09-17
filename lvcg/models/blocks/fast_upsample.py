"""A faster 2x linear upsampling, numerically equivalent to ``F.interpolate``.

``BeatDecoder`` doubles the length five times per decode with
``F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)``, and decodes three
times per pretraining step. On CUDA that kernel, and above all its backward, is bound by
memory transfer rather than compute: profiled at batch 64 it took 52% of the GPU time of a
step, which is why a faster GPU or a larger batch barely helped.

For a factor of exactly 2 the interpolation has a closed form. Output sample k reads the
source coordinate (k + 0.5) / 2 - 0.5, clamped at 0, so

    y[2i]   = 0.25 x[i-1] + 0.75 x[i]      (y[0] = x[0])
    y[2i+1] = 0.75 x[i]   + 0.25 x[i+1]    (y[-1] = x[-1])

with the neighbour outside either end replaced by the end sample, which reproduces both
clamps. The weights are exact binary fractions and the two terms are added in the same
order as the kernel, so the result matches it to rounding, in values and gradients.
"""

import torch


def upsample_linear_x2(x: torch.Tensor) -> torch.Tensor:
    """[..., L] -> [..., 2L], equal to linear ``F.interpolate`` with scale 2."""
    left = torch.cat((x[..., :1], x[..., :-1]), dim=-1)
    right = torch.cat((x[..., 1:], x[..., -1:]), dim=-1)
    even = 0.25 * left + 0.75 * x
    odd = 0.75 * x + 0.25 * right
    return torch.stack((even, odd), dim=-1).flatten(-2)
