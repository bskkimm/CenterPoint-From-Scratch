"""CenterNet heatmap primitives used by CenterPoint target generation."""

import math
from typing import Sequence, Tuple

import torch
from torch import Tensor


def gaussian_radius(det_size: Sequence[float], min_overlap: float = 0.5) -> float:
    """Return the official CornerNet Gaussian radius for ``(height, width)``.

    The denominators intentionally match the official CenterPoint implementation rather than a
    generalized quadratic solver.
    """

    height, width = float(det_size[0]), float(det_size[1])

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1**2 - 4 * a1 * c1)) / 2

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2**2 - 4 * a2 * c2)) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3**2 - 4 * a3 * c3)) / 2

    return min(r1, r2, r3)


def gaussian_2d(
    shape: Tuple[int, int],
    sigma: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Create an unnormalized centered 2D Gaussian with official tail truncation."""

    height, width = shape
    y_radius = (height - 1.0) / 2.0
    x_radius = (width - 1.0) / 2.0
    y = torch.arange(height, dtype=dtype, device=device) - y_radius
    x = torch.arange(width, dtype=dtype, device=device) - x_radius
    gaussian = torch.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma**2))
    gaussian[gaussian < torch.finfo(dtype).eps * gaussian.max()] = 0
    return gaussian


def draw_gaussian(heatmap: Tensor, center: Sequence[float], radius: int, k: float = 1.0) -> Tensor:
    """Draw a cropped Gaussian onto a 2D heatmap using elementwise maximum."""

    if heatmap.ndim != 2:
        raise ValueError("heatmap must have shape [height, width]")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    diameter = 2 * radius + 1
    gaussian = gaussian_2d(
        (diameter, diameter),
        sigma=diameter / 6,
        dtype=heatmap.dtype,
        device=heatmap.device,
    )

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom,
        radius - left : radius + right,
    ]
    if masked_heatmap.numel() and masked_gaussian.numel():
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
