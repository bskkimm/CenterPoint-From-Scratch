"""CenterPoint box and angle conventions.

The internal nuScenes box layout is ``[x, y, z, w, l, h, vx, vy, yaw]``.
Positive internal yaw rotates clockwise in the XY plane when viewed from +Z.
"""

import math
from typing import Sequence

import torch
from torch import Tensor


INTERNAL_BOX_FIELDS = ("x", "y", "z", "w", "l", "h", "vx", "vy", "yaw")
NMS_BOX_FIELDS = ("x", "y", "z", "length", "width", "h", "yaw")


def limit_period(values: Tensor, offset: float = 0.5, period: float = 2 * math.pi) -> Tensor:
    """Wrap values using the period convention from the official target assigner."""

    return values - torch.floor(values / period + offset) * period


def nuscenes_yaw_to_internal(yaw: Tensor) -> Tensor:
    """Convert nuScenes quaternion yaw to CenterPoint's internal LiDAR yaw."""

    return -yaw - math.pi / 2


def internal_yaw_to_nuscenes(yaw: Tensor) -> Tensor:
    """Convert internal LiDAR yaw to the nuScenes box yaw convention."""

    return -yaw - math.pi / 2


def boxes_to_corners_3d(
    centers: Tensor,
    dimensions: Tensor,
    yaw: Tensor,
    origin: Sequence[float] = (0.5, 0.5, 0.5),
) -> Tensor:
    """Convert center boxes to corners in the official CenterPoint corner order.

    Args:
        centers: Tensor shaped ``[N, 3]`` containing ``x, y, z``.
        dimensions: Tensor shaped ``[N, 3]`` containing ``w, l, h``.
        yaw: Tensor shaped ``[N]`` in the internal clockwise convention.
        origin: Relative point in each box represented by ``centers``.

    Returns:
        Tensor shaped ``[N, 8, 3]``.
    """

    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape [N, 3]")
    if dimensions.shape != centers.shape:
        raise ValueError("dimensions must have shape [N, 3]")
    if yaw.ndim != 1 or yaw.shape[0] != centers.shape[0]:
        raise ValueError("yaw must have shape [N]")
    if len(origin) != 3:
        raise ValueError("origin must contain three values")

    corner_template = centers.new_tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
        ]
    )
    local = (corner_template - centers.new_tensor(origin)) * dimensions[:, None, :]

    cosine = torch.cos(yaw)[:, None]
    sine = torch.sin(yaw)[:, None]
    x = local[..., 0] * cosine + local[..., 1] * sine
    y = -local[..., 0] * sine + local[..., 1] * cosine
    rotated = torch.stack((x, y, local[..., 2]), dim=-1)
    return rotated + centers[:, None, :]


def boxes_to_nms_format(boxes: Tensor) -> Tensor:
    """Convert ``[x,y,z,w,l,h,yaw]`` boxes to the official PCDet NMS format."""

    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("boxes must have shape [N, 7]")

    converted = boxes[:, [0, 1, 2, 4, 3, 5, 6]].clone()
    converted[:, 6] = -converted[:, 6] - math.pi / 2
    return converted
