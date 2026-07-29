"""Typed tensor contracts shared across data, model, and engine boundaries."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class VoxelBatch:
    """Batched hard voxels and sparse coordinates.

    Coordinates use ``[batch, z, y, x]`` order.
    """

    voxels: Tensor
    num_points: Tensor
    coordinates: Tensor
    batch_size: int

    def __post_init__(self) -> None:
        if self.voxels.ndim != 3:
            raise ValueError("voxels must have shape [M, P, F]")
        num_voxels, max_points, _ = self.voxels.shape
        if self.num_points.shape != (num_voxels,):
            raise ValueError("num_points must have shape [M]")
        if self.coordinates.shape != (num_voxels, 4):
            raise ValueError("coordinates must have shape [M, 4]")
        if self.coordinates.dtype not in (torch.int32, torch.int64):
            raise ValueError("coordinates must use an integer dtype")
        if self.num_points.dtype not in (torch.int32, torch.int64):
            raise ValueError("num_points must use an integer dtype")
        if self.num_points.device != self.voxels.device or self.coordinates.device != self.voxels.device:
            raise ValueError("voxel batch tensors must be on the same device")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if torch.any(self.num_points <= 0) or torch.any(self.num_points > max_points):
            raise ValueError("num_points must be in [1, P]")
        if num_voxels:
            batch_indices = self.coordinates[:, 0]
            if torch.any(batch_indices < 0) or torch.any(batch_indices >= self.batch_size):
                raise ValueError("coordinate batch indices are out of range")


@dataclass(frozen=True)
class TaskTargets:
    """Batched targets for one CenterHead task."""

    heatmap: Tensor
    annotation: Tensor
    indices: Tensor
    mask: Tensor
    categories: Tensor

    def __post_init__(self) -> None:
        if self.heatmap.ndim != 4:
            raise ValueError("heatmap must have shape [B, C, H, W]")
        batch_size = self.heatmap.shape[0]
        if self.annotation.ndim != 3 or self.annotation.shape[0] != batch_size:
            raise ValueError("annotation must have shape [B, M, 10]")
        if self.annotation.shape[2] != 10:
            raise ValueError("annotation must contain 10 regression codes")
        object_shape = self.annotation.shape[:2]
        if self.indices.shape != object_shape:
            raise ValueError("indices must have shape [B, M]")
        if self.mask.shape != object_shape:
            raise ValueError("mask must have shape [B, M]")
        if self.categories.shape != object_shape:
            raise ValueError("categories must have shape [B, M]")
        if self.indices.dtype != torch.int64 or self.categories.dtype != torch.int64:
            raise ValueError("indices and categories must use int64")
        if self.mask.dtype not in (torch.uint8, torch.bool):
            raise ValueError("mask must use uint8 or bool")
        tensors = (self.annotation, self.indices, self.mask, self.categories)
        if any(tensor.device != self.heatmap.device for tensor in tensors):
            raise ValueError("all task targets must be on the same device")
        if self.annotation.dtype != self.heatmap.dtype:
            raise ValueError("heatmap and annotation must use the same dtype")


@dataclass(frozen=True)
class TaskPredictions:
    """NCHW prediction maps emitted by one CenterHead task."""

    heatmap: Tensor
    center_offset: Tensor
    height: Tensor
    dimensions: Tensor
    rotation: Tensor
    velocity: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if self.heatmap.ndim != 4 or self.heatmap.shape[1] <= 0:
            raise ValueError("heatmap must have shape [B, C, H, W] with C > 0")
        batch, _, height, width = self.heatmap.shape
        expected = {
            "center_offset": (batch, 2, height, width),
            "height": (batch, 1, height, width),
            "dimensions": (batch, 3, height, width),
            "rotation": (batch, 2, height, width),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.velocity is not None and self.velocity.shape != (batch, 2, height, width):
            raise ValueError(f"velocity must have shape {(batch, 2, height, width)}")

        tensors = [
            self.center_offset,
            self.height,
            self.dimensions,
            self.rotation,
        ]
        if self.velocity is not None:
            tensors.append(self.velocity)
        if any(tensor.device != self.heatmap.device for tensor in tensors):
            raise ValueError("all task predictions must be on the same device")
        if any(tensor.dtype != self.heatmap.dtype for tensor in tensors):
            raise ValueError("all task predictions must use the same dtype")

    def as_dict(self) -> Dict[str, Tensor]:
        """Return official CenterHead keys without exposing mutable internal state."""

        values = {
            "hm": self.heatmap,
            "reg": self.center_offset,
            "height": self.height,
            "dim": self.dimensions,
            "rot": self.rotation,
        }
        if self.velocity is not None:
            values["vel"] = self.velocity
        return values
