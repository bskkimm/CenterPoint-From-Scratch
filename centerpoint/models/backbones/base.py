"""Backend-independent sparse-backbone boundary."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SparseBackboneInput:
    """Mean voxel features and coordinates consumed by a sparse backbone."""

    features: Tensor
    coordinates: Tensor
    spatial_shape: Tuple[int, int, int]
    batch_size: int

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must have shape [M, C]")
        if self.coordinates.shape != (self.features.shape[0], 4):
            raise ValueError("coordinates must have shape [M, 4]")
        if self.coordinates.dtype not in (torch.int32, torch.int64):
            raise ValueError("coordinates must use an integer dtype")
        if self.coordinates.device != self.features.device:
            raise ValueError("features and coordinates must be on the same device")
        if len(self.spatial_shape) != 3 or any(size <= 0 for size in self.spatial_shape):
            raise ValueError("spatial_shape must contain positive z, y, x sizes")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.coordinates.shape[0]:
            lower_valid = torch.all(self.coordinates >= 0)
            upper_bounds = self.coordinates.new_tensor(
                (self.batch_size, *self.spatial_shape)
            )
            upper_valid = torch.all(self.coordinates < upper_bounds)
            if not bool(lower_valid and upper_valid):
                raise ValueError("sparse coordinates are outside batch or spatial bounds")


class SparseBackbone(nn.Module, ABC):
    """Validated interface for sparse 3D backbone implementations.

    Subclasses own sparse-kernel integration but must emit a dense NCHW BEV tensor.
    """

    def __init__(self, output_channels: int = 256, output_stride: int = 8) -> None:
        super().__init__()
        if output_channels <= 0 or output_stride <= 0:
            raise ValueError("output channels and stride must be positive")
        self.output_channels = output_channels
        self.output_stride = output_stride

    def forward(self, inputs: SparseBackboneInput) -> Tensor:
        """Run the backend and validate its dense BEV contract."""

        bev = self.forward_sparse(inputs)
        output_height = _ceil_divide(inputs.spatial_shape[1], self.output_stride)
        output_width = _ceil_divide(inputs.spatial_shape[2], self.output_stride)
        expected_shape = (
            inputs.batch_size,
            self.output_channels,
            output_height,
            output_width,
        )
        if bev.shape != expected_shape:
            raise ValueError(
                f"sparse backbone must emit dense BEV shape {expected_shape}, got {tuple(bev.shape)}"
            )
        if bev.device != inputs.features.device:
            raise ValueError("sparse backbone output must remain on the input device")
        if bev.dtype != inputs.features.dtype:
            raise ValueError("sparse backbone output must preserve the feature dtype")
        return bev

    @abstractmethod
    def forward_sparse(self, inputs: SparseBackboneInput) -> Tensor:
        """Implement sparse feature extraction and densification."""


def _ceil_divide(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
