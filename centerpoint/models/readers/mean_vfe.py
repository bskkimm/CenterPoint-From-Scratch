"""Parameter-free voxel feature extraction."""

import torch
from torch import Tensor, nn


class MeanVoxelFeatureEncoder(nn.Module):
    """Average every input feature over the valid points in each voxel."""

    def __init__(self, num_input_features: int) -> None:
        super().__init__()
        if num_input_features <= 0:
            raise ValueError("num_input_features must be positive")
        self.num_input_features = num_input_features

    def forward(self, features: Tensor, num_points: Tensor) -> Tensor:
        """Encode ``[M, P, F]`` voxels as contiguous ``[M, F]`` means."""

        if features.ndim != 3 or features.shape[-1] != self.num_input_features:
            raise ValueError("features must have shape [M, P, num_input_features]")
        if num_points.ndim != 1 or num_points.shape[0] != features.shape[0]:
            raise ValueError("num_points must have shape [M]")
        if torch.any(num_points <= 0):
            raise ValueError("every voxel must contain at least one point")

        normalizer = num_points.to(dtype=features.dtype, device=features.device).view(-1, 1)
        return (features.sum(dim=1) / normalizer).contiguous()
