"""Order-preserving hard voxelization compatible with official CenterPoint semantics."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class VoxelizationResult:
    """Compact hard-voxel tensors."""

    voxels: Tensor
    coordinates: Tensor
    num_points: Tensor


class HardVoxelizer:
    """Reference CPU voxelizer preserving input encounter and truncation order."""

    def __init__(
        self,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        max_points_per_voxel: int,
        max_voxels: int,
    ) -> None:
        if len(voxel_size) != 3:
            raise ValueError("voxel_size must contain x, y, z sizes")
        if len(point_cloud_range) != 6:
            raise ValueError("point_cloud_range must contain three minima and three maxima")
        if max_points_per_voxel <= 0 or max_voxels <= 0:
            raise ValueError("voxel limits must be positive")

        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float64)
        self.minimum = torch.tensor(point_cloud_range[:3], dtype=torch.float64)
        self.maximum = torch.tensor(point_cloud_range[3:], dtype=torch.float64)
        if torch.any(self.voxel_size <= 0) or torch.any(self.maximum <= self.minimum):
            raise ValueError("voxel sizes and point-cloud extents must be positive")

        self.grid_size = torch.round(
            (self.maximum - self.minimum) / self.voxel_size
        ).to(torch.int64)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels

    def __call__(self, points: Tensor) -> VoxelizationResult:
        """Voxelize ``[N, F]`` CPU points and emit coordinates in ``z, y, x`` order."""

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points must have shape [N, F] with F >= 3")
        if points.device.type != "cpu":
            raise ValueError("the reference ordered voxelizer requires CPU points")

        voxels = points.new_zeros(
            (self.max_voxels, self.max_points_per_voxel, points.shape[1])
        )
        coordinates = torch.zeros((self.max_voxels, 3), dtype=torch.int32)
        num_points = torch.zeros((self.max_voxels,), dtype=torch.int32)
        coordinate_to_index: Dict[Tuple[int, int, int], int] = {}
        voxel_count = 0

        minimum = self.minimum.to(points.dtype)
        voxel_size = self.voxel_size.to(points.dtype)
        for point in points:
            coordinate_xyz = torch.floor((point[:3] - minimum) / voxel_size).to(torch.int64)
            if torch.any(coordinate_xyz < 0) or torch.any(coordinate_xyz >= self.grid_size):
                continue

            coordinate_zyx = (
                int(coordinate_xyz[2]),
                int(coordinate_xyz[1]),
                int(coordinate_xyz[0]),
            )
            voxel_index = coordinate_to_index.get(coordinate_zyx)
            if voxel_index is None:
                if voxel_count >= self.max_voxels:
                    continue
                voxel_index = voxel_count
                voxel_count += 1
                coordinate_to_index[coordinate_zyx] = voxel_index
                coordinates[voxel_index] = torch.tensor(coordinate_zyx, dtype=torch.int32)

            point_index = int(num_points[voxel_index])
            if point_index < self.max_points_per_voxel:
                voxels[voxel_index, point_index] = point
                num_points[voxel_index] += 1

        return VoxelizationResult(
            voxels=voxels[:voxel_count],
            coordinates=coordinates[:voxel_count],
            num_points=num_points[:voxel_count],
        )
