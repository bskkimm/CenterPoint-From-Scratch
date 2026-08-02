"""Batch local voxelization and CenterHead target contracts."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from centerpoint.contracts import TaskTargets, VoxelBatch
from centerpoint.data.targets import CenterTarget
from centerpoint.data.voxelization import VoxelizationResult


@dataclass(frozen=True)
class PreparedSample:
    """One fully prepared sample before sparse batch collation."""

    points: Tensor
    voxelization: VoxelizationResult
    targets: Tuple[CenterTarget, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedBatch:
    """Sparse inputs, dense task targets, and sample-level context."""

    voxels: VoxelBatch
    targets: Tuple[TaskTargets, ...]
    points: Tuple[Tensor, ...]
    metadata: Tuple[Mapping[str, Any], ...]


def collate_samples(samples: Sequence[PreparedSample]) -> PreparedBatch:
    """Collate samples without changing point, voxel, or target-slot order."""

    if not samples:
        raise ValueError("cannot collate an empty sample sequence")
    task_count = len(samples[0].targets)
    if task_count == 0 or any(len(sample.targets) != task_count for sample in samples):
        raise ValueError("all samples must contain the same non-zero task count")

    voxel_tensors = []
    point_counts = []
    coordinates = []
    for batch_index, sample in enumerate(samples):
        result = sample.voxelization
        voxel_tensors.append(result.voxels)
        point_counts.append(result.num_points)
        prefix = torch.full(
            (result.coordinates.shape[0], 1),
            batch_index,
            dtype=result.coordinates.dtype,
            device=result.coordinates.device,
        )
        coordinates.append(torch.cat((prefix, result.coordinates), dim=1))

    voxel_batch = VoxelBatch(
        voxels=torch.cat(voxel_tensors, dim=0),
        num_points=torch.cat(point_counts, dim=0),
        coordinates=torch.cat(coordinates, dim=0),
        batch_size=len(samples),
    )
    task_targets = tuple(
        TaskTargets(
            heatmap=torch.stack([sample.targets[index].heatmap for sample in samples]),
            annotation=torch.stack([sample.targets[index].annotation for sample in samples]),
            indices=torch.stack([sample.targets[index].indices for sample in samples]),
            mask=torch.stack([sample.targets[index].mask for sample in samples]),
            categories=torch.stack([sample.targets[index].categories for sample in samples]),
        )
        for index in range(task_count)
    )
    return PreparedBatch(
        voxels=voxel_batch,
        targets=task_targets,
        points=tuple(sample.points for sample in samples),
        metadata=tuple(sample.metadata for sample in samples),
    )
