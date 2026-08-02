import pytest
import torch

from centerpoint.data import (
    CenterTarget,
    PreparedSample,
    VoxelizationResult,
    collate_samples,
)


def make_target(value, classes=1):
    return CenterTarget(
        heatmap=torch.full((classes, 2, 3), float(value)),
        annotation=torch.full((2, 10), float(value)),
        indices=torch.tensor([value, 0], dtype=torch.int64),
        mask=torch.tensor([1, 0], dtype=torch.uint8),
        categories=torch.zeros((2,), dtype=torch.int64),
    )


def make_sample(value, voxel_count=1):
    voxels = torch.full((voxel_count, 2, 5), float(value))
    coordinates = torch.tensor([[value, value + 1, value + 2]], dtype=torch.int32)
    if voxel_count == 0:
        coordinates = torch.empty((0, 3), dtype=torch.int32)
    return PreparedSample(
        points=torch.full((value + 1, 5), float(value)),
        voxelization=VoxelizationResult(
            voxels=voxels,
            coordinates=coordinates,
            num_points=torch.ones((voxel_count,), dtype=torch.int32),
        ),
        targets=(make_target(value), make_target(value + 10, classes=2)),
        metadata={"token": str(value)},
    )


def test_collate_preserves_sparse_and_sample_order():
    batch = collate_samples([make_sample(1), make_sample(4)])

    assert batch.voxels.voxels[:, 0, 0].tolist() == [1.0, 4.0]
    assert batch.voxels.coordinates.tolist() == [[0, 1, 2, 3], [1, 4, 5, 6]]
    assert batch.voxels.batch_size == 2
    assert [points.shape[0] for points in batch.points] == [2, 5]
    assert [metadata["token"] for metadata in batch.metadata] == ["1", "4"]


def test_collate_stacks_each_task_without_compacting_slots():
    batch = collate_samples([make_sample(1), make_sample(4)])

    assert len(batch.targets) == 2
    assert batch.targets[0].heatmap.shape == (2, 1, 2, 3)
    assert batch.targets[1].heatmap.shape == (2, 2, 2, 3)
    assert batch.targets[0].indices.tolist() == [[1, 0], [4, 0]]
    assert batch.targets[0].mask.tolist() == [[1, 0], [1, 0]]


def test_collate_supports_empty_voxel_samples():
    batch = collate_samples([make_sample(0, voxel_count=0), make_sample(2)])

    assert batch.voxels.voxels.shape == (1, 2, 5)
    assert batch.voxels.coordinates.tolist() == [[1, 2, 3, 4]]


def test_collate_rejects_empty_or_inconsistent_batches():
    with pytest.raises(ValueError, match="empty"):
        collate_samples([])

    sample = make_sample(1)
    inconsistent = PreparedSample(
        points=sample.points,
        voxelization=sample.voxelization,
        targets=sample.targets[:1],
        metadata=sample.metadata,
    )
    with pytest.raises(ValueError, match="task count"):
        collate_samples([sample, inconsistent])
