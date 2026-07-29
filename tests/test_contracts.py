import pytest
import torch

from centerpoint.contracts import TaskPredictions, TaskTargets, VoxelBatch
from centerpoint.models.heads import CenterPointDecoder


def make_task_predictions():
    return TaskPredictions(
        heatmap=torch.zeros((1, 2, 2, 3)),
        center_offset=torch.zeros((1, 2, 2, 3)),
        height=torch.zeros((1, 1, 2, 3)),
        dimensions=torch.zeros((1, 3, 2, 3)),
        rotation=torch.zeros((1, 2, 2, 3)),
        velocity=torch.zeros((1, 2, 2, 3)),
    )


def test_voxel_batch_accepts_batched_zyx_coordinates():
    batch = VoxelBatch(
        voxels=torch.zeros((2, 3, 5)),
        num_points=torch.tensor([1, 3], dtype=torch.int32),
        coordinates=torch.tensor([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=torch.int32),
        batch_size=2,
    )

    assert batch.coordinates[:, 0].tolist() == [0, 1]


def test_voxel_batch_rejects_invalid_counts_and_batch_indices():
    with pytest.raises(ValueError, match="num_points"):
        VoxelBatch(
            torch.zeros((1, 2, 5)),
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
            1,
        )
    with pytest.raises(ValueError, match="batch indices"):
        VoxelBatch(
            torch.zeros((1, 2, 5)),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([[1, 0, 0, 0]], dtype=torch.int32),
            1,
        )


def test_task_targets_enforce_shared_batch_and_object_shapes():
    targets = TaskTargets(
        heatmap=torch.zeros((2, 1, 4, 4)),
        annotation=torch.zeros((2, 3, 10)),
        indices=torch.zeros((2, 3), dtype=torch.int64),
        mask=torch.zeros((2, 3), dtype=torch.uint8),
        categories=torch.zeros((2, 3), dtype=torch.int64),
    )

    assert targets.annotation.shape[-1] == 10


def test_task_predictions_use_official_head_keys():
    predictions = make_task_predictions()

    assert tuple(predictions.as_dict()) == ("hm", "reg", "height", "dim", "rot", "vel")


def test_task_predictions_reject_mismatched_head_shape():
    with pytest.raises(ValueError, match="center_offset"):
        TaskPredictions(
            heatmap=torch.zeros((1, 1, 2, 3)),
            center_offset=torch.zeros((1, 2, 3, 2)),
            height=torch.zeros((1, 1, 2, 3)),
            dimensions=torch.zeros((1, 3, 2, 3)),
            rotation=torch.zeros((1, 2, 2, 3)),
        )


def test_decoder_accepts_typed_task_predictions():
    predictions = make_task_predictions()
    predictions.rotation[:, 1] = 1

    result = CenterPointDecoder((1, 1, 1), (0, 0, -5, 3, 2, 5), 1)(predictions)[0]

    assert result.boxes.shape == (6, 9)
    assert result.labels.shape == (6,)
