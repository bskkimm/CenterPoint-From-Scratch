import pytest
import torch

from centerpoint.data import HardVoxelizer
from centerpoint.models.readers import MeanVoxelFeatureEncoder


def test_hard_voxelizer_preserves_first_occurrence_and_zyx_coordinates():
    voxelizer = HardVoxelizer(
        voxel_size=(1, 1, 1),
        point_cloud_range=(0, 0, 0, 2, 2, 2),
        max_points_per_voxel=2,
        max_voxels=3,
    )
    points = torch.tensor(
        [
            [1.1, 0.1, 1.1, 10.0],
            [0.1, 1.1, 0.1, 20.0],
            [1.2, 0.2, 1.2, 30.0],
        ]
    )

    result = voxelizer(points)

    torch.testing.assert_allclose(
        result.coordinates,
        torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.int32),
    )
    torch.testing.assert_allclose(result.num_points, torch.tensor([2, 1], dtype=torch.int32))
    torch.testing.assert_allclose(result.voxels[0, :2], points[[0, 2]])
    torch.testing.assert_allclose(result.voxels[1, 0], points[1])


def test_hard_voxelizer_excludes_range_boundaries_and_truncates_points():
    voxelizer = HardVoxelizer((1, 1, 1), (0, 0, 0, 2, 2, 2), 2, 2)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.1, 0.1, 2.0],
            [0.2, 0.2, 0.2, 3.0],
            [2.0, 1.0, 1.0, 4.0],
            [-0.1, 0.0, 0.0, 5.0],
        ]
    )

    result = voxelizer(points)

    assert result.voxels.shape == (1, 2, 4)
    torch.testing.assert_allclose(result.voxels[0], points[:2])
    assert result.num_points.tolist() == [2]


def test_existing_voxel_can_receive_points_after_max_voxel_limit():
    voxelizer = HardVoxelizer((1, 1, 1), (0, 0, 0, 3, 1, 1), 3, 1)
    points = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.2, 0.1, 0.1],
        ]
    )

    result = voxelizer(points)

    assert result.num_points.tolist() == [2]
    torch.testing.assert_allclose(result.voxels[0, :2], points[[0, 2]])


def test_mean_voxel_feature_encoder_averages_all_channels():
    features = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0, 0.0], [3.0, 4.0, 5.0, 6.0, 0.2], [0, 0, 0, 0, 0]],
            [[2.0, 4.0, 6.0, 8.0, -0.1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ]
    )

    encoded = MeanVoxelFeatureEncoder(5)(features, torch.tensor([2, 1]))

    torch.testing.assert_allclose(
        encoded,
        torch.tensor([[2.0, 3.0, 4.0, 5.0, 0.1], [2.0, 4.0, 6.0, 8.0, -0.1]]),
    )
    assert encoded.is_contiguous()


def test_voxelizer_and_encoder_validate_inputs():
    with pytest.raises(ValueError, match="points"):
        HardVoxelizer((1, 1, 1), (0, 0, 0, 1, 1, 1), 1, 1)(
            torch.empty((1, 2))
        )
    with pytest.raises(ValueError, match="at least one"):
        MeanVoxelFeatureEncoder(3)(torch.zeros((1, 2, 3)), torch.tensor([0]))
