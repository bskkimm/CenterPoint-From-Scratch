from dataclasses import FrozenInstanceError

import pytest

from centerpoint.config import NUSCENES_VOXELNET_075


def test_canonical_config_derives_official_grid_and_feature_map():
    config = NUSCENES_VOXELNET_075

    assert config.voxel.grid_size == (1440, 1440, 40)
    assert config.voxel.grid_size[0] // config.target.output_stride == 180
    assert config.voxel.grid_size[1] // config.target.output_stride == 180


def test_canonical_factories_propagate_frozen_values():
    config = NUSCENES_VOXELNET_075

    train_voxelizer = config.make_voxelizer(training=True)
    eval_voxelizer = config.make_voxelizer(training=False)
    assigner = config.make_target_assigner()
    decoder = config.make_decoder()

    assert train_voxelizer.max_voxels == 120_000
    assert eval_voxelizer.max_voxels == 160_000
    assert assigner.tasks == config.tasks
    assert assigner.feature_width == assigner.feature_height == 180
    assert decoder.output_stride == 8
    assert decoder.score_threshold == 0.1
    assert decoder.post_center_range == config.inference.post_center_range
    assert decoder.voxel_size[:2] == config.inference.voxel_size_xy
    assert decoder.point_cloud_range[:2] == config.inference.point_cloud_min_xy


def test_canonical_config_is_immutable():
    with pytest.raises(FrozenInstanceError):
        NUSCENES_VOXELNET_075.num_sweeps = 1
