"""Contract for the integrated preprocessing path that composes the tested stages."""

from dataclasses import replace
from itertools import cycle

import numpy as np
import pytest
import torch

from centerpoint.config import NUSCENES_VOXELNET_075
from centerpoint.data import (
    CenterPointDataset,
    HardVoxelizer,
    augment_global,
    collate_samples,
    filter_boxes_by_bev_range,
    select_classes,
)
from centerpoint.data.nuscenes import PointCloudRecord, load_point_cloud


class FixedRNG:
    """Deterministic stand-in consuming the exact call sequence of augment_global."""

    def __init__(self):
        # Cycled so repeated __getitem__ calls stay deterministic per sample.
        self.flips = cycle((True, False))
        self.uniforms = cycle((0.0, 1.0))
        self.translations = cycle((0.0, 0.0, 0.0))

    def choice(self, values, **kwargs):
        return next(self.flips)

    def uniform(self, lower, upper):
        return next(self.uniforms)

    def normal(self, mean, standard_deviation, size):
        return np.array([next(self.translations)])

    def shuffle(self, values):
        values[:] = values[::-1]


class RepeatingRNG:
    """Class-balanced sampling stand-in that keeps the leading infos of each group."""

    def __init__(self):
        self.sizes = []

    def choice(self, values, size):
        self.sizes.append(size)
        return np.asarray(list(values)[:size], dtype=object)


def tiny_config(maximum_y=4.0):
    base = NUSCENES_VOXELNET_075
    return replace(
        base,
        num_sweeps=1,
        voxel=replace(
            base.voxel,
            point_cloud_range=(-4.0, -4.0, -1.0, 4.0, maximum_y, 1.0),
            size=(1.0, 1.0, 0.5),
            max_voxels=(16, 8),
        ),
        target=replace(base.target, output_stride=2, max_objects=4),
    )


def class_names_of(config):
    return tuple(name for task in config.tasks for name in task)


def make_info(tmp_path, token, points, boxes, names):
    path = tmp_path / f"{token}.bin"
    np.asarray(points, dtype=np.float32).tofile(path)
    return {
        "token": token,
        "lidar_path": path,
        "sweeps": (),
        "gt_boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 9),
        "gt_names": np.asarray(names, dtype=object),
    }


def simple_info(tmp_path, token="a"):
    return make_info(
        tmp_path,
        token,
        [[0.5, 0.5, 0.0, 7.0, 0.0], [1.5, -1.5, 0.25, 8.0, 0.0], [-2.5, 2.5, -0.5, 9.0, 0.0]],
        [[1.0, 1.0, 0.0, 1.0, 2.0, 1.5, 0.5, -0.5, 0.25]],
        ["car"],
    )


def test_dataset_matches_a_manual_composition_of_the_pinned_stages(tmp_path):
    config = tiny_config()
    info = simple_info(tmp_path)

    sample = CenterPointDataset(
        config, [info], training=True, rng=FixedRNG(), class_balanced_sampling=False
    )[0]

    expected_points = load_point_cloud(
        PointCloudRecord(lidar_path=info["lidar_path"], sweeps=()), num_sweeps=1
    )
    expected_boxes, _, expected_ids = select_classes(
        info["gt_boxes"], info["gt_names"], class_names_of(config)
    )
    expected_points, expected_boxes = augment_global(
        expected_points,
        expected_boxes,
        rotation_range=config.augmentation.rotation_range,
        scale_range=config.augmentation.scale_range,
        translation_std=config.augmentation.translation_std,
        shuffle_points=config.augmentation.shuffle_points,
        rng=FixedRNG(),
    )
    keep = filter_boxes_by_bev_range(expected_boxes, (-4.0, -4.0, 4.0, 4.0))
    expected_voxels = HardVoxelizer((1.0, 1.0, 0.5), (-4.0, -4.0, -1.0, 4.0, 4.0, 1.0), 10, 16)(
        torch.from_numpy(np.ascontiguousarray(expected_points))
    )

    np.testing.assert_allclose(sample.points.numpy(), expected_points)
    np.testing.assert_allclose(sample.voxelization.voxels.numpy(), expected_voxels.voxels.numpy())
    assert sample.voxelization.coordinates.tolist() == expected_voxels.coordinates.tolist()
    assert sample.metadata["token"] == "a"
    assert sample.metadata["num_boxes"] == int(keep.sum())
    assert len(sample.targets) == len(config.tasks)


def test_dataset_filters_boxes_after_augmentation(tmp_path):
    # y=3 sits outside the asymmetric range, and only the flip brings it back inside.
    config = tiny_config(maximum_y=2.0)
    info = make_info(
        tmp_path,
        "flip",
        [[0.0, 0.0, 0.0, 1.0, 0.0]],
        [[0.0, 3.0, 0.0, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0]],
        ["car"],
    )

    sample = CenterPointDataset(
        config, [info], training=True, rng=FixedRNG(), class_balanced_sampling=False
    )[0]

    assert sample.metadata["num_boxes"] == 1
    assert int(sample.targets[0].mask.sum()) == 1


def test_dataset_evaluation_mode_skips_augmentation_and_uses_the_val_voxel_limit(tmp_path):
    config = tiny_config()
    info = simple_info(tmp_path)

    dataset = CenterPointDataset(config, [info], training=False, rng=FixedRNG())
    sample = dataset[0]

    unaugmented = load_point_cloud(
        PointCloudRecord(lidar_path=info["lidar_path"], sweeps=()), num_sweeps=1
    )
    np.testing.assert_allclose(sample.points.numpy(), unaugmented)
    assert dataset.voxelizer.max_voxels == config.voxel.max_voxels[1]
    assert sample.metadata["training"] is False


def test_dataset_prepares_samples_without_annotations(tmp_path):
    config = tiny_config()
    info = make_info(tmp_path, "empty", [[0.5, 0.5, 0.0, 1.0, 0.0]], [], [])

    sample = CenterPointDataset(
        config, [info], training=True, rng=FixedRNG(), class_balanced_sampling=False
    )[0]

    assert sample.metadata["num_boxes"] == 0
    assert sample.voxelization.voxels.shape[0] == 1
    assert all(int(target.mask.sum()) == 0 for target in sample.targets)


def test_dataset_drops_boxes_outside_the_configured_classes(tmp_path):
    config = tiny_config()
    info = make_info(
        tmp_path,
        "mixed",
        [[0.5, 0.5, 0.0, 1.0, 0.0]],
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0],
        ],
        ["car", "animal"],
    )

    sample = CenterPointDataset(
        config, [info], training=True, rng=FixedRNG(), class_balanced_sampling=False
    )[0]

    assert sample.metadata["num_boxes"] == 1


def test_dataset_samples_collate_for_batch_sizes_one_and_many(tmp_path):
    config = tiny_config()
    infos = [simple_info(tmp_path, "a"), simple_info(tmp_path, "b")]
    dataset = CenterPointDataset(
        config, infos, training=True, rng=FixedRNG(), class_balanced_sampling=False
    )

    single = collate_samples([dataset[0]])
    assert single.voxels.batch_size == 1
    assert single.voxels.coordinates[:, 0].tolist() == [0] * single.voxels.coordinates.shape[0]

    batch = collate_samples([dataset[0], dataset[1]])
    assert batch.voxels.batch_size == 2
    assert len(batch.targets) == len(config.tasks)
    assert sorted(set(batch.voxels.coordinates[:, 0].tolist())) == [0, 1]
    for task, targets in zip(config.tasks, batch.targets):
        assert targets.heatmap.shape == (2, len(task), 4, 4)
        assert targets.annotation.shape == (2, config.target.max_objects, 10)
    assert [metadata["token"] for metadata in batch.metadata] == ["a", "b"]


def test_dataset_resamples_training_infos_by_class(tmp_path):
    config = tiny_config()
    names = class_names_of(config)
    boxes = np.tile(
        np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0]]), (len(names), 1)
    )
    infos = [
        make_info(tmp_path, "one", [[0.5, 0.5, 0.0, 1.0, 0.0]], boxes, names),
        make_info(tmp_path, "two", [[0.5, 0.5, 0.0, 1.0, 0.0]], boxes, names),
    ]
    rng = RepeatingRNG()

    dataset = CenterPointDataset(config, infos, training=True, rng=rng)

    assert len(dataset) == 2 * len(names)
    assert rng.sizes == [2] * len(names)


def test_dataset_rejects_incomplete_info_records(tmp_path):
    config = tiny_config()
    info = dict(simple_info(tmp_path))
    del info["gt_names"]

    with pytest.raises(ValueError, match="gt_names"):
        CenterPointDataset(config, [info], training=True, class_balanced_sampling=False)
