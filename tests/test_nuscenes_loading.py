import numpy as np
import pytest

from centerpoint.data.nuscenes import (
    PointCloudRecord,
    SweepRecord,
    load_point_cloud,
    load_sweep,
    read_lidar_file,
)


class ReverseRNG:
    def choice(self, size, count, replace):
        assert replace is False
        return np.arange(size - 1, size - count - 1, -1)


def write_points(path, points):
    np.asarray(points, dtype=np.float32).tofile(path)


def test_lidar_reader_drops_the_fifth_raw_feature(tmp_path):
    path = tmp_path / "points.bin"
    write_points(path, [[1, 2, 3, 4, 99], [5, 6, 7, 8, 98]])

    points = read_lidar_file(path)

    np.testing.assert_array_equal(
        points,
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
    )


def test_lidar_reader_rejects_partial_records(tmp_path):
    path = tmp_path / "bad.bin"
    np.arange(6, dtype=np.float32).tofile(path)

    with pytest.raises(ValueError, match="five"):
        read_lidar_file(path)


def test_historical_sweep_filters_before_reference_transform(tmp_path):
    path = tmp_path / "sweep.bin"
    write_points(
        path,
        [
            [0.5, 0.5, 1, 10, 0],
            [1.0, 0.5, 2, 20, 0],
            [0.5, -1.0, 3, 30, 0],
        ],
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [10, 20, 30]

    points, times = load_sweep(SweepRecord(path, transform, 0.25))

    np.testing.assert_allclose(
        points,
        np.array([[11, 20.5, 32, 20], [10.5, 19, 33, 30]], dtype=np.float32),
    )
    np.testing.assert_array_equal(times, np.full((2, 1), 0.25, dtype=np.float32))


def test_point_cloud_keeps_current_points_and_uses_selected_sweep_order(tmp_path):
    current_path = tmp_path / "current.bin"
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    write_points(current_path, [[0, 0, 0, 1, 0]])
    write_points(first_path, [[2, 0, 0, 2, 0]])
    write_points(second_path, [[3, 0, 0, 3, 0]])
    record = PointCloudRecord(
        current_path,
        (
            SweepRecord(first_path, None, 0.1),
            SweepRecord(second_path, None, 0.2),
        ),
    )

    points = load_point_cloud(record, num_sweeps=3, rng=ReverseRNG())

    np.testing.assert_allclose(
        points,
        np.array(
            [[0, 0, 0, 1, 0], [3, 0, 0, 3, 0.2], [2, 0, 0, 2, 0.1]],
            dtype=np.float32,
        ),
    )


def test_canonical_loader_requires_nine_historical_records(tmp_path):
    path = tmp_path / "current.bin"
    write_points(path, [])
    record = PointCloudRecord(path, ())

    with pytest.raises(ValueError, match="enough"):
        load_point_cloud(record)
