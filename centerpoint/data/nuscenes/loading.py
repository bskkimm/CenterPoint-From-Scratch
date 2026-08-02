"""SDK-independent loading of nuScenes current and historical LiDAR sweeps."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SweepRecord:
    """One historical LiDAR file expressed relative to the reference LiDAR."""

    lidar_path: PathLike
    transform_matrix: Optional[np.ndarray]
    time_lag: float

    def __post_init__(self) -> None:
        if self.transform_matrix is not None and self.transform_matrix.shape != (4, 4):
            raise ValueError("transform_matrix must have shape [4, 4]")


@dataclass(frozen=True)
class PointCloudRecord:
    """Reference LiDAR path and its cached historical sweep records."""

    lidar_path: PathLike
    sweeps: Tuple[SweepRecord, ...]


def read_lidar_file(path: PathLike) -> np.ndarray:
    """Read nuScenes float32 records and retain x, y, z, and intensity."""

    values = np.fromfile(Path(path), dtype=np.float32)
    if values.size % 5 != 0:
        raise ValueError("nuScenes LiDAR files must contain five float32 values per point")
    return values.reshape(-1, 5)[:, :4].copy()


def load_sweep(sweep: SweepRecord) -> Tuple[np.ndarray, np.ndarray]:
    """Filter and transform one historical sweep into the reference frame."""

    points = read_lidar_file(sweep.lidar_path)
    keep = ~((np.abs(points[:, 0]) < 1.0) & (np.abs(points[:, 1]) < 1.0))
    points = points[keep]
    if sweep.transform_matrix is not None and points.shape[0]:
        homogeneous = np.column_stack(
            (points[:, :3], np.ones((points.shape[0],), dtype=points.dtype))
        )
        points[:, :3] = (homogeneous @ sweep.transform_matrix.T)[:, :3]
    times = np.full((points.shape[0], 1), sweep.time_lag, dtype=points.dtype)
    return points, times


def load_point_cloud(
    record: PointCloudRecord,
    *,
    num_sweeps: int = 10,
    rng: Any = np.random,
) -> np.ndarray:
    """Load current points followed by a random no-replacement sweep selection."""

    if num_sweeps <= 0:
        raise ValueError("num_sweeps must be positive")
    historical_count = num_sweeps - 1
    if len(record.sweeps) < historical_count:
        raise ValueError("point-cloud record does not contain enough historical sweeps")

    current = read_lidar_file(record.lidar_path)
    point_parts = [current]
    time_parts = [np.zeros((current.shape[0], 1), dtype=current.dtype)]
    if historical_count:
        indices: Sequence[int] = rng.choice(
            len(record.sweeps),
            historical_count,
            replace=False,
        )
        for index in indices:
            points, times = load_sweep(record.sweeps[int(index)])
            point_parts.append(points)
            time_parts.append(times)

    points = np.concatenate(point_parts, axis=0)
    times = np.concatenate(time_parts, axis=0).astype(points.dtype, copy=False)
    return np.hstack((points, times))
