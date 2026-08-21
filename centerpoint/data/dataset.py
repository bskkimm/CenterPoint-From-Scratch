"""Integrated preprocessing path composing the tested nuScenes pipeline stages."""

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from centerpoint.data.augmentation import (
    augment_global,
    filter_boxes_by_bev_range,
    select_classes,
)
from centerpoint.data.collate import PreparedSample
from centerpoint.data.nuscenes.loading import PointCloudRecord, load_point_cloud
from centerpoint.data.sampling import class_balanced_infos
from centerpoint.data.targets import CenterTarget


REQUIRED_INFO_KEYS = ("token", "lidar_path", "sweeps", "gt_boxes", "gt_names")


class CenterPointDataset:
    """Prepare samples in the pinned official stage order.

    The order is sweep loading, class selection, global augmentation, BEV range
    filtering, hard voxelization, then target assignment. Augmentation and BEV range
    filtering are training-only, matching the pinned preprocessing and voxelization
    stages; the validation voxel limit is selected from the same configuration.

    Metadata records are injected rather than built here, because the nuScenes metadata
    builder is still gated on official data. Each record must provide
    :data:`REQUIRED_INFO_KEYS`, with ``gt_boxes`` shaped ``[N, 9]`` in the internal box
    layout and ``gt_names`` shaped ``[N]``.

    The reference voxelizer iterates points in Python, so this path is sized for contract
    tests and small samples rather than full-speed training.
    """

    def __init__(
        self,
        config,
        infos: Sequence[Mapping[str, Any]],
        *,
        training: bool,
        rng: Any = np.random,
        class_balanced_sampling: Optional[bool] = None,
        assign_targets: bool = True,
    ) -> None:
        for index, info in enumerate(infos):
            missing = [key for key in REQUIRED_INFO_KEYS if key not in info]
            if missing:
                raise ValueError(f"info record {index} is missing keys: {missing}")

        self.config = config
        self.training = bool(training)
        self.rng = rng
        self.class_names = tuple(name for task in config.tasks for name in task)
        self.voxelizer = config.make_voxelizer(training=self.training)
        self.target_assigner = config.make_target_assigner() if assign_targets else None

        resample = self.training if class_balanced_sampling is None else bool(class_balanced_sampling)
        self.infos = (
            list(class_balanced_infos(infos, self.class_names, rng=rng))
            if resample
            else list(infos)
        )

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, index: int) -> PreparedSample:
        """Prepare one sample as points, hard voxels, per-task targets, and metadata."""

        info = self.infos[index]
        points = load_point_cloud(
            PointCloudRecord(
                lidar_path=info["lidar_path"],
                sweeps=tuple(info["sweeps"]),
            ),
            num_sweeps=self.config.num_sweeps,
            rng=self.rng,
        )
        boxes = np.asarray(info["gt_boxes"], dtype=np.float64).reshape(-1, 9)
        names = np.asarray(info["gt_names"], dtype=object).reshape(-1)
        boxes, _, class_ids = select_classes(boxes, names, self.class_names)

        if self.training:
            points, boxes = augment_global(
                points,
                boxes,
                rotation_range=self.config.augmentation.rotation_range,
                scale_range=self.config.augmentation.scale_range,
                translation_std=self.config.augmentation.translation_std,
                shuffle_points=self.config.augmentation.shuffle_points,
                rng=self.rng,
            )
            keep = filter_boxes_by_bev_range(boxes, self._bev_range())
            boxes, class_ids = boxes[keep], class_ids[keep]

        point_tensor = torch.from_numpy(np.ascontiguousarray(points))
        return PreparedSample(
            points=point_tensor,
            voxelization=self.voxelizer(point_tensor),
            targets=self._assign(boxes, class_ids),
            metadata={
                "token": info["token"],
                "num_points": int(points.shape[0]),
                "num_boxes": int(boxes.shape[0]),
                "training": self.training,
            },
        )

    def _assign(self, boxes: np.ndarray, class_ids: np.ndarray) -> Tuple[CenterTarget, ...]:
        if self.target_assigner is None:
            return ()
        return tuple(
            self.target_assigner(
                torch.from_numpy(np.ascontiguousarray(boxes)).to(torch.float32),
                torch.from_numpy(np.ascontiguousarray(class_ids)).to(torch.int64),
            )
        )

    def _bev_range(self) -> Tuple[float, float, float, float]:
        extents = self.config.voxel.point_cloud_range
        return (extents[0], extents[1], extents[3], extents[4])
