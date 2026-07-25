"""CenterPoint heatmap and box target assignment."""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

from centerpoint.utils.geometry import limit_period
from centerpoint.utils.heatmap import draw_gaussian, gaussian_radius


NUSCENES_TASKS: Tuple[Tuple[str, ...], ...] = (
    ("car",),
    ("truck", "construction_vehicle"),
    ("bus", "trailer"),
    ("barrier",),
    ("motorcycle", "bicycle"),
    ("pedestrian", "traffic_cone"),
)


@dataclass(frozen=True)
class CenterTarget:
    """Targets for one CenterHead task and one sample."""

    heatmap: Tensor
    annotation: Tensor
    indices: Tensor
    mask: Tensor
    categories: Tensor


class CenterTargetAssigner:
    """Assign official CenterPoint targets from internal-format boxes."""

    def __init__(
        self,
        tasks: Sequence[Sequence[str]],
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        output_stride: int,
        gaussian_overlap: float = 0.1,
        max_objects: int = 500,
        min_radius: int = 2,
    ) -> None:
        if not tasks or any(not task for task in tasks):
            raise ValueError("tasks must contain at least one class each")
        if len(voxel_size) != 3 or len(point_cloud_range) != 6:
            raise ValueError("voxel_size and point_cloud_range have invalid lengths")
        if output_stride <= 0 or max_objects <= 0 or min_radius < 0:
            raise ValueError("stride/object limits must be positive and radius non-negative")

        self.tasks = tuple(tuple(task) for task in tasks)
        self.voxel_size = tuple(float(value) for value in voxel_size)
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        self.output_stride = output_stride
        self.gaussian_overlap = gaussian_overlap
        self.max_objects = max_objects
        self.min_radius = min_radius

        grid_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]
        )
        grid_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]
        )
        self.feature_width = grid_x // output_stride
        self.feature_height = grid_y // output_stride

    def __call__(self, boxes: Tensor, class_ids: Tensor) -> List[CenterTarget]:
        """Build targets from ``[x,y,z,w,l,h,vx,vy,yaw]`` and 1-based class IDs."""

        if boxes.ndim != 2 or boxes.shape[1] != 9:
            raise ValueError("boxes must have shape [N, 9]")
        if class_ids.ndim != 1 or class_ids.shape[0] != boxes.shape[0]:
            raise ValueError("class_ids must have shape [N]")
        if boxes.device != class_ids.device:
            raise ValueError("boxes and class_ids must be on the same device")

        targets: List[CenterTarget] = []
        global_class_offset = 0
        for task in self.tasks:
            task_boxes = []
            task_categories = []
            for local_class, _ in enumerate(task):
                global_class = global_class_offset + local_class + 1
                selected = boxes[class_ids == global_class]
                task_boxes.append(selected)
                task_categories.append(
                    torch.full(
                        (selected.shape[0],),
                        local_class,
                        dtype=torch.int64,
                        device=boxes.device,
                    )
                )

            grouped_boxes = torch.cat(task_boxes, dim=0)
            grouped_categories = torch.cat(task_categories, dim=0)
            targets.append(self._assign_task(grouped_boxes, grouped_categories, len(task)))
            global_class_offset += len(task)
        return targets

    def _assign_task(
        self,
        boxes: Tensor,
        categories: Tensor,
        num_classes: int,
    ) -> CenterTarget:
        heatmap = boxes.new_zeros(
            (num_classes, self.feature_height, self.feature_width), dtype=torch.float32
        )
        annotation = boxes.new_zeros((self.max_objects, 10), dtype=torch.float32)
        indices = torch.zeros((self.max_objects,), dtype=torch.int64, device=boxes.device)
        mask = torch.zeros((self.max_objects,), dtype=torch.uint8, device=boxes.device)
        target_categories = torch.zeros(
            (self.max_objects,), dtype=torch.int64, device=boxes.device
        )

        for object_index in range(min(boxes.shape[0], self.max_objects)):
            box = boxes[object_index]
            width, length = float(box[3]), float(box[4])
            if width <= 0 or length <= 0:
                continue

            width_on_map = width / self.voxel_size[0] / self.output_stride
            length_on_map = length / self.voxel_size[1] / self.output_stride
            radius = max(
                self.min_radius,
                int(
                    gaussian_radius(
                        (length_on_map, width_on_map),
                        min_overlap=self.gaussian_overlap,
                    )
                ),
            )

            center = torch.tensor(
                [
                    (float(box[0]) - self.point_cloud_range[0])
                    / self.voxel_size[0]
                    / self.output_stride,
                    (float(box[1]) - self.point_cloud_range[1])
                    / self.voxel_size[1]
                    / self.output_stride,
                ],
                dtype=torch.float32,
                device=boxes.device,
            )
            center_int = center.to(torch.int32)
            center_x, center_y = int(center_int[0]), int(center_int[1])
            if not (
                0 <= center_x < self.feature_width
                and 0 <= center_y < self.feature_height
            ):
                continue

            category = int(categories[object_index])
            draw_gaussian(heatmap[category], center, radius)
            yaw = limit_period(box[8].reshape(1))[0]

            indices[object_index] = center_y * self.feature_width + center_x
            mask[object_index] = 1
            target_categories[object_index] = category
            annotation[object_index] = torch.stack(
                (
                    center[0] - center_int[0],
                    center[1] - center_int[1],
                    box[2],
                    torch.log(box[3]),
                    torch.log(box[4]),
                    torch.log(box[5]),
                    box[6],
                    box[7],
                    torch.sin(yaw),
                    torch.cos(yaw),
                )
            ).to(torch.float32)

        return CenterTarget(heatmap, annotation, indices, mask, target_categories)
