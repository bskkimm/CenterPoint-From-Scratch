"""Dense CenterHead prediction decoding before non-maximum suppression."""

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class DetectionCandidates:
    """Variable-length pre-NMS candidates for one sample and one task."""

    boxes: Tensor
    scores: Tensor
    labels: Tensor


class CenterPointDecoder:
    """Decode CenterHead maps using the official repository behavior."""

    def __init__(
        self,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        output_stride: int,
        score_threshold: Optional[float] = None,
        post_center_range: Optional[Sequence[float]] = None,
    ) -> None:
        if len(voxel_size) != 3 or len(point_cloud_range) != 6:
            raise ValueError("voxel_size and point_cloud_range have invalid lengths")
        if output_stride <= 0:
            raise ValueError("output_stride must be positive")
        if post_center_range is not None and len(post_center_range) != 6:
            raise ValueError("post_center_range must contain six values")

        self.voxel_size = tuple(float(value) for value in voxel_size)
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        self.output_stride = output_stride
        self.score_threshold = score_threshold
        self.post_center_range = (
            tuple(float(value) for value in post_center_range)
            if post_center_range is not None
            else None
        )

    def __call__(
        self,
        predictions: Mapping[str, Tensor],
        label_offset: int = 0,
    ) -> List[DetectionCandidates]:
        """Decode one task's NCHW prediction maps for every batch sample."""

        required_channels = {"hm": None, "reg": 2, "height": 1, "dim": 3, "rot": 2}
        if any(name not in predictions for name in required_channels):
            raise ValueError("predictions must contain hm, reg, height, dim, and rot")

        heatmap = predictions["hm"]
        if heatmap.ndim != 4:
            raise ValueError("prediction maps must have shape [B, C, H, W]")
        batch, _, height, width = heatmap.shape
        for name, channels in required_channels.items():
            value = predictions[name]
            if value.ndim != 4 or value.shape[0] != batch or value.shape[2:] != (height, width):
                raise ValueError(f"{name} has incompatible spatial or batch dimensions")
            if channels is not None and value.shape[1] != channels:
                raise ValueError(f"{name} must contain {channels} channels")

        velocity = predictions.get("vel")
        if velocity is not None and velocity.shape != (batch, 2, height, width):
            raise ValueError("vel must have shape [B, 2, H, W]")

        scores = heatmap.sigmoid().permute(0, 2, 3, 1).contiguous()
        offsets = predictions["reg"].permute(0, 2, 3, 1).contiguous()
        center_z = predictions["height"].permute(0, 2, 3, 1).contiguous()
        dimensions = predictions["dim"].exp().permute(0, 2, 3, 1).contiguous()
        rotation = predictions["rot"].permute(0, 2, 3, 1).contiguous()
        yaw = torch.atan2(rotation[..., 0:1], rotation[..., 1:2])

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=offsets.dtype, device=offsets.device),
            torch.arange(width, dtype=offsets.dtype, device=offsets.device),
        )
        grid_x = grid_x.reshape(1, height, width, 1)
        grid_y = grid_y.reshape(1, height, width, 1)
        center_x = (
            (grid_x + offsets[..., 0:1])
            * self.output_stride
            * self.voxel_size[0]
            + self.point_cloud_range[0]
        )
        center_y = (
            (grid_y + offsets[..., 1:2])
            * self.output_stride
            * self.voxel_size[1]
            + self.point_cloud_range[1]
        )

        box_parts = [center_x, center_y, center_z, dimensions]
        if velocity is not None:
            box_parts.append(velocity.permute(0, 2, 3, 1).contiguous())
        box_parts.append(yaw)
        boxes = torch.cat(box_parts, dim=-1).view(batch, height * width, -1)
        scores = scores.view(batch, height * width, -1)
        best_scores, labels = torch.max(scores, dim=-1)
        labels = labels + label_offset

        results: List[DetectionCandidates] = []
        for batch_index in range(batch):
            keep = torch.ones_like(best_scores[batch_index], dtype=torch.bool)
            if self.score_threshold is not None:
                keep &= best_scores[batch_index] > self.score_threshold
            if self.post_center_range is not None:
                limits = boxes.new_tensor(self.post_center_range)
                centers = boxes[batch_index, :, :3]
                keep &= torch.all(centers >= limits[:3], dim=1)
                keep &= torch.all(centers <= limits[3:], dim=1)

            results.append(
                DetectionCandidates(
                    boxes=boxes[batch_index, keep],
                    scores=best_scores[batch_index, keep],
                    labels=labels[batch_index, keep],
                )
            )
        return results
