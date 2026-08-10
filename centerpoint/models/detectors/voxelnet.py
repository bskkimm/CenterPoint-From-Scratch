"""Boundary-complete VoxelNet assembly with an injected sparse backbone."""

from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from centerpoint.contracts import TaskTargets, VoxelBatch
from centerpoint.models.backbones import SparseBackbone, SparseBackboneInput
from centerpoint.models.heads import Detections


class VoxelNet(nn.Module):
    """Connect voxel features, sparse BEV extraction, neck, and CenterHead."""

    def __init__(
        self,
        reader: nn.Module,
        backbone: SparseBackbone,
        neck: nn.Module,
        bbox_head: nn.Module,
        postprocessor: Callable,
        spatial_shape: Sequence[int],
    ) -> None:
        super().__init__()
        spatial_shape = tuple(spatial_shape)
        if len(spatial_shape) != 3 or any(size <= 0 for size in spatial_shape):
            raise ValueError("spatial_shape must contain positive z, y, x sizes")
        if not isinstance(backbone, SparseBackbone):
            raise TypeError("backbone must implement SparseBackbone")
        if isinstance(postprocessor, nn.Module):
            raise TypeError("postprocessor must be a non-module callable")
        if not callable(postprocessor):
            raise ValueError("postprocessor must be callable")

        self.reader = reader
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head
        # Postprocessing must not contribute a fifth top-level module namespace.
        object.__setattr__(self, "_postprocessor", postprocessor)
        self._spatial_shape = spatial_shape

    def _sparse_inputs(self, voxels: VoxelBatch) -> SparseBackboneInput:
        features = self.reader(voxels.voxels, voxels.num_points)
        if (
            features.ndim != 2
            or features.shape[1] != self.backbone.input_channels
        ):
            raise ValueError("VFE feature channels must match backbone input_channels")
        return SparseBackboneInput(
            features, voxels.coordinates, self._spatial_shape, voxels.batch_size
        )

    def forward_features(
        self, voxels: VoxelBatch
    ) -> tuple[list[dict[str, Tensor]], Tensor]:
        """Return six task predictions and the shared CenterHead feature map."""

        sparse = self._sparse_inputs(voxels)
        bev = self.backbone(sparse)
        neck_features = self.neck(bev)
        predictions, _ = self.bbox_head(neck_features)
        return predictions, neck_features

    def loss(
        self, voxels: VoxelBatch, targets: Sequence[TaskTargets]
    ) -> dict[str, list[Tensor]]:
        """Compute per-task CenterHead losses for a voxel batch."""

        if not targets:
            raise ValueError("targets must match the configured task count")
        predictions, _ = self.forward_features(voxels)
        return self.bbox_head.loss(predictions, targets)

    @torch.no_grad()
    def predict(self, voxels: VoxelBatch) -> list[Detections]:
        """Decode and suppress six-task CenterHead predictions."""

        predictions, _ = self.forward_features(voxels)
        return self._postprocessor(predictions)
