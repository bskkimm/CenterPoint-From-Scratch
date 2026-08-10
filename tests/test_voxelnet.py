import pytest
import torch
from torch import nn

from centerpoint import NUSCENES_VOXELNET_075
from centerpoint.contracts import TaskTargets, VoxelBatch
from centerpoint.models import (
    CenterHead,
    CenterPointPostprocessor,
    MeanVoxelFeatureEncoder,
    RPN,
    SparseBackbone,
    VoxelNet,
)
from centerpoint.ops import rotated_nms


class TestSparseBackbone(SparseBackbone):
    def __init__(self, input_channels=5):
        super().__init__(input_channels=input_channels, output_channels=256)

    def forward_sparse(self, inputs):
        height = inputs.spatial_shape[1] // self.output_stride
        width = inputs.spatial_shape[2] // self.output_stride
        bev = inputs.features.new_zeros(
            (inputs.batch_size, self.output_channels, height, width)
        )
        if inputs.features.shape[0]:
            values = inputs.features.sum(dim=1)
            batch, _, y, x = inputs.coordinates.long().unbind(dim=1)
            bev[batch, :, y // self.output_stride, x // self.output_stride] = values.unsqueeze(1)
        return bev


class IncompatibleFeatureBackbone(TestSparseBackbone):
    def __init__(self):
        super().__init__(input_channels=4)


class ModulePostprocessor(nn.Module):
    def forward(self, predictions):
        return predictions


def make_model(*, backbone=None, postprocessor=None):
    config = NUSCENES_VOXELNET_075
    return VoxelNet(
        reader=MeanVoxelFeatureEncoder(config.model.num_input_features),
        backbone=TestSparseBackbone() if backbone is None else backbone,
        neck=RPN(
            layer_nums=config.model.neck.layer_numbers,
            ds_layer_strides=config.model.neck.downsample_strides,
            ds_num_filters=config.model.neck.downsample_filters,
            us_layer_strides=config.model.neck.upsample_strides,
            us_num_filters=config.model.neck.upsample_filters,
            num_input_features=config.model.neck.input_channels,
        ),
        bbox_head=CenterHead(
            in_channels=config.model.head.input_channels,
            tasks=config.tasks,
            common_heads={
                branch.name: (branch.output_channels, branch.num_convolutions)
                for branch in config.model.head.branches
            },
            share_conv_channel=config.model.head.shared_channels,
            loss_weight=config.model.head.loss_weight,
            code_weights=config.model.head.code_weights,
        ),
        postprocessor=(
            CenterPointPostprocessor(
                config.make_decoder(),
                rotated_nms,
                config.tasks,
                iou_threshold=config.inference.nms.iou_threshold,
                pre_max_size=config.inference.nms.pre_max_size,
                post_max_size=config.inference.nms.post_max_size,
            )
            if postprocessor is None
            else postprocessor
        ),
        spatial_shape=(4, 32, 32),
    )


def make_voxel_batch():
    return VoxelBatch(
        voxels=torch.arange(200, dtype=torch.float32).reshape(4, 10, 5),
        num_points=torch.tensor([10, 8, 6, 4], dtype=torch.int32),
        coordinates=torch.tensor(
            [[0, 0, 0, 0], [0, 1, 8, 8], [1, 2, 16, 16], [1, 3, 24, 24]],
            dtype=torch.int32,
        ),
        batch_size=2,
    )


def test_voxelnet_registers_frozen_top_level_modules_and_traces_features():
    model = make_model()
    predictions, bev = model.forward_features(make_voxel_batch())

    assert tuple(model._modules) == ("reader", "backbone", "neck", "bbox_head")
    assert bev.shape == (2, 512, 4, 4)
    assert [task["hm"].shape[1] for task in predictions] == [1, 2, 2, 1, 2, 2]


def make_task_targets(batch_size=2):
    targets = []
    for task in NUSCENES_VOXELNET_075.tasks:
        heatmap = torch.zeros((batch_size, len(task), 4, 4))
        heatmap[:, 0, 0, 0] = 1
        targets.append(
            TaskTargets(
                heatmap=heatmap,
                annotation=torch.zeros((batch_size, 1, 10)),
                indices=torch.zeros((batch_size, 1), dtype=torch.int64),
                mask=torch.ones((batch_size, 1), dtype=torch.uint8),
                categories=torch.zeros((batch_size, 1), dtype=torch.int64),
            )
        )
    return targets


def test_voxelnet_training_loss_backpropagates_through_trainable_stages():
    model = make_model()
    losses = model.loss(make_voxel_batch(), make_task_targets(batch_size=2))
    total = sum(losses["loss"])
    total.backward()

    assert model.neck.blocks[0][1].weight.grad is not None
    assert model.bbox_head.shared_conv[0].weight.grad is not None


def test_voxelnet_predict_merges_six_tasks():
    detections = make_model().eval().predict(make_voxel_batch())

    assert len(detections) == 2
    assert detections[0].boxes.shape[1] == 9


def test_voxelnet_rejects_incompatible_targets_and_feature_channels():
    with pytest.raises(ValueError, match="task count"):
        make_model().loss(make_voxel_batch(), make_task_targets()[:5])

    incompatible_features = VoxelBatch(
        voxels=torch.zeros((1, 10, 4)),
        num_points=torch.tensor([1], dtype=torch.int32),
        coordinates=torch.tensor([[0, 0, 0, 0]], dtype=torch.int32),
        batch_size=1,
    )
    with pytest.raises(ValueError, match="features"):
        make_model().forward_features(incompatible_features)


def test_voxelnet_rejects_non_sparse_backbone():
    with pytest.raises(TypeError, match="SparseBackbone"):
        make_model(backbone=nn.Identity())


def test_voxelnet_rejects_vfe_channels_incompatible_with_backbone():
    with pytest.raises(ValueError, match="feature channels"):
        make_model(backbone=IncompatibleFeatureBackbone()).forward_features(make_voxel_batch())


def test_voxelnet_rejects_module_postprocessor():
    with pytest.raises(TypeError, match="postprocessor"):
        make_model(postprocessor=ModulePostprocessor())


def test_voxelnet_supports_empty_voxel_batches():
    empty_voxels = VoxelBatch(
        voxels=torch.empty((0, 10, 5)),
        num_points=torch.empty((0,), dtype=torch.int32),
        coordinates=torch.empty((0, 4), dtype=torch.int32),
        batch_size=1,
    )

    predictions, bev = make_model().forward_features(empty_voxels)

    assert bev.shape == (1, 512, 4, 4)
    assert [task["hm"].shape for task in predictions] == [
        (1, 1, 4, 4),
        (1, 2, 4, 4),
        (1, 2, 4, 4),
        (1, 1, 4, 4),
        (1, 2, 4, 4),
        (1, 2, 4, 4),
    ]
