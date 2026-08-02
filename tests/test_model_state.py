import torch
from torch import nn

from centerpoint.config import NUSCENES_VOXELNET_075
from centerpoint.engine.checkpoint import load_checkpoint, save_checkpoint
from centerpoint.models.heads import CenterHead
from centerpoint.models.necks import RPN


def make_model():
    config = NUSCENES_VOXELNET_075
    model = nn.Module()
    model.neck = RPN(
        layer_nums=config.model.neck.layer_numbers,
        ds_layer_strides=config.model.neck.downsample_strides,
        ds_num_filters=config.model.neck.downsample_filters,
        us_layer_strides=config.model.neck.upsample_strides,
        us_num_filters=config.model.neck.upsample_filters,
        num_input_features=config.model.neck.input_channels,
    )
    model.bbox_head = CenterHead(
        in_channels=config.model.head.input_channels,
        tasks=config.tasks,
        common_heads={
            branch.name: (branch.output_channels, branch.num_convolutions)
            for branch in config.model.head.branches
        },
        share_conv_channel=config.model.head.shared_channels,
        loss_weight=config.model.head.loss_weight,
        code_weights=config.model.head.code_weights,
    )
    return model


def test_neck_and_head_round_trip_through_versioned_checkpoint(tmp_path):
    torch.manual_seed(41)
    model = make_model()
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    path = tmp_path / "model.pth"
    save_checkpoint(path, model=model, config={}, epoch=0, global_step=0)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for buffer in model.buffers():
            buffer.zero_()
    load_checkpoint(path, model=model)

    assert tuple(model.state_dict()) == tuple(expected)
    for name, value in model.state_dict().items():
        torch.testing.assert_allclose(value, expected[name])
