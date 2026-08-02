import pytest
import torch

from centerpoint.config import NUSCENES_VOXELNET_075
from centerpoint.models.necks import RPN


def make_neck():
    config = NUSCENES_VOXELNET_075.model.neck
    return RPN(
        layer_nums=config.layer_numbers,
        ds_layer_strides=config.downsample_strides,
        ds_num_filters=config.downsample_filters,
        us_layer_strides=config.upsample_strides,
        us_num_filters=config.upsample_filters,
        num_input_features=config.input_channels,
    )


def test_rpn_matches_canonical_parameter_and_state_layout():
    neck = make_neck()

    assert sum(parameter.numel() for parameter in neck.parameters()) == 4_576_768
    state = neck.state_dict()
    assert state["blocks.0.1.weight"].shape == (128, 256, 3, 3)
    assert state["blocks.1.1.weight"].shape == (256, 128, 3, 3)
    assert state["deblocks.0.0.weight"].shape == (256, 128, 1, 1)
    assert state["deblocks.1.0.weight"].shape == (256, 256, 2, 2)
    assert "blocks.0.1.bias" not in state
    assert state["blocks.0.2.num_batches_tracked"].shape == ()


def test_rpn_fuses_two_branches_and_backpropagates():
    torch.manual_seed(3)
    neck = make_neck()
    features = torch.randn((1, 256, 8, 8), requires_grad=True)

    output = neck(features)
    output.square().mean().backward()

    assert output.shape == (1, 512, 8, 8)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert neck.blocks[0][1].weight.grad is not None
    assert neck.blocks[0][2].eps == 1e-3
    assert neck.blocks[0][2].momentum == 0.01


def test_rpn_rejects_spatial_branch_mismatch():
    neck = make_neck()

    with pytest.raises(ValueError, match="equal spatial"):
        neck(torch.zeros((1, 256, 7, 7)))
