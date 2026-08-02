"""Pinned two-stage BEV RPN neck used by the nuScenes VoxelNet baseline."""

from typing import Sequence

import torch
from torch import Tensor, nn


class RPN(nn.Module):
    """Fuse two dense BEV scales using the official module layout."""

    def __init__(
        self,
        layer_nums: Sequence[int],
        ds_layer_strides: Sequence[int],
        ds_num_filters: Sequence[int],
        us_layer_strides: Sequence[int],
        us_num_filters: Sequence[int],
        num_input_features: int,
    ) -> None:
        super().__init__()
        lengths = {
            len(layer_nums),
            len(ds_layer_strides),
            len(ds_num_filters),
            len(us_layer_strides),
            len(us_num_filters),
        }
        if lengths != {2}:
            raise ValueError("the frozen RPN requires exactly two stages")
        if num_input_features <= 0:
            raise ValueError("num_input_features must be positive")
        if any(value <= 0 for values in (
            layer_nums,
            ds_layer_strides,
            ds_num_filters,
            us_layer_strides,
            us_num_filters,
        ) for value in values):
            raise ValueError("RPN layer counts, strides, and channels must be positive")

        input_filters = [num_input_features, *ds_num_filters[:-1]]
        blocks = []
        deblocks = []
        for index in range(2):
            block_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    input_filters[index],
                    ds_num_filters[index],
                    3,
                    stride=ds_layer_strides[index],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(ds_num_filters[index], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(layer_nums[index]):
                block_layers.extend(
                    [
                        nn.Conv2d(
                            ds_num_filters[index],
                            ds_num_filters[index],
                            3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(ds_num_filters[index], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )
            blocks.append(nn.Sequential(*block_layers))

            stride = us_layer_strides[index]
            convolution = (
                nn.Conv2d(
                    ds_num_filters[index],
                    us_num_filters[index],
                    stride,
                    stride=stride,
                    bias=False,
                )
                if stride == 1
                else nn.ConvTranspose2d(
                    ds_num_filters[index],
                    us_num_filters[index],
                    stride,
                    stride=stride,
                    bias=False,
                )
            )
            deblocks.append(
                nn.Sequential(
                    convolution,
                    nn.BatchNorm2d(us_num_filters[index], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, features: Tensor) -> Tensor:
        """Return concatenated same-resolution BEV branches."""

        if features.ndim != 4:
            raise ValueError("features must have shape [B, C, H, W]")
        ups = []
        output = features
        for block, deblock in zip(self.blocks, self.deblocks):
            output = block(output)
            ups.append(deblock(output))
        if ups[0].shape[2:] != ups[1].shape[2:]:
            raise ValueError("RPN branches must produce equal spatial shapes")
        return torch.cat(ups, dim=1)
