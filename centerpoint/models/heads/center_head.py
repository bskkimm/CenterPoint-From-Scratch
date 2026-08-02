"""Trainable six-task CenterHead from the pinned nuScenes baseline."""

from collections import defaultdict
from typing import DefaultDict, Dict, List, Mapping, MutableMapping, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from centerpoint.contracts import TaskPredictions, TaskTargets
from centerpoint.models.losses import FastFocalLoss, RegressionLoss, clipped_sigmoid


Prediction = Union[Mapping[str, Tensor], TaskPredictions]


class SepHead(nn.Module):
    """Independent convolution branches for one detection task."""

    def __init__(
        self,
        in_channels: int,
        heads: Mapping[str, Tuple[int, int]],
        head_channels: int = 64,
        final_kernel: int = 3,
        init_bias: float = -2.19,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or head_channels <= 0 or final_kernel <= 0:
            raise ValueError("head channels and kernel size must be positive")
        for name, (output_channels, num_convolutions) in heads.items():
            if output_channels <= 0 or num_convolutions <= 0:
                raise ValueError("head outputs and convolution counts must be positive")
            layers = []
            current_channels = in_channels
            for _ in range(num_convolutions - 1):
                layers.extend(
                    [
                        nn.Conv2d(current_channels, head_channels, 3, padding=1, bias=True),
                        nn.BatchNorm2d(head_channels),
                        nn.ReLU(),
                    ]
                )
                current_channels = head_channels
            layers.append(
                nn.Conv2d(
                    current_channels,
                    output_channels,
                    final_kernel,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )
            branch = nn.Sequential(*layers)
            if "hm" in name:
                nn.init.constant_(branch[-1].bias, init_bias)
            else:
                for module in branch.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            module.weight,
                            a=0,
                            mode="fan_out",
                            nonlinearity="relu",
                        )
                        nn.init.constant_(module.bias, 0)
            self.add_module(name, branch)

        self._head_names = tuple(heads)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        return {name: getattr(self, name)(features) for name in self._head_names}


class CenterHead(nn.Module):
    """Produce and train the six canonical nuScenes CenterPoint tasks."""

    def __init__(
        self,
        in_channels: int,
        tasks: Sequence[Sequence[str]],
        common_heads: Mapping[str, Tuple[int, int]],
        share_conv_channel: int = 64,
        num_hm_conv: int = 2,
        init_bias: float = -2.19,
        loss_weight: float = 0.25,
        code_weights: Sequence[float] = (1, 1, 1, 1, 1, 1, 0.2, 0.2, 1, 1),
    ) -> None:
        super().__init__()
        if not tasks or any(not task for task in tasks):
            raise ValueError("tasks must contain at least one class each")
        if tuple(common_heads) != ("reg", "height", "dim", "rot", "vel"):
            raise ValueError("common_heads must use canonical branch order")
        if len(code_weights) != 10:
            raise ValueError("code_weights must contain ten regression weights")

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel, 3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True),
        )
        task_heads = []
        for task in tasks:
            heads = dict(common_heads)
            heads["hm"] = (len(task), num_hm_conv)
            task_heads.append(
                SepHead(
                    share_conv_channel,
                    heads,
                    head_channels=share_conv_channel,
                    final_kernel=3,
                    init_bias=init_bias,
                )
            )
        self.tasks = nn.ModuleList(task_heads)
        self.crit = FastFocalLoss()
        self.crit_reg = RegressionLoss()
        self.loss_weight = float(loss_weight)
        self.code_weights = tuple(float(weight) for weight in code_weights)

    def forward(self, features: Tensor) -> Tuple[List[Dict[str, Tensor]], Tensor]:
        """Return official task dictionaries and the shared task feature map."""

        if features.ndim != 4:
            raise ValueError("features must have shape [B, C, H, W]")
        shared = self.shared_conv(features)
        return [task(shared) for task in self.tasks], shared

    def loss(
        self,
        predictions: Sequence[Prediction],
        targets: Sequence[TaskTargets],
    ) -> Dict[str, List[Tensor]]:
        """Return the official per-task loss lists without averaging tasks."""

        if len(predictions) != len(self.tasks) or len(targets) != len(self.tasks):
            raise ValueError("predictions and targets must match the configured task count")

        losses: DefaultDict[str, List[Tensor]] = defaultdict(list)
        for prediction, target in zip(predictions, targets):
            values = prediction.as_dict() if isinstance(prediction, TaskPredictions) else prediction
            missing = {"hm", "reg", "height", "dim", "vel", "rot"}.difference(values)
            if missing:
                raise ValueError(f"prediction is missing heads: {sorted(missing)}")

            heatmap = clipped_sigmoid(values["hm"])
            if isinstance(values, MutableMapping):
                values["hm"] = heatmap
            heatmap_loss = self.crit(
                heatmap,
                target.heatmap,
                target.indices,
                target.mask,
                target.categories,
            )
            annotation = torch.cat(
                [
                    values["reg"],
                    values["height"],
                    values["dim"],
                    values["vel"],
                    values["rot"],
                ],
                dim=1,
            )
            if isinstance(values, MutableMapping):
                values["anno_box"] = annotation
            box_loss = self.crit_reg(
                annotation,
                target.mask,
                target.indices,
                target.annotation,
            )
            weights = box_loss.new_tensor(self.code_weights)
            location_loss = (box_loss * weights).sum()
            task_loss = heatmap_loss + self.loss_weight * location_loss

            losses["loss"].append(task_loss)
            losses["hm_loss"].append(heatmap_loss.detach().cpu())
            losses["loc_loss"].append(location_loss)
            losses["loc_loss_elem"].append(box_loss.detach().cpu())
            losses["num_positive"].append(target.mask.float().sum())
        return dict(losses)
