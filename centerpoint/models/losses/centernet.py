"""Official CenterPoint heatmap and box regression losses."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from centerpoint.utils.tensor import transpose_and_gather_feature


def clipped_sigmoid(logits: Tensor) -> Tensor:
    """Apply the official sigmoid clamp used before focal loss."""

    return logits.sigmoid().clamp(min=1e-4, max=1 - 1e-4)


class FastFocalLoss(nn.Module):
    """CornerNet focal loss using indexed positive centers."""

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        indices: Tensor,
        mask: Tensor,
        categories: Tensor,
    ) -> Tensor:
        """Calculate heatmap loss from probabilities and CenterPoint targets."""

        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("prediction and target must have equal [B, C, H, W] shapes")
        if indices.shape != mask.shape or indices.shape != categories.shape:
            raise ValueError("indices, mask, and categories must have equal [B, M] shapes")

        negative_weights = torch.pow(1 - target, 4)
        negative_loss = (
            torch.log(1 - prediction) * torch.pow(prediction, 2) * negative_weights
        ).sum()

        gathered = transpose_and_gather_feature(prediction, indices)
        positive_prediction = gathered.gather(2, categories.unsqueeze(2)).squeeze(2)
        positive_loss = (
            torch.log(positive_prediction)
            * torch.pow(1 - positive_prediction, 2)
            * mask.to(prediction.dtype)
        ).sum()

        positive_count = mask.to(prediction.dtype).sum()
        if positive_count.item() == 0:
            return -negative_loss
        return -(positive_loss + negative_loss) / positive_count


class RegressionLoss(nn.Module):
    """Per-code gathered L1 loss normalized by the number of objects."""

    def forward(
        self,
        prediction: Tensor,
        mask: Tensor,
        indices: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Return one normalized L1 value for each regression channel."""

        if prediction.ndim != 4:
            raise ValueError("prediction must have shape [B, C, H, W]")
        if mask.shape != indices.shape or mask.ndim != 2:
            raise ValueError("mask and indices must have equal [B, M] shapes")
        if target.shape != (mask.shape[0], mask.shape[1], prediction.shape[1]):
            raise ValueError("target must have shape [B, M, C]")

        gathered = transpose_and_gather_feature(prediction, indices)
        expanded_mask = mask.unsqueeze(2).to(prediction.dtype)
        loss = F.l1_loss(
            gathered * expanded_mask,
            target * expanded_mask,
            reduction="none",
        )
        loss = loss / (expanded_mask.sum() + 1e-4)
        return loss.sum(dim=(0, 1))
