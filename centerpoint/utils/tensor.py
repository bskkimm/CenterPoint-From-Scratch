"""Tensor indexing operations shared by CenterPoint heads and losses."""

from torch import Tensor


def gather_feature(features: Tensor, indices: Tensor) -> Tensor:
    """Gather ``[B, N, C]`` features at ``[B, M]`` indices."""

    if features.ndim != 3:
        raise ValueError("features must have shape [B, N, C]")
    if indices.ndim != 2 or indices.shape[0] != features.shape[0]:
        raise ValueError("indices must have shape [B, M]")

    expanded = indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
    return features.gather(1, expanded)


def transpose_and_gather_feature(features: Tensor, indices: Tensor) -> Tensor:
    """Flatten NCHW features in row-major spatial order, then gather by index."""

    if features.ndim != 4:
        raise ValueError("features must have shape [B, C, H, W]")
    batch, channels, height, width = features.shape
    flattened = features.permute(0, 2, 3, 1).contiguous().view(
        batch, height * width, channels
    )
    return gather_feature(flattened, indices)
