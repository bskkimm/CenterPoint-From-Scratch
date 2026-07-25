import pytest
import torch

from centerpoint.utils.tensor import gather_feature, transpose_and_gather_feature


def test_gather_feature_selects_each_batch_independently():
    features = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]
    )
    indices = torch.tensor([[2, 0], [1, 1]])

    gathered = gather_feature(features, indices)

    torch.testing.assert_allclose(
        gathered,
        torch.tensor([[[4, 5], [0, 1]], [[8, 9], [8, 9]]]),
    )


def test_transpose_and_gather_uses_row_major_spatial_indices():
    features = torch.tensor(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]]]
    )

    gathered = transpose_and_gather_feature(features, torch.tensor([[5, 1, 3]]))

    torch.testing.assert_allclose(
        gathered,
        torch.tensor([[[5.0, 15.0], [1.0, 11.0], [3.0, 13.0]]]),
    )


def test_gather_utilities_validate_shapes():
    with pytest.raises(ValueError, match="features"):
        gather_feature(torch.zeros((2, 3)), torch.zeros((2, 1), dtype=torch.long))
    with pytest.raises(ValueError, match="indices"):
        gather_feature(torch.zeros((2, 3, 4)), torch.zeros((1, 1), dtype=torch.long))
    with pytest.raises(ValueError, match="features"):
        transpose_and_gather_feature(
            torch.zeros((2, 3, 4)), torch.zeros((2, 1), dtype=torch.long)
        )
