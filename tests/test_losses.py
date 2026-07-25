import math

import pytest
import torch

from centerpoint.models.losses import FastFocalLoss, RegressionLoss, clipped_sigmoid


def test_clipped_sigmoid_bounds_extreme_logits():
    probabilities = clipped_sigmoid(torch.tensor([-100.0, 0.0, 100.0]))

    torch.testing.assert_allclose(
        probabilities,
        torch.tensor([1e-4, 0.5, 1 - 1e-4]),
    )


def test_fast_focal_loss_matches_indexed_official_formula():
    prediction = torch.tensor([[[[0.2, 0.7]], [[0.4, 0.1]]]])
    target = torch.tensor([[[[0.0, 1.0]], [[0.5, 0.0]]]])
    indices = torch.tensor([[1]])
    categories = torch.tensor([[0]])
    mask = torch.tensor([[1]], dtype=torch.uint8)

    loss = FastFocalLoss()(prediction, target, indices, mask, categories)

    negative = (
        math.log(0.8) * 0.2**2
        + math.log(0.6) * 0.4**2 * 0.5**4
        + math.log(0.9) * 0.1**2
    )
    positive = math.log(0.7) * 0.3**2
    assert loss.item() == pytest.approx(-(negative + positive))


def test_fast_focal_loss_does_not_normalize_empty_positive_case():
    prediction = torch.full((2, 1, 1, 1), 0.25)
    target = torch.zeros_like(prediction)
    indices = torch.zeros((2, 1), dtype=torch.long)
    mask = torch.zeros((2, 1), dtype=torch.uint8)
    categories = torch.zeros((2, 1), dtype=torch.long)

    loss = FastFocalLoss()(prediction, target, indices, mask, categories)

    expected = -2 * math.log(0.75) * 0.25**2
    assert loss.item() == pytest.approx(expected)


def test_regression_loss_returns_per_code_values_normalized_by_objects():
    prediction = torch.tensor(
        [[[[1.0, 5.0]], [[2.0, 8.0]], [[3.0, 9.0]]]]
    )
    indices = torch.tensor([[0, 1]])
    mask = torch.tensor([[1, 1]], dtype=torch.uint8)
    target = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]]])

    loss = RegressionLoss()(prediction, mask, indices, target)

    denominator = 2.0001
    torch.testing.assert_allclose(
        loss,
        torch.tensor([(1 + 2) / denominator, (2 + 4) / denominator, (3 + 4) / denominator]),
    )


def test_regression_loss_is_zero_without_objects():
    loss = RegressionLoss()(
        torch.randn((1, 2, 2, 2)),
        torch.zeros((1, 3), dtype=torch.uint8),
        torch.zeros((1, 3), dtype=torch.long),
        torch.randn((1, 3, 2)),
    )

    torch.testing.assert_allclose(loss, torch.zeros(2))


def test_losses_reject_incompatible_shapes():
    with pytest.raises(ValueError, match="prediction and target"):
        FastFocalLoss()(
            torch.zeros((1, 1, 1, 1)),
            torch.zeros((1, 2, 1, 1)),
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1), dtype=torch.uint8),
            torch.zeros((1, 1), dtype=torch.long),
        )
    with pytest.raises(ValueError, match="target"):
        RegressionLoss()(
            torch.zeros((1, 2, 1, 1)),
            torch.ones((1, 1), dtype=torch.uint8),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1, 3)),
        )
