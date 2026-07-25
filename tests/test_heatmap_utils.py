import math

import pytest
import torch

from centerpoint.utils.heatmap import draw_gaussian, gaussian_2d, gaussian_radius


def test_gaussian_radius_matches_official_three_root_formula():
    height, width, overlap = 20.0, 10.0, 0.1
    roots = (
        (height + width + math.sqrt((height + width) ** 2 - 4 * width * height * 0.9 / 1.1)) / 2,
        (2 * (height + width) + math.sqrt((2 * (height + width)) ** 2 - 16 * 0.9 * width * height)) / 2,
        (-2 * overlap * (height + width) + math.sqrt((2 * overlap * (height + width)) ** 2 - 16 * overlap * (overlap - 1) * width * height)) / 2,
    )

    assert gaussian_radius((height, width), overlap) == pytest.approx(min(roots))


def test_gaussian_2d_is_centered_and_unnormalized():
    gaussian = gaussian_2d((3, 3), sigma=0.5)

    assert gaussian[1, 1].item() == 1.0
    torch.testing.assert_allclose(gaussian, torch.flip(gaussian, dims=(0, 1)))


def test_draw_gaussian_crops_at_border_without_renormalizing():
    heatmap = torch.zeros((4, 4))

    draw_gaussian(heatmap, center=(0.0, 0.0), radius=2)

    assert heatmap[0, 0].item() == 1.0
    assert torch.count_nonzero(heatmap).item() == 9


def test_draw_gaussian_combines_overlaps_by_maximum():
    heatmap = torch.zeros((5, 5))
    draw_gaussian(heatmap, center=(2.0, 2.0), radius=1, k=0.5)
    first = heatmap.clone()

    draw_gaussian(heatmap, center=(2.0, 2.0), radius=1, k=0.25)

    torch.testing.assert_allclose(heatmap, first)


def test_draw_gaussian_truncates_fractional_center_toward_zero():
    heatmap = torch.zeros((3, 3))

    draw_gaussian(heatmap, center=(-0.2, 1.8), radius=0)

    assert heatmap[1, 0].item() == 1.0


def test_draw_gaussian_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="heatmap"):
        draw_gaussian(torch.zeros((1, 2, 3)), center=(1, 1), radius=1)
    with pytest.raises(ValueError, match="non-negative"):
        draw_gaussian(torch.zeros((3, 3)), center=(1, 1), radius=-1)
