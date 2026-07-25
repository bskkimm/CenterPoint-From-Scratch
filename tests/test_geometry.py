import math

import pytest
import torch

from centerpoint.utils.geometry import (
    boxes_to_corners_3d,
    boxes_to_nms_format,
    internal_yaw_to_nuscenes,
    limit_period,
    nuscenes_yaw_to_internal,
)


def test_limit_period_matches_official_half_period_interval():
    angles = torch.tensor([-math.pi, math.pi, 3 * math.pi, -3 * math.pi, 0.25])

    wrapped = limit_period(angles)

    torch.testing.assert_allclose(
        wrapped,
        torch.tensor([-math.pi, -math.pi, -math.pi, -math.pi, 0.25]),
    )


def test_nuscenes_yaw_conversion_is_an_involution():
    yaw = torch.tensor([-math.pi, -0.7, 0.0, 1.2, math.pi])

    internal = nuscenes_yaw_to_internal(yaw)
    restored = internal_yaw_to_nuscenes(internal)

    torch.testing.assert_allclose(restored, yaw)


def test_boxes_to_corners_3d_preserves_official_order():
    centers = torch.tensor([[10.0, 20.0, 30.0]])
    dimensions = torch.tensor([[2.0, 4.0, 6.0]])

    corners = boxes_to_corners_3d(centers, dimensions, torch.tensor([0.0]))

    expected = torch.tensor(
        [
            [9.0, 18.0, 27.0],
            [9.0, 18.0, 33.0],
            [9.0, 22.0, 33.0],
            [9.0, 22.0, 27.0],
            [11.0, 18.0, 27.0],
            [11.0, 18.0, 33.0],
            [11.0, 22.0, 33.0],
            [11.0, 22.0, 27.0],
        ]
    )
    torch.testing.assert_allclose(corners[0], expected)


def test_positive_internal_yaw_rotates_clockwise():
    corners = boxes_to_corners_3d(
        torch.zeros((1, 3)),
        torch.tensor([[2.0, 4.0, 2.0]]),
        torch.tensor([math.pi / 2]),
    )

    torch.testing.assert_allclose(corners[0, 0], torch.tensor([-2.0, 1.0, -1.0]))


def test_boxes_to_nms_format_swaps_width_length_and_converts_yaw():
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25]])

    converted = boxes_to_nms_format(boxes)

    torch.testing.assert_allclose(
        converted,
        torch.tensor([[1.0, 2.0, 3.0, 5.0, 4.0, 6.0, -0.25 - math.pi / 2]]),
    )
    torch.testing.assert_allclose(boxes[:, 3:5], torch.tensor([[4.0, 5.0]]))


@pytest.mark.parametrize(
    "centers,dimensions,yaw,message",
    [
        (torch.zeros(3), torch.zeros((1, 3)), torch.zeros(1), "centers"),
        (torch.zeros((1, 3)), torch.zeros((2, 3)), torch.zeros(1), "dimensions"),
        (torch.zeros((1, 3)), torch.zeros((1, 3)), torch.zeros(2), "yaw"),
    ],
)
def test_boxes_to_corners_3d_rejects_invalid_shapes(centers, dimensions, yaw, message):
    with pytest.raises(ValueError, match=message):
        boxes_to_corners_3d(centers, dimensions, yaw)
