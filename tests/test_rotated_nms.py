import math

import pytest
import torch

from centerpoint.ops import pairwise_rotated_bev_iou, rotated_nms


def box(x=0, y=0, width=2, length=2, yaw=0):
    return [x, y, 0, width, length, 1, yaw]


def test_rotated_iou_covers_identity_overlap_rotation_and_separation():
    boxes = torch.tensor(
        [
            box(),
            box(x=1),
            box(x=3),
            box(width=2, length=4, yaw=math.pi / 2),
        ],
        dtype=torch.float32,
    )

    iou = pairwise_rotated_bev_iou(boxes, boxes)

    torch.testing.assert_allclose(torch.diag(iou), torch.ones(4, dtype=torch.float64))
    assert iou[0, 1].item() == pytest.approx(1 / 3)
    assert iou[0, 2].item() == 0
    assert iou[0, 3].item() == pytest.approx(0.5)
    torch.testing.assert_allclose(iou, iou.T)


def test_rotated_nms_is_class_agnostic_and_uses_strict_threshold():
    boxes = torch.tensor([box(), box(x=1), box(x=4)], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7])
    exact_overlap = pairwise_rotated_bev_iou(boxes[:1], boxes[1:2]).item()

    retained_at_threshold = rotated_nms(
        boxes,
        scores,
        iou_threshold=exact_overlap,
        pre_max_size=10,
        post_max_size=10,
    )
    suppressed_above_threshold = rotated_nms(
        boxes,
        scores,
        iou_threshold=exact_overlap - 1e-6,
        pre_max_size=10,
        post_max_size=10,
    )

    assert retained_at_threshold.tolist() == [0, 1, 2]
    assert suppressed_above_threshold.tolist() == [0, 2]


def test_rotated_nms_applies_pre_and_post_limits_in_score_order():
    boxes = torch.tensor([box(x=index * 3) for index in range(5)], dtype=torch.float32)
    scores = torch.tensor([0.1, 0.5, 0.4, 0.3, 0.2])

    keep = rotated_nms(
        boxes,
        scores,
        iou_threshold=0.2,
        pre_max_size=3,
        post_max_size=2,
    )

    assert keep.tolist() == [1, 2]


def test_rotated_nms_supports_empty_and_zero_area_boxes():
    empty = torch.empty((0, 7))
    assert rotated_nms(
        empty,
        torch.empty((0,)),
        iou_threshold=0.2,
        pre_max_size=1,
        post_max_size=1,
    ).shape == (0,)

    zero = torch.tensor([box(width=0), box(width=0)], dtype=torch.float32)
    assert pairwise_rotated_bev_iou(zero, zero).sum().item() == 0


def test_rotated_iou_rejects_invalid_or_non_cpu_contracts():
    with pytest.raises(ValueError, match="shape"):
        pairwise_rotated_bev_iou(torch.zeros((1, 9)), torch.zeros((1, 7)))
    with pytest.raises(ValueError, match="non-negative"):
        pairwise_rotated_bev_iou(
            torch.tensor([box(width=-1)], dtype=torch.float32),
            torch.tensor([box()], dtype=torch.float32),
        )
