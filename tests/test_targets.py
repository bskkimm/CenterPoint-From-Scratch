import math

import torch

from centerpoint.data import CenterTargetAssigner


def make_assigner(tasks=(("car",),), max_objects=5):
    return CenterTargetAssigner(
        tasks=tasks,
        voxel_size=(1.0, 1.0, 1.0),
        point_cloud_range=(0.0, 0.0, -5.0, 4.0, 4.0, 5.0),
        output_stride=1,
        gaussian_overlap=0.1,
        max_objects=max_objects,
        min_radius=0,
    )


def test_assigns_exact_center_and_ten_regression_codes():
    box = torch.tensor([[1.25, 2.75, 0.5, 2.0, 1.0, 1.5, 3.0, -2.0, math.pi / 2]])

    target = make_assigner()(box, torch.tensor([1]))[0]

    assert target.heatmap.shape == (1, 4, 4)
    assert target.heatmap[0, 2, 1].item() == 1.0
    assert target.indices[0].item() == 2 * 4 + 1
    assert target.mask.tolist() == [1, 0, 0, 0, 0]
    assert target.categories[0].item() == 0
    torch.testing.assert_allclose(
        target.annotation[0],
        torch.tensor(
            [
                0.25,
                0.75,
                0.5,
                math.log(2.0),
                math.log(1.0),
                math.log(1.5),
                3.0,
                -2.0,
                1.0,
                0.0,
            ]
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_groups_objects_by_task_and_class_major_order():
    assigner = make_assigner(tasks=(("car",), ("truck", "bus")))
    boxes = torch.tensor(
        [
            [0.2, 0.2, 0, 1, 1, 1, 0, 0, 0],
            [1.2, 1.2, 0, 1, 1, 1, 0, 0, 0],
            [2.2, 2.2, 0, 1, 1, 1, 0, 0, 0],
            [3.2, 3.2, 0, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    car, vehicle = assigner(boxes, torch.tensor([3, 2, 3, 1]))

    assert car.indices[0].item() == 15
    assert vehicle.indices[:3].tolist() == [5, 0, 10]
    assert vehicle.categories[:3].tolist() == [0, 1, 1]


def test_invalid_object_does_not_compact_later_target_slot():
    boxes = torch.tensor(
        [
            [1.0, 1.0, 0, 0, 1, 1, 0, 0, 0],
            [2.0, 2.0, 0, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    target = make_assigner()(boxes, torch.tensor([1, 1]))[0]

    assert target.mask.tolist() == [0, 1, 0, 0, 0]
    assert target.indices.tolist()[:2] == [0, 10]


def test_fractional_negative_center_uses_truncation_toward_zero():
    box = torch.tensor([[-0.2, 1.8, 0, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)

    target = make_assigner()(box, torch.tensor([1]))[0]

    assert target.mask[0].item() == 1
    assert target.indices[0].item() == 4
    torch.testing.assert_allclose(target.annotation[0, :2], torch.tensor([-0.2, 0.8]))


def test_colliding_centers_keep_both_regression_records():
    boxes = torch.tensor(
        [
            [1.2, 1.2, 0, 1, 1, 1, 0, 0, 0],
            [1.8, 1.8, 0, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    target = make_assigner()(boxes, torch.tensor([1, 1]))[0]

    assert target.indices[:2].tolist() == [5, 5]
    assert target.mask[:2].tolist() == [1, 1]
    torch.testing.assert_allclose(target.annotation[:2, :2], torch.tensor([[0.2, 0.2], [0.8, 0.8]]))


def test_assignment_considers_only_max_objects():
    boxes = torch.tensor(
        [
            [0.2, 0.2, 0, 1, 1, 1, 0, 0, 0],
            [1.2, 1.2, 0, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    target = make_assigner(max_objects=1)(boxes, torch.tensor([1, 1]))[0]

    assert target.mask.tolist() == [1]
    assert target.indices.tolist() == [0]
