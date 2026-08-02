import math

import numpy as np

from centerpoint.data import augment_global, filter_boxes_by_bev_range, select_classes


class FixedRNG:
    def __init__(self, flips, angle, scale, translation):
        self.flips = iter(flips)
        self.uniforms = iter((angle, scale))
        self.translation = iter(translation)

    def choice(self, values, **kwargs):
        return next(self.flips)

    def uniform(self, lower, upper):
        return next(self.uniforms)

    def normal(self, mean, std, size):
        return np.array([next(self.translation)])

    def shuffle(self, values):
        values[:] = values[::-1]


def test_augmentation_matches_pinned_operation_order_and_box_fields():
    points = np.array([[1, 2, 3, 9, 8], [4, 5, 6, 7, 6]], dtype=np.float32)
    boxes = np.array([[1, 2, 3, 2, 4, 6, 7, 8, 0.25]], dtype=np.float32)
    rng = FixedRNG(
        flips=(True, False),
        angle=math.pi / 2,
        scale=2.0,
        translation=(10.0, 20.0, 30.0),
    )

    augmented_points, augmented_boxes = augment_global(
        points,
        boxes,
        rotation_range=(-1, 1),
        scale_range=(0.5, 2),
        translation_std=0.5,
        rng=rng,
    )

    np.testing.assert_allclose(
        augmented_points,
        np.array([[0, 12, 42, 7, 6], [6, 18, 36, 9, 8]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        augmented_boxes[0],
        np.array(
            [6, 18, 36, 4, 8, 12, -16, -14, 3 * math.pi / 2 - 0.25],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    np.testing.assert_array_equal(points[:, 3:], np.array([[9, 8], [7, 6]], dtype=np.float32))


def test_both_flips_preserve_official_unnormalized_yaw_result():
    points = np.zeros((0, 5), dtype=np.float32)
    boxes = np.array([[1, 2, 3, 2, 4, 6, 7, 8, 0.25]], dtype=np.float32)
    rng = FixedRNG((True, True), angle=0, scale=1, translation=(0, 0, 0))

    _, augmented = augment_global(
        points,
        boxes,
        rotation_range=(0, 0),
        scale_range=(1, 1),
        translation_std=(1, 2, 3),
        shuffle_points=False,
        rng=rng,
    )

    np.testing.assert_allclose(augmented[0, :2], [-1, -2])
    np.testing.assert_allclose(augmented[0, 6:8], [-7, -8])
    np.testing.assert_allclose(augmented[0, 8], 0.25 + math.pi)


def test_bev_filter_requires_a_strictly_interior_corner():
    boxes = np.array(
        [
            [0, 0, 0, 2, 2, 1, 0, 0, 0],
            [2, 0, 0, 2, 2, 1, 0, 0, 0],
            [0, 0, 0, 20, 20, 1, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    keep = filter_boxes_by_bev_range(boxes, (-1, -1, 1, 1))

    assert keep.tolist() == [False, False, False]
    boxes[0, 3:5] = 1
    assert filter_boxes_by_bev_range(boxes, (-1, -1, 1, 1)).tolist() == [True, False, False]


def test_class_selection_uses_configured_one_based_order():
    boxes = np.arange(27, dtype=np.float32).reshape(3, 9)
    names = np.array(["car", "ignore", "pedestrian"])

    selected_boxes, selected_names, class_ids = select_classes(
        boxes,
        names,
        ("car", "truck", "pedestrian"),
    )

    np.testing.assert_array_equal(selected_boxes, boxes[[0, 2]])
    assert selected_names.tolist() == ["car", "pedestrian"]
    assert class_ids.tolist() == [1, 3]


def test_seeded_float32_augmentation_matches_pinned_helper_dtype_flow():
    points = np.array(
        [[1.234567, -2.345678, 0.456789, 0.2, 0.0], [-3.25, 4.75, 1.5, 0.8, 0.1]],
        dtype=np.float32,
    )
    boxes = np.array(
        [[1.125, -2.25, 0.75, 1.8, 4.2, 1.6, 3.125, -0.875, 0.45]],
        dtype=np.float32,
    )
    expected_points = points.copy()
    expected_boxes = boxes.copy()
    reference_rng = np.random.RandomState(29)
    if reference_rng.choice([False, True], replace=False, p=[0.5, 0.5]):
        expected_boxes[:, 1] *= -1
        expected_boxes[:, -1] = -expected_boxes[:, -1] + np.pi
        expected_points[:, 1] *= -1
        expected_boxes[:, 7] *= -1
    if reference_rng.choice([False, True], replace=False, p=[0.5, 0.5]):
        expected_boxes[:, 0] *= -1
        expected_points[:, 0] *= -1
        expected_boxes[:, -1] = -expected_boxes[:, -1] + 2 * np.pi
        expected_boxes[:, 6] *= -1
    angle = reference_rng.uniform(-0.7, 0.7)
    cosine, sine = np.cos(angle), np.sin(angle)
    for values in (expected_points[:, :3], expected_boxes[:, :3]):
        rotation = np.array(
            [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]],
            dtype=values.dtype,
        )
        values[:] = values @ rotation
    velocity = np.hstack((expected_boxes[:, 6:8], np.zeros((1, 1))))
    velocity_rotation = np.array(
        [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]],
        dtype=velocity.dtype,
    )
    expected_boxes[:, 6:8] = (velocity @ velocity_rotation)[:, :2]
    expected_boxes[:, -1] += angle
    scale = reference_rng.uniform(0.9, 1.1)
    expected_points[:, :3] *= scale
    expected_boxes[:, :-1] *= scale
    translation = np.array(
        [
            reference_rng.normal(0, 0.5, 1),
            reference_rng.normal(0, 0.5, 1),
            reference_rng.normal(0, 0.5, 1),
        ]
    ).T
    expected_points[:, :3] += translation
    expected_boxes[:, :3] += translation
    reference_rng.shuffle(expected_points)

    actual_points, actual_boxes = augment_global(
        points,
        boxes,
        rotation_range=(-0.7, 0.7),
        scale_range=(0.9, 1.1),
        translation_std=0.5,
        rng=np.random.RandomState(29),
    )

    np.testing.assert_array_equal(actual_points, expected_points)
    np.testing.assert_array_equal(actual_boxes, expected_boxes)
