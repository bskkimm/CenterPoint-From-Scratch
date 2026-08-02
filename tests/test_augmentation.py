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
