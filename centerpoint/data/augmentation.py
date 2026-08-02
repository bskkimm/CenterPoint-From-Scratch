"""Pinned NumPy augmentation primitives for nuScenes training samples."""

import math
from typing import Any, Sequence, Tuple, Union

import numpy as np


def augment_global(
    points: np.ndarray,
    boxes: np.ndarray,
    *,
    rotation_range: Sequence[float],
    scale_range: Sequence[float],
    translation_std: Union[float, Sequence[float]],
    shuffle_points: bool = True,
    rng: Any = np.random,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the official flip, rotation, scale, translation, and shuffle order."""

    _validate_inputs(points, boxes)
    if len(rotation_range) != 2 or len(scale_range) != 2:
        raise ValueError("rotation_range and scale_range must contain two values")
    augmented_points = points.copy()
    augmented_boxes = boxes.copy()

    if bool(rng.choice([False, True], replace=False, p=[0.5, 0.5])):
        augmented_points[:, 1] = -augmented_points[:, 1]
        augmented_boxes[:, 1] = -augmented_boxes[:, 1]
        augmented_boxes[:, 7] = -augmented_boxes[:, 7]
        augmented_boxes[:, 8] = -augmented_boxes[:, 8] + math.pi

    if bool(rng.choice([False, True], replace=False, p=[0.5, 0.5])):
        augmented_points[:, 0] = -augmented_points[:, 0]
        augmented_boxes[:, 0] = -augmented_boxes[:, 0]
        augmented_boxes[:, 6] = -augmented_boxes[:, 6]
        augmented_boxes[:, 8] = -augmented_boxes[:, 8] + 2 * math.pi

    angle = float(rng.uniform(rotation_range[0], rotation_range[1]))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=augmented_points.dtype,
    )
    augmented_points[:, :3] = augmented_points[:, :3] @ rotation
    augmented_boxes[:, :3] = augmented_boxes[:, :3] @ rotation
    velocity = np.column_stack(
        (
            augmented_boxes[:, 6:8],
            np.zeros((augmented_boxes.shape[0],), dtype=augmented_boxes.dtype),
        )
    )
    augmented_boxes[:, 6:8] = (velocity @ rotation)[:, :2]
    augmented_boxes[:, 8] += angle

    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    augmented_points[:, :3] *= scale
    augmented_boxes[:, :-1] *= scale

    if np.isscalar(translation_std):
        standard_deviation = (float(translation_std),) * 3
    else:
        standard_deviation = tuple(float(value) for value in translation_std)
        if len(standard_deviation) != 3:
            raise ValueError("translation_std must be scalar or contain three values")
    # The pinned helper uses the x standard deviation for z as well.
    translation = np.array(
        [
            rng.normal(0, standard_deviation[0], 1)[0],
            rng.normal(0, standard_deviation[1], 1)[0],
            rng.normal(0, standard_deviation[0], 1)[0],
        ],
        dtype=augmented_points.dtype,
    )
    augmented_points[:, :3] += translation
    augmented_boxes[:, :3] += translation

    if shuffle_points:
        rng.shuffle(augmented_points)
    return augmented_points, augmented_boxes


def filter_boxes_by_bev_range(
    boxes: np.ndarray,
    xy_range: Sequence[float],
) -> np.ndarray:
    """Keep boxes with at least one BEV corner strictly inside the range."""

    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError("boxes must have shape [N, 9]")
    if len(xy_range) != 4:
        raise ValueError("xy_range must contain min_x, min_y, max_x, max_y")
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.bool_)

    template = np.array(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
        dtype=boxes.dtype,
    )
    local = template[None, :, :] * boxes[:, None, 3:5]
    cosine = np.cos(boxes[:, 8])[:, None]
    sine = np.sin(boxes[:, 8])[:, None]
    x = local[..., 0] * cosine + local[..., 1] * sine + boxes[:, None, 0]
    y = -local[..., 0] * sine + local[..., 1] * cosine + boxes[:, None, 1]
    minimum_x, minimum_y, maximum_x, maximum_y = xy_range
    inside = (x > minimum_x) & (x < maximum_x) & (y > minimum_y) & (y < maximum_y)
    return np.any(inside, axis=1)


def select_classes(
    boxes: np.ndarray,
    names: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter ignored names and assign 1-based global class identifiers."""

    if boxes.ndim != 2 or boxes.shape[1] != 9 or names.shape != (boxes.shape[0],):
        raise ValueError("boxes and names must have shapes [N, 9] and [N]")
    class_to_id = {name: index + 1 for index, name in enumerate(class_names)}
    keep = np.array([name in class_to_id for name in names], dtype=np.bool_)
    selected_names = names[keep]
    class_ids = np.array([class_to_id[name] for name in selected_names], dtype=np.int64)
    return boxes[keep], selected_names, class_ids


def _validate_inputs(points: np.ndarray, boxes: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape [N, F] with F >= 3")
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError("boxes must have shape [N, 9]")
    if not np.issubdtype(points.dtype, np.floating) or not np.issubdtype(boxes.dtype, np.floating):
        raise ValueError("points and boxes must use floating dtypes")
