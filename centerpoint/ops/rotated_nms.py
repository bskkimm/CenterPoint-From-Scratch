"""Slow CPU rotated-BEV IoU and NMS correctness oracle."""

import math
from typing import List, Sequence, Tuple

import torch
from torch import Tensor


Point = Tuple[float, float]


def pairwise_rotated_bev_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute pairwise IoU for internal ``[x,y,z,w,l,h,yaw]`` boxes."""

    _validate_boxes(boxes_a)
    _validate_boxes(boxes_b)
    output = torch.zeros(
        (boxes_a.shape[0], boxes_b.shape[0]),
        dtype=torch.float64,
        device=boxes_a.device,
    )
    polygons_a = [_corners(box) for box in boxes_a]
    polygons_b = [_corners(box) for box in boxes_b]
    areas_a = [max(float(box[3]) * float(box[4]), 0.0) for box in boxes_a]
    areas_b = [max(float(box[3]) * float(box[4]), 0.0) for box in boxes_b]
    for row, polygon_a in enumerate(polygons_a):
        for column, polygon_b in enumerate(polygons_b):
            intersection = _polygon_area(_clip_polygon(polygon_a, polygon_b))
            union = areas_a[row] + areas_b[column] - intersection
            if union > 0:
                output[row, column] = intersection / max(union, 1e-8)
    return output


def rotated_nms(
    boxes: Tensor,
    scores: Tensor,
    *,
    iou_threshold: float,
    pre_max_size: int,
    post_max_size: int,
) -> Tensor:
    """Greedily suppress boxes by rotated BEV IoU and return original indices."""

    _validate_boxes(boxes)
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape [N]")
    if scores.device != boxes.device:
        raise ValueError("boxes and scores must be on the same device")
    if pre_max_size <= 0 or post_max_size <= 0:
        raise ValueError("NMS limits must be positive")
    if iou_threshold < 0:
        raise ValueError("iou_threshold must be non-negative")
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    order = scores.sort(0, descending=True)[1][:pre_max_size]
    keep: List[int] = []
    while order.numel():
        current = int(order[0])
        keep.append(current)
        if len(keep) >= post_max_size or order.numel() == 1:
            break
        remaining = order[1:]
        overlap = pairwise_rotated_bev_iou(
            boxes[current : current + 1],
            boxes[remaining],
        )[0]
        order = remaining[overlap <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def _validate_boxes(boxes: Tensor) -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("boxes must have shape [N, 7]")
    if boxes.device.type != "cpu":
        raise ValueError("the rotated IoU oracle requires CPU tensors")
    if not torch.is_floating_point(boxes):
        raise ValueError("boxes must use a floating dtype")
    if boxes.numel() and (not torch.isfinite(boxes).all() or torch.any(boxes[:, 3:5] < 0)):
        raise ValueError("boxes must be finite with non-negative width and length")


def _corners(box: Tensor) -> List[Point]:
    center_x, center_y = float(box[0]), float(box[1])
    half_width, half_length = float(box[3]) / 2, float(box[4]) / 2
    yaw = float(box[6])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    corners = []
    for local_x, local_y in (
        (-half_width, -half_length),
        (half_width, -half_length),
        (half_width, half_length),
        (-half_width, half_length),
    ):
        corners.append(
            (
                center_x + local_x * cosine + local_y * sine,
                center_y - local_x * sine + local_y * cosine,
            )
        )
    return corners


def _clip_polygon(subject: Sequence[Point], clip: Sequence[Point]) -> List[Point]:
    output = list(subject)
    for clip_index in range(len(clip)):
        edge_start = clip[clip_index]
        edge_end = clip[(clip_index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside(current, edge_start, edge_end)
            previous_inside = _inside(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def _inside(point: Point, edge_start: Point, edge_end: Point) -> bool:
    return (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    ) >= -1e-12


def _intersection(start: Point, end: Point, clip_start: Point, clip_end: Point) -> Point:
    direction = (end[0] - start[0], end[1] - start[1])
    clip_direction = (clip_end[0] - clip_start[0], clip_end[1] - clip_start[1])
    denominator = direction[0] * clip_direction[1] - direction[1] * clip_direction[0]
    if abs(denominator) < 1e-12:
        return end
    offset = (clip_start[0] - start[0], clip_start[1] - start[1])
    factor = (offset[0] * clip_direction[1] - offset[1] * clip_direction[0]) / denominator
    return start[0] + factor * direction[0], start[1] + factor * direction[1]


def _polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    twice_area = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        twice_area += point[0] * following[1] - following[0] * point[1]
    return abs(twice_area) / 2
