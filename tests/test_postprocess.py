import math

import pytest
import torch

from centerpoint.models.heads import (
    CenterPointDecoder,
    CenterPointPostprocessor,
    DetectionCandidates,
)
from centerpoint.ops import rotated_nms


TASKS = (
    ("car",),
    ("truck", "construction_vehicle"),
    ("bus", "trailer"),
    ("barrier",),
    ("motorcycle", "bicycle"),
    ("pedestrian", "traffic_cone"),
)


def make_prediction(class_count, width=1, preferred_class=0):
    heatmap = torch.full((1, class_count, 1, width), -5.0)
    heatmap[:, preferred_class] = 5.0
    return {
        "hm": heatmap,
        "reg": torch.zeros((1, 2, 1, width)),
        "height": torch.zeros((1, 1, 1, width)),
        "dim": torch.full((1, 3, 1, width), math.log(2.0)),
        "vel": torch.zeros((1, 2, 1, width)),
        "rot": torch.cat(
            (torch.zeros((1, 1, 1, width)), torch.ones((1, 1, 1, width))),
            dim=1,
        ),
    }


def make_postprocessor(nms=rotated_nms, post_max_size=83):
    return CenterPointPostprocessor(
        CenterPointDecoder(
            voxel_size=(1, 1, 1),
            point_cloud_range=(0, 0, -5, 10, 10, 5),
            output_stride=1,
            score_threshold=0.1,
            post_center_range=(0, 0, -5, 10, 10, 5),
        ),
        nms,
        TASKS,
        iou_threshold=0.2,
        pre_max_size=1000,
        post_max_size=post_max_size,
    )


def test_postprocessor_applies_all_global_offsets_and_task_order():
    predictions = [
        make_prediction(len(task), preferred_class=min(1, len(task) - 1))
        for task in TASKS
    ]

    detections = make_postprocessor()(predictions)[0]

    assert detections.boxes.shape == (6, 9)
    assert detections.labels.tolist() == [0, 2, 4, 5, 7, 9]
    assert detections.scores.shape == (6,)


def test_postprocessor_suppresses_within_but_not_across_tasks():
    predictions = [make_prediction(len(task), width=2) for task in TASKS]

    detections = make_postprocessor()(predictions)[0]

    assert detections.boxes.shape[0] == 6
    assert detections.labels.tolist() == [0, 1, 3, 5, 6, 8]


def test_postprocessor_injects_nms_and_does_not_apply_global_sort_or_cap():
    calls = []

    def retain_all(boxes, scores, **kwargs):
        calls.append((boxes.clone(), dict(kwargs)))
        return torch.arange(boxes.shape[0], dtype=torch.int64)

    predictions = [make_prediction(len(task)) for task in TASKS]
    predictions[0]["hm"][:] = 1.0
    predictions[1]["hm"][:] = 5.0

    detections = make_postprocessor(nms=retain_all, post_max_size=1)(predictions)[0]

    assert len(calls) == 6
    assert all(call[0].shape == (1, 7) for call in calls)
    assert all(call[1]["post_max_size"] == 1 for call in calls)
    assert detections.labels.tolist() == [0, 1, 3, 5, 6, 8]
    assert detections.scores[0] < detections.scores[1]


def test_postprocessor_preserves_empty_shapes():
    predictions = [make_prediction(len(task)) for task in TASKS]
    for prediction in predictions:
        prediction["hm"].fill_(-100)

    detections = make_postprocessor()(predictions)[0]

    assert detections.boxes.shape == (0, 9)
    assert detections.scores.shape == (0,)
    assert detections.labels.shape == (0,)


def test_postprocessor_validates_task_channels_and_casts_nms_inputs():
    calls = []

    def retain_all(boxes, scores, **kwargs):
        calls.append((boxes.dtype, scores.dtype))
        return torch.arange(boxes.shape[0], dtype=torch.int64)

    class HalfDecoder:
        def __call__(self, prediction):
            return [
                DetectionCandidates(
                    boxes=torch.zeros((1, 9), dtype=torch.float16),
                    scores=torch.ones((1,), dtype=torch.float16),
                    labels=torch.zeros((1,), dtype=torch.int64),
                )
            ]

    predictions = [make_prediction(len(task)) for task in TASKS]
    postprocessor = CenterPointPostprocessor(
        HalfDecoder(),
        retain_all,
        TASKS,
        iou_threshold=0.2,
        pre_max_size=1000,
        post_max_size=83,
    )
    postprocessor(predictions)
    assert calls == [(torch.float32, torch.float32)] * 6

    predictions[0] = make_prediction(2)
    with pytest.raises(ValueError, match="channels"):
        make_postprocessor(nms=retain_all)(predictions)
