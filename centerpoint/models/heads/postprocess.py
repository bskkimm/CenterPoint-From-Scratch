"""Canonical task-wise NMS and six-task result merging."""

from dataclasses import dataclass
from typing import Callable, List, Mapping, Sequence, Union

import torch
from torch import Tensor

from centerpoint.contracts import TaskPredictions
from centerpoint.models.heads.decoder import CenterPointDecoder, DetectionCandidates


Prediction = Union[Mapping[str, Tensor], TaskPredictions]
NMSCallable = Callable[..., Tensor]


@dataclass(frozen=True)
class Detections:
    """Merged zero-based global detections for one sample."""

    boxes: Tensor
    scores: Tensor
    labels: Tensor


class CenterPointPostprocessor:
    """Decode, suppress, and merge the six task outputs in task order."""

    def __init__(
        self,
        decoder: CenterPointDecoder,
        rotated_nms: NMSCallable,
        tasks: Sequence[Sequence[str]],
        *,
        iou_threshold: float,
        pre_max_size: int,
        post_max_size: int,
    ) -> None:
        if not tasks or any(not task for task in tasks):
            raise ValueError("tasks must contain at least one class each")
        if iou_threshold < 0 or pre_max_size <= 0 or post_max_size <= 0:
            raise ValueError("NMS threshold and limits are invalid")
        self.decoder = decoder
        self.rotated_nms = rotated_nms
        self.tasks = tuple(tuple(task) for task in tasks)
        self.iou_threshold = float(iou_threshold)
        self.pre_max_size = pre_max_size
        self.post_max_size = post_max_size

    @torch.no_grad()
    def __call__(self, predictions: Sequence[Prediction]) -> List[Detections]:
        if len(predictions) != len(self.tasks):
            raise ValueError("predictions must match the configured task count")

        decoded = [self.decoder(prediction) for prediction in predictions]
        batch_size = len(decoded[0])
        if any(len(task_results) != batch_size for task_results in decoded):
            raise ValueError("all task predictions must have the same batch size")

        results = []
        for batch_index in range(batch_size):
            boxes_by_task = []
            scores_by_task = []
            labels_by_task = []
            label_offset = 0
            for task, task_results in zip(self.tasks, decoded):
                candidates = task_results[batch_index]
                if candidates.boxes.ndim != 2 or candidates.boxes.shape[1] != 9:
                    raise ValueError("canonical postprocessing requires velocity boxes shaped [N, 9]")
                keep = self.rotated_nms(
                    _nms_boxes(candidates),
                    candidates.scores,
                    iou_threshold=self.iou_threshold,
                    pre_max_size=self.pre_max_size,
                    post_max_size=self.post_max_size,
                )
                boxes_by_task.append(candidates.boxes[keep])
                scores_by_task.append(candidates.scores[keep])
                labels_by_task.append(candidates.labels[keep] + label_offset)
                label_offset += len(task)

            results.append(
                Detections(
                    boxes=torch.cat(boxes_by_task, dim=0),
                    scores=torch.cat(scores_by_task, dim=0),
                    labels=torch.cat(labels_by_task, dim=0),
                )
            )
        return results


def _nms_boxes(candidates: DetectionCandidates) -> Tensor:
    return torch.cat((candidates.boxes[:, :6], candidates.boxes[:, 8:9]), dim=1)
