"""Correctness oracles and explicit production backend boundaries."""

from .rotated_nms import pairwise_rotated_bev_iou, rotated_nms

__all__ = ["pairwise_rotated_bev_iou", "rotated_nms"]
