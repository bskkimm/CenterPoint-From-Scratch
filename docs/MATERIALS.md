# Minimal Implementation Materials

These are the minimum primary sources for implementing CenterPoint and reproducing its official
nuScenes and Waymo workflows. Use the paper for intent, the official code and configs for exact
behavior, and the dataset devkits for evaluation contracts.

## Core Model

| Material | Use |
| --- | --- |
| [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) | Defines the center representation, prediction heads, optional second stage, and tracking formulation. Start here to establish the intended model. |
| [Official CenterPoint repository](https://github.com/tianweiy/CenterPoint) | Implementation ground truth for modules, model zoo, configurations, supported datasets, and reproduction workflow. |

## Dataset Workflows

| Material | Use |
| --- | --- |
| [CenterPoint nuScenes guide](https://github.com/tianweiy/CenterPoint/blob/master/docs/NUSC.md) | Defines data layout, preprocessing, info files, ground-truth database generation, sweep settings, training, evaluation, and tracking commands. |
| [CenterPoint Waymo guide](https://github.com/tianweiy/CenterPoint/blob/master/docs/WAYMO.md) | Defines TFRecord-to-pickle conversion, info generation, training, evaluation, and submission packaging. |

## Canonical Configurations

| Material | Use |
| --- | --- |
| [nuScenes voxel configuration](https://github.com/tianweiy/CenterPoint/blob/master/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py) | Source for exact voxelization, model heads, loss weights, augmentation, optimizer, schedule, and inference settings for the canonical nuScenes voxel model. |
| [Waymo voxel configuration](https://github.com/tianweiy/CenterPoint/blob/master/configs/waymo/voxelnet/waymo_centerpoint_voxelnet_1x.py) | Source for the official one-sweep Waymo baseline, including classes, voxelization, optimization, and inference thresholds. |

## Evaluation Contracts

| Material | Use |
| --- | --- |
| [nuScenes detection evaluation reference](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md) | Defines result JSON requirements and official metrics: mAP, NDS, mATE, mASE, mAOE, mAVE, and mAAE. |
| [Waymo Open Dataset evaluation code](https://github.com/waymo-research/waymo-open-dataset) | Provides official detection and tracking metrics, evaluation tools, and the final Waymo-compatible binary evaluation/submission flow. |

## Recommended Order

1. Read the paper and identify the core detector, optional refinement stage, and tracking stage.
2. Trace those components in the official repository.
3. Freeze one dataset configuration as the first implementation target.
4. Reproduce its preprocessing contract before model training.
5. Validate decoded predictions against the official dataset evaluation format.
6. Record any deliberate difference from the official config or code in `docs/analysis/`.
