# Frozen Reproduction Baseline

This document fixes the first implementation and reproduction target. Changes to this target
must be reviewed as a new baseline rather than silently incorporated into the existing one.

## Target

| Item | Frozen value |
| --- | --- |
| Dataset | nuScenes `v1.0-trainval` |
| Task | 3D object detection on the validation split |
| Model | One-stage CenterPoint VoxelNet |
| Input | 10 LiDAR sweeps, five features: `x, y, z, intensity, time_lag` |
| Official repository commit | [`3cf7d870537e287c99b43b68636ea392a5e6f519`](https://github.com/tianweiy/CenterPoint/tree/3cf7d870537e287c99b43b68636ea392a5e6f519) |
| Official configuration | [`nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py) |
| Configuration Git blob | `a4b8db5ae623909d7ffe9258b457e319d8d8e3c6` |
| Official model-zoo reference | [`configs/nusc/README.md`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/configs/nusc/README.md), blob `9f57ee6ca1e6ad81b16d07c92fd5019acf756cf4` |
| Reported validation result | 59.6 mAP, 66.8 NDS |

The benchmark values above are reference targets, not results produced by this repository.
Reproduction will be claimed only with a published configuration, environment, checkpoint,
evaluation output, and training log.

## Geometry And Heads

- Point-cloud range: `[-54, -54, -5, 54, 54, 3]` metres.
- Voxel size: `[0.075, 0.075, 0.2]` metres.
- Sparse grid size: `[1440, 1440, 40]` in `x, y, z` order.
- BEV output stride: 8, producing a `180 x 180` prediction map.
- Maximum points per voxel: 10.
- Maximum voxels: 120,000 during training and 160,000 during evaluation.
- Six detection tasks cover the ten nuScenes detection classes.
- Each task predicts heatmap, center offset, height, dimensions, velocity, and sine/cosine yaw.
- Canonical inference uses task-wise rotated NMS without double-flip testing.

## Scope

The first reproduction includes the detector, preprocessing, training, decoding, nuScenes export,
and official detection evaluation required by the frozen configuration.

The following are explicitly separate targets:

- The paper-era 0.10 m voxel configuration
- Optional two-stage box refinement
- Tracking
- Double-flip inference
- DCN heads
- PointPillars and virtual-point variants
- Waymo training and evaluation

## Fidelity Rule

The paper defines the method, but the pinned repository commit and configuration define exact
behavior for this reproduction. Any deliberate difference must be documented in `docs/analysis/`
and supported by parity or experiment evidence before it is merged into `main`.
