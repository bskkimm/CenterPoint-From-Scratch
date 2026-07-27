# Tensor And Coordinate Contracts

These contracts apply to the frozen one-stage nuScenes baseline. Shapes use `B` for batch size,
`N` for points, `M` for voxels or objects, `P` for points per voxel, `C` for channels, and `H/W`
for spatial dimensions.

## Coordinates And Units

- Distances, box dimensions, and velocities use metres and metres per second.
- Point and box coordinates are expressed in the reference LiDAR frame unless explicitly marked
  as ego or global coordinates.
- The LiDAR axes are `x` forward, `y` left, and `z` up.
- Internal positive yaw follows the official CenterPoint clockwise row-vector rotation convention.
- nuScenes yaw and internal yaw convert in both directions as `-yaw - pi/2`.

## Points And Sweeps

| Tensor | Shape | Order | Dtype |
| --- | --- | --- | --- |
| Current points | `[N, 4]` | `[x, y, z, intensity]` | `float32` |
| Aggregated points | `[N, 5]` | `[x, y, z, intensity, time_lag]` | `float32` |

Historical sweeps are transformed into the current reference-LiDAR frame before concatenation.
The current sweep has zero time lag.

## Hard Voxels

| Tensor | Shape | Contract |
| --- | --- | --- |
| `voxels` | `[M, P, 5]` | Zero-padded points in input encounter order |
| `num_points` | `[M]` | Number of retained points in each voxel |
| `coordinates` | `[M, 3]` | Integer `[z, y, x]` coordinates in first-occurrence order |
| Batched coordinates | `[M, 4]` | Integer `[batch, z, y, x]` coordinates |
| Mean VFE output | `[M, 5]` | Mean of every input feature over valid points |

The lower point-cloud boundary is included and the effective upper boundary is excluded. The first
10 points per voxel and first 120,000/160,000 train/evaluation voxels are retained.

## Boxes

The canonical internal box layout is:

```text
[x, y, z, w, l, h, vx, vy, yaw]
```

`x, y, z` are geometric box centers. Dimensions remain in nuScenes `w, l, h` order. Temporary
NMS input selects `[x, y, z, w, l, h, yaw]` and the PCDet-compatible adapter converts it to
`[x, y, z, l, w, h, -yaw-pi/2]`.

## Tasks And Labels

Global 1-based dataset classes are grouped into six ordered tasks:

1. `car`
2. `truck`, `construction_vehicle`
3. `bus`, `trailer`
4. `barrier`
5. `motorcycle`, `bicycle`
6. `pedestrian`, `traffic_cone`

Targets and task predictions use zero-based task-local labels. Postprocessing adds cumulative task
class offsets to produce zero-based global prediction labels.

## Center Targets

For each task:

| Tensor | Shape | Dtype |
| --- | --- | --- |
| Heatmap | `[num_task_classes, 180, 180]` | `float32` |
| Annotation | `[500, 10]` | `float32` |
| Flat index | `[500]` | `int64` |
| Valid mask | `[500]` | `uint8` |
| Local category | `[500]` | `int64` |

The annotation order is fixed:

```text
[dx, dy, z, log(w), log(l), log(h), vx, vy, sin(yaw), cos(yaw)]
```

The flat index is `y * feature_width + x`. Invalid objects do not compact later target slots, and
objects sharing a center retain separate regression records.

## CenterHead Outputs

Every task produces NCHW maps:

| Head | Shape |
| --- | --- |
| `hm` | `[B, num_task_classes, H, W]` |
| `reg` | `[B, 2, H, W]` |
| `height` | `[B, 1, H, W]` |
| `dim` | `[B, 3, H, W]` |
| `vel` | `[B, 2, H, W]` |
| `rot` | `[B, 2, H, W]` as `[sin, cos]` |

The frozen model uses `H=W=180`. Decoding exponentiates dimensions, computes yaw with
`atan2(sin, cos)`, and emits `[x, y, z, w, l, h, vx, vy, yaw]`.

## Dense Decoding

- Decode all `H*W` cells; do not add top-K selection or local-maximum pooling.
- Keep only the highest-scoring class at each cell.
- Apply score filtering with strict `>`.
- Apply post-center range filtering inclusively on `x, y, z`.
- Apply rotated NMS independently per task before adding global class offsets.

## Sparse And BEV Features

The production sparse adapter accepts mean VFE features and `[batch,z,y,x]` coordinates. The
pinned sparse backbone must return a dense `[B,256,180,180]` BEV tensor to the neck. The official
neck concatenates two branches into `[B,512,180,180]` for CenterHead.
