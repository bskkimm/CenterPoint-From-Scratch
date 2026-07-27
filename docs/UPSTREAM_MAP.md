# Pinned Upstream Source Map

All links refer to official CenterPoint commit
[`3cf7d870`](https://github.com/tianweiy/CenterPoint/tree/3cf7d870537e287c99b43b68636ea392a5e6f519).

| Local responsibility | Official source | Status |
| --- | --- | --- |
| Frozen recipe | [`configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py) | Baseline pinned; local manifest pending |
| Gaussian heatmaps and gather | [`det3d/core/utils/center_utils.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/core/utils/center_utils.py) | Implemented and fixture-tested |
| Ordered hard voxelization | [`det3d/ops/point_cloud/point_cloud_ops.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/ops/point_cloud/point_cloud_ops.py) | Reference CPU implementation complete |
| Mean voxel encoder | [`det3d/models/readers/voxel_encoder.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/models/readers/voxel_encoder.py) | Implemented and fixture-tested |
| Target assignment | [`det3d/datasets/pipelines/preprocess.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/datasets/pipelines/preprocess.py) | Core assignment implemented and fixture-tested |
| Focal and regression losses | [`det3d/models/losses/centernet_loss.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/models/losses/centernet_loss.py) | Implemented and fixture-tested |
| Box corners and periodic angles | [`det3d/core/bbox/box_np_ops.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/core/bbox/box_np_ops.py) | Implemented and fixture-tested |
| CenterHead decode | [`det3d/models/bbox_heads/center_head.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/models/bbox_heads/center_head.py) | Pre-NMS decode implemented and fixture-tested |
| nuScenes sweeps and box conversion | [`det3d/datasets/nuscenes/nusc_common.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/datasets/nuscenes/nusc_common.py) | Pending |
| Point and sweep loading | [`det3d/datasets/pipelines/loading.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/datasets/pipelines/loading.py) | Pending |
| Augmentation and preprocessing | [`det3d/datasets/pipelines/preprocess.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/datasets/pipelines/preprocess.py) | Pending except target assignment |
| Batch collation | [`det3d/torchie/parallel/collate.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/torchie/parallel/collate.py) | Pending |
| Sparse backbone | [`det3d/models/backbones/scn.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/models/backbones/scn.py) | Pending backend decision and CUDA parity |
| BEV RPN neck | [`det3d/models/necks/rpn.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/models/necks/rpn.py) | Pending |
| CenterHead modules | [`det3d/models/bbox_heads/center_head.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/models/bbox_heads/center_head.py) | Pending except losses/decode |
| Detector assembly | [`det3d/models/detectors/voxelnet.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/models/detectors/voxelnet.py) | Pending |
| Rotated NMS adapter | [`det3d/core/bbox/box_torch_ops.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/core/bbox/box_torch_ops.py) | Pending production backend |
| nuScenes export/evaluation | [`det3d/datasets/nuscenes/nuscenes.py`](https://github.com/tianweiy/CenterPoint/blob/3cf7d870537e287c99b43b68636ea392a5e6f519/det3d/datasets/nuscenes/nuscenes.py) | Pending |

The map records semantic provenance, not permission to copy framework code wholesale. Local modules
should expose small PyTorch-oriented interfaces and preserve only behavior required by the baseline.
