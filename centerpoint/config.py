"""Frozen local representation of the canonical nuScenes CenterPoint recipe."""

from dataclasses import dataclass
from typing import Tuple


OFFICIAL_COMMIT = "3cf7d870537e287c99b43b68636ea392a5e6f519"
OFFICIAL_CONFIG_BLOB = "a4b8db5ae623909d7ffe9258b457e319d8d8e3c6"


@dataclass(frozen=True)
class VoxelConfig:
    point_cloud_range: Tuple[float, float, float, float, float, float]
    size: Tuple[float, float, float]
    max_points: int
    max_voxels: Tuple[int, int]

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        return tuple(
            round((self.point_cloud_range[index + 3] - self.point_cloud_range[index]) / size)
            for index, size in enumerate(self.size)
        )


@dataclass(frozen=True)
class TargetConfig:
    output_stride: int
    gaussian_overlap: float
    max_objects: int
    min_radius: int


@dataclass(frozen=True)
class NeckConfig:
    layer_numbers: Tuple[int, int]
    downsample_strides: Tuple[int, int]
    downsample_filters: Tuple[int, int]
    upsample_strides: Tuple[int, int]
    upsample_filters: Tuple[int, int]
    input_channels: int


@dataclass(frozen=True)
class HeadBranch:
    name: str
    output_channels: int
    num_convolutions: int


@dataclass(frozen=True)
class HeadConfig:
    input_channels: int
    shared_channels: int
    branches: Tuple[HeadBranch, ...]
    loss_weight: float
    code_weights: Tuple[float, ...]
    use_dcn: bool


@dataclass(frozen=True)
class ModelConfig:
    reader: str
    num_input_features: int
    backbone: str
    backbone_downsample_factor: int
    neck: NeckConfig
    head: HeadConfig


@dataclass(frozen=True)
class AugmentationConfig:
    shuffle_points: bool
    rotation_range: Tuple[float, float]
    scale_range: Tuple[float, float]
    translation_std: float
    database_sampling_enabled: bool


@dataclass(frozen=True)
class NMSConfig:
    rotated: bool
    multi_class: bool
    pre_max_size: int
    post_max_size: int
    iou_threshold: float


@dataclass(frozen=True)
class InferenceConfig:
    post_center_range: Tuple[float, float, float, float, float, float]
    max_per_image: int
    score_threshold: float
    output_stride: int
    voxel_size_xy: Tuple[float, float]
    point_cloud_min_xy: Tuple[float, float]
    nms: NMSConfig


@dataclass(frozen=True)
class TrainingConfig:
    samples_per_gpu: int
    workers_per_gpu: int
    optimizer: str
    amsgrad: bool
    weight_decay: float
    fixed_weight_decay: bool
    moving_average: bool
    max_gradient_norm: float
    gradient_norm_type: float
    schedule: str
    max_learning_rate: float
    momentums: Tuple[float, float]
    division_factor: float
    warmup_fraction: float
    epochs: int


@dataclass(frozen=True)
class CenterPointConfig:
    official_commit: str
    official_config_blob: str
    dataset: str
    num_sweeps: int
    point_features: Tuple[str, ...]
    tasks: Tuple[Tuple[str, ...], ...]
    voxel: VoxelConfig
    target: TargetConfig
    model: ModelConfig
    augmentation: AugmentationConfig
    inference: InferenceConfig
    training: TrainingConfig

    def to_official_manifest(self):
        """Return the local values represented directly by the pinned official config."""

        return {
            "dataset": self.dataset,
            "num_sweeps": self.num_sweeps,
            "tasks": [list(task) for task in self.tasks],
            "voxel": {
                "point_cloud_range": list(self.voxel.point_cloud_range),
                "size": list(self.voxel.size),
                "max_points": self.voxel.max_points,
                "max_voxels": list(self.voxel.max_voxels),
            },
            "target": {
                "output_stride": self.target.output_stride,
                "gaussian_overlap": self.target.gaussian_overlap,
                "max_objects": self.target.max_objects,
                "min_radius": self.target.min_radius,
            },
            "model": {
                "reader": self.model.reader,
                "num_input_features": self.model.num_input_features,
                "backbone": self.model.backbone,
                "backbone_downsample_factor": self.model.backbone_downsample_factor,
                "neck": {
                    "layer_numbers": list(self.model.neck.layer_numbers),
                    "downsample_strides": list(self.model.neck.downsample_strides),
                    "downsample_filters": list(self.model.neck.downsample_filters),
                    "upsample_strides": list(self.model.neck.upsample_strides),
                    "upsample_filters": list(self.model.neck.upsample_filters),
                    "input_channels": self.model.neck.input_channels,
                },
                "head": {
                    "input_channels": self.model.head.input_channels,
                    "shared_channels": self.model.head.shared_channels,
                    "branches": {
                        branch.name: [branch.output_channels, branch.num_convolutions]
                        for branch in self.model.head.branches
                    },
                    "loss_weight": self.model.head.loss_weight,
                    "code_weights": list(self.model.head.code_weights),
                    "use_dcn": self.model.head.use_dcn,
                },
            },
            "augmentation": {
                "shuffle_points": self.augmentation.shuffle_points,
                "rotation_range": list(self.augmentation.rotation_range),
                "scale_range": list(self.augmentation.scale_range),
                "translation_std": self.augmentation.translation_std,
                "database_sampling_enabled": self.augmentation.database_sampling_enabled,
            },
            "inference": {
                "post_center_range": list(self.inference.post_center_range),
                "max_per_image": self.inference.max_per_image,
                "score_threshold": self.inference.score_threshold,
                "output_stride": self.inference.output_stride,
                "voxel_size_xy": list(self.inference.voxel_size_xy),
                "point_cloud_min_xy": list(self.inference.point_cloud_min_xy),
                "nms": {
                    "rotated": self.inference.nms.rotated,
                    "multi_class": self.inference.nms.multi_class,
                    "pre_max_size": self.inference.nms.pre_max_size,
                    "post_max_size": self.inference.nms.post_max_size,
                    "iou_threshold": self.inference.nms.iou_threshold,
                },
            },
            "training": {
                "samples_per_gpu": self.training.samples_per_gpu,
                "workers_per_gpu": self.training.workers_per_gpu,
                "optimizer": self.training.optimizer,
                "amsgrad": self.training.amsgrad,
                "weight_decay": self.training.weight_decay,
                "fixed_weight_decay": self.training.fixed_weight_decay,
                "moving_average": self.training.moving_average,
                "max_gradient_norm": self.training.max_gradient_norm,
                "gradient_norm_type": self.training.gradient_norm_type,
                "schedule": self.training.schedule,
                "max_learning_rate": self.training.max_learning_rate,
                "momentums": list(self.training.momentums),
                "division_factor": self.training.division_factor,
                "warmup_fraction": self.training.warmup_fraction,
                "epochs": self.training.epochs,
            },
        }

    def make_voxelizer(self, training: bool):
        """Build the order-preserving reference voxelizer for this recipe."""

        from centerpoint.data.voxelization import HardVoxelizer

        return HardVoxelizer(
            voxel_size=self.voxel.size,
            point_cloud_range=self.voxel.point_cloud_range,
            max_points_per_voxel=self.voxel.max_points,
            max_voxels=self.voxel.max_voxels[0 if training else 1],
        )

    def make_target_assigner(self):
        """Build the canonical six-task target assigner."""

        from centerpoint.data.targets import CenterTargetAssigner

        return CenterTargetAssigner(
            tasks=self.tasks,
            voxel_size=self.voxel.size,
            point_cloud_range=self.voxel.point_cloud_range,
            output_stride=self.target.output_stride,
            gaussian_overlap=self.target.gaussian_overlap,
            max_objects=self.target.max_objects,
            min_radius=self.target.min_radius,
        )

    def make_decoder(self):
        """Build the canonical dense decoder before rotated NMS."""

        from centerpoint.models.heads.decoder import CenterPointDecoder

        return CenterPointDecoder(
            voxel_size=(*self.inference.voxel_size_xy, self.voxel.size[2]),
            point_cloud_range=(
                *self.inference.point_cloud_min_xy,
                self.voxel.point_cloud_range[2],
                *self.voxel.point_cloud_range[3:],
            ),
            output_stride=self.inference.output_stride,
            score_threshold=self.inference.score_threshold,
            post_center_range=self.inference.post_center_range,
        )

    def make_voxelnet(self, backbone, postprocessor):
        """Build the frozen detector around caller-provided sparse and NMS backends."""

        from centerpoint.models import (
            CenterHead,
            MeanVoxelFeatureEncoder,
            RPN,
            SparseBackbone,
            VoxelNet,
        )

        if not isinstance(backbone, SparseBackbone):
            raise TypeError("backbone must implement SparseBackbone")
        expected_contract = {
            "input_channels": self.model.num_input_features,
            "output_channels": self.model.neck.input_channels,
            "output_stride": self.model.backbone_downsample_factor,
        }
        for field, expected in expected_contract.items():
            if getattr(backbone, field) != expected:
                raise ValueError(
                    f"backbone {field} must be {expected} for the frozen VoxelNet recipe"
                )

        return VoxelNet(
            reader=MeanVoxelFeatureEncoder(self.model.num_input_features),
            backbone=backbone,
            neck=RPN(
                layer_nums=self.model.neck.layer_numbers,
                ds_layer_strides=self.model.neck.downsample_strides,
                ds_num_filters=self.model.neck.downsample_filters,
                us_layer_strides=self.model.neck.upsample_strides,
                us_num_filters=self.model.neck.upsample_filters,
                num_input_features=self.model.neck.input_channels,
            ),
            bbox_head=CenterHead(
                in_channels=self.model.head.input_channels,
                tasks=self.tasks,
                common_heads={
                    branch.name: (branch.output_channels, branch.num_convolutions)
                    for branch in self.model.head.branches
                },
                share_conv_channel=self.model.head.shared_channels,
                loss_weight=self.model.head.loss_weight,
                code_weights=self.model.head.code_weights,
            ),
            postprocessor=postprocessor,
            spatial_shape=tuple(reversed(self.voxel.grid_size)),
        )


NUSCENES_VOXELNET_075 = CenterPointConfig(
    official_commit=OFFICIAL_COMMIT,
    official_config_blob=OFFICIAL_CONFIG_BLOB,
    dataset="NuScenesDataset",
    num_sweeps=10,
    point_features=("x", "y", "z", "intensity", "time_lag"),
    tasks=(
        ("car",),
        ("truck", "construction_vehicle"),
        ("bus", "trailer"),
        ("barrier",),
        ("motorcycle", "bicycle"),
        ("pedestrian", "traffic_cone"),
    ),
    voxel=VoxelConfig(
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        size=(0.075, 0.075, 0.2),
        max_points=10,
        max_voxels=(120_000, 160_000),
    ),
    target=TargetConfig(
        output_stride=8,
        gaussian_overlap=0.1,
        max_objects=500,
        min_radius=2,
    ),
    model=ModelConfig(
        reader="VoxelFeatureExtractorV3",
        num_input_features=5,
        backbone="SpMiddleResNetFHD",
        backbone_downsample_factor=8,
        neck=NeckConfig(
            layer_numbers=(5, 5),
            downsample_strides=(1, 2),
            downsample_filters=(128, 256),
            upsample_strides=(1, 2),
            upsample_filters=(256, 256),
            input_channels=256,
        ),
        head=HeadConfig(
            input_channels=512,
            shared_channels=64,
            branches=(
                HeadBranch("reg", 2, 2),
                HeadBranch("height", 1, 2),
                HeadBranch("dim", 3, 2),
                HeadBranch("rot", 2, 2),
                HeadBranch("vel", 2, 2),
            ),
            loss_weight=0.25,
            code_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0),
            use_dcn=False,
        ),
    ),
    augmentation=AugmentationConfig(
        shuffle_points=True,
        rotation_range=(-0.78539816, 0.78539816),
        scale_range=(0.9, 1.1),
        translation_std=0.5,
        database_sampling_enabled=False,
    ),
    inference=InferenceConfig(
        post_center_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
        max_per_image=500,
        score_threshold=0.1,
        output_stride=8,
        voxel_size_xy=(0.075, 0.075),
        point_cloud_min_xy=(-54.0, -54.0),
        nms=NMSConfig(
            rotated=True,
            multi_class=False,
            pre_max_size=1000,
            post_max_size=83,
            iou_threshold=0.2,
        ),
    ),
    training=TrainingConfig(
        samples_per_gpu=4,
        workers_per_gpu=6,
        optimizer="adam",
        amsgrad=False,
        weight_decay=0.01,
        fixed_weight_decay=True,
        moving_average=False,
        max_gradient_norm=35.0,
        gradient_norm_type=2.0,
        schedule="one_cycle",
        max_learning_rate=0.001,
        momentums=(0.95, 0.85),
        division_factor=10.0,
        warmup_fraction=0.4,
        epochs=20,
    ),
)
