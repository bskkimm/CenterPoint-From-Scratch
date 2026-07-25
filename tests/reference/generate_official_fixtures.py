"""Generate small golden fixtures from a pinned official CenterPoint checkout.

Usage:
    python tests/reference/generate_official_fixtures.py /path/to/CenterPoint

The loader isolates the relevant source modules so the historical training framework and CUDA
extensions do not need to be installed. It still executes the functions and class methods from
the pinned official files rather than reimplementing them here.
"""

import argparse
import ast
from collections import defaultdict
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch


OFFICIAL_COMMIT = "3cf7d870537e287c99b43b68636ea392a5e6f519"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def load_center_utils(root):
    for package in ("det3d", "det3d.core", "det3d.core.utils"):
        install_package(package)
    circle = types.ModuleType("det3d.core.utils.circle_nms_jit")
    circle.circle_nms = lambda boxes, thresh: []
    sys.modules[circle.__name__] = circle
    return load_module(
        "det3d.core.utils.center_utils",
        root / "det3d/core/utils/center_utils.py",
    )


def load_losses(root, center_utils):
    for package in ("det3d.models", "det3d.models.losses"):
        install_package(package)
    sys.modules["det3d.core.utils.center_utils"] = center_utils
    return load_module(
        "det3d.models.losses.centernet_loss",
        root / "det3d/models/losses/centernet_loss.py",
    )


class DummyRegistry:
    def register_module(self, value):
        return value


def load_voxel_encoder(root):
    install_package("det3d.models.readers")
    registry = types.ModuleType("det3d.models.registry")
    registry.READERS = DummyRegistry()
    sys.modules[registry.__name__] = registry
    return load_module(
        "det3d.models.readers.voxel_encoder",
        root / "det3d/models/readers/voxel_encoder.py",
    )


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def load_assign_label(root, center_utils):
    source_path = root / "det3d/datasets/pipelines/preprocess.py"
    tree = ast.parse(source_path.read_text())
    selected = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in {"flatten", "merge_multi_group_label"})
        or (isinstance(node, ast.ClassDef) and node.name == "AssignLabel")
    ]
    module = ast.Module(body=selected, type_ignores=[])

    class BoxOps:
        @staticmethod
        def limit_period(values, offset=0.5, period=2 * np.pi):
            return values - np.floor(values / period + offset) * period

    namespace = {
        "PIPELINES": DummyRegistry(),
        "box_np_ops": BoxOps,
        "draw_umich_gaussian": center_utils.draw_umich_gaussian,
        "gaussian_radius": center_utils.gaussian_radius,
        "np": np,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["AssignLabel"]


def load_box_geometry(root):
    source_path = root / "det3d/core/bbox/box_np_ops.py"
    tree = ast.parse(source_path.read_text())
    names = {"corners_nd", "rotation_3d_in_axis", "center_to_corner_box3d"}
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["center_to_corner_box3d"]


def load_official_predict(root):
    source_path = root / "det3d/models/bbox_heads/center_head.py"
    tree = ast.parse(source_path.read_text())
    center_head = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CenterHead"
    )
    predict = next(
        node
        for node in center_head.body
        if isinstance(node, ast.FunctionDef) and node.name == "predict"
    )
    namespace = {"defaultdict": defaultdict, "torch": torch}
    exec(
        compile(ast.Module(body=[predict], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["predict"]


def tensor_list(value):
    return value.detach().cpu().tolist()


def generate(root):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"expected {OFFICIAL_COMMIT}, found {commit}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if dirty:
        raise RuntimeError("official checkout must have a clean working tree")

    center_utils = load_center_utils(root)
    loss_module = load_losses(root, center_utils)
    voxel_module = load_module(
        "official_point_cloud_ops",
        root / "det3d/ops/point_cloud/point_cloud_ops.py",
    )
    reader_module = load_voxel_encoder(root)
    assign_label = load_assign_label(root, center_utils)
    center_to_corner_box3d = load_box_geometry(root)
    official_predict = load_official_predict(root)

    heatmap = np.zeros((5, 5), dtype=np.float32)
    center_utils.draw_umich_gaussian(heatmap, np.array([0.0, 2.0]), 1)

    points = np.array(
        [
            [1.1, 0.1, 1.1, 10.0],
            [0.1, 1.1, 0.1, 20.0],
            [1.2, 0.2, 1.2, 30.0],
            [2.0, 1.0, 1.0, 40.0],
            [-0.1, 0.0, 0.0, 50.0],
        ],
        dtype=np.float32,
    )
    voxels, coordinates, num_points = voxel_module.points_to_voxel(
        points,
        voxel_size=[1.0, 1.0, 1.0],
        coors_range=[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        max_points=2,
        max_voxels=3,
    )
    encoded = reader_module.VoxelFeatureExtractorV3(4)(
        torch.from_numpy(voxels), torch.from_numpy(num_points)
    )

    probabilities = torch.tensor([[[[0.2, 0.7]], [[0.4, 0.1]]]])
    target_heatmap = torch.tensor([[[[0.0, 1.0]], [[0.5, 0.0]]]])
    indices = torch.tensor([[1]])
    categories = torch.tensor([[0]])
    mask = torch.tensor([[1]], dtype=torch.uint8)
    focal = loss_module.FastFocalLoss()(
        probabilities, target_heatmap, indices, mask, categories
    )

    regression_prediction = torch.tensor(
        [[[[1.0, 5.0]], [[2.0, 8.0]], [[3.0, 9.0]]]]
    )
    regression_target = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]]])
    regression = loss_module.RegLoss()(
        regression_prediction,
        torch.tensor([[1, 1]], dtype=torch.uint8),
        torch.tensor([[0, 1]]),
        regression_target,
    )

    config = AttrDict(
        out_size_factor=1,
        target_assigner=AttrDict(tasks=[AttrDict(num_class=1, class_names=["car"])]),
        gaussian_overlap=0.1,
        max_objs=5,
        min_radius=0,
        pc_range=[0.0, 0.0, -5.0, 4.0, 4.0, 5.0],
        voxel_size=[1.0, 1.0, 1.0],
    )
    boxes = np.array(
        [
            [1.25, 2.75, 0.5, 2.0, 1.0, 1.5, 3.0, -2.0, np.pi / 2],
            [2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [-0.2, 1.8, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    result, _ = assign_label(cfg=config)(
        {
            "mode": "train",
            "type": "NuScenesDataset",
            "lidar": {
                "annotations": {
                    "gt_boxes": boxes,
                    "gt_classes": np.ones((3,), dtype=np.int64),
                    "gt_names": np.array(["car", "car", "car"]),
                }
            },
        },
        {},
    )
    target = result["lidar"]["targets"]

    corners = center_to_corner_box3d(
        np.array([[10.0, 20.0, 30.0]], dtype=np.float32),
        np.array([[2.0, 4.0, 6.0]], dtype=np.float32),
        np.array([np.pi / 2], dtype=np.float32),
    )

    class CaptureDecoder:
        num_classes = [2]

        @staticmethod
        def post_processing(batch_boxes, batch_heatmap, *_args):
            outputs = []
            for boxes_for_sample, heatmap_for_sample in zip(batch_boxes, batch_heatmap):
                scores, labels = torch.max(heatmap_for_sample, dim=-1)
                outputs.append(
                    {
                        "box3d_lidar": boxes_for_sample,
                        "scores": scores,
                        "label_preds": labels,
                    }
                )
            return outputs

    CaptureDecoder.predict = official_predict
    decoder_predictions = {
        "hm": torch.tensor([[[[0.0, 1.0]], [[2.0, -1.0]]]]),
        "reg": torch.tensor([[[[0.25, 0.5]], [[0.75, 0.0]]]]),
        "height": torch.tensor([[[[1.5, -2.0]]]]),
        "dim": torch.tensor(
            [
                [
                    [[np.log(2.0), np.log(3.0)]],
                    [[np.log(4.0), np.log(5.0)]],
                    [[np.log(6.0), np.log(7.0)]],
                ]
            ],
            dtype=torch.float32,
        ),
        "vel": torch.tensor([[[[8.0, 9.0]], [[10.0, 11.0]]]]),
        "rot": torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]]),
    }
    decoded = CaptureDecoder().predict(
        {"metadata": []},
        [decoder_predictions],
        AttrDict(
            double_flip=False,
            post_center_limit_range=[-100, -100, -100, 100, 100, 100],
            out_size_factor=2,
            voxel_size=[0.5, 1.0],
            pc_range=[-10.0, -20.0],
            score_threshold=0.0,
            per_class_nms=False,
        ),
    )[0]

    return {
        "metadata": {
            "official_commit": commit,
            "sources": [
                "det3d/core/utils/center_utils.py",
                "det3d/ops/point_cloud/point_cloud_ops.py",
                "det3d/models/readers/voxel_encoder.py",
                "det3d/models/losses/centernet_loss.py",
                "det3d/datasets/pipelines/preprocess.py:AssignLabel",
                "det3d/core/bbox/box_np_ops.py:center_to_corner_box3d",
                "det3d/models/bbox_heads/center_head.py:CenterHead.predict",
            ],
        },
        "geometry": {"corners": corners.tolist()},
        "gaussian": {
            "radius": float(center_utils.gaussian_radius((2.5, 1.25), 0.1)),
            "heatmap": heatmap.tolist(),
        },
        "voxelization": {
            "voxels": voxels.tolist(),
            "coordinates": coordinates.tolist(),
            "num_points": num_points.tolist(),
            "encoded": tensor_list(encoded),
        },
        "losses": {
            "focal": float(focal),
            "regression": tensor_list(regression),
        },
        "target": {
            "heatmap": target["hm"][0].tolist(),
            "annotation": target["anno_box"][0].tolist(),
            "indices": target["ind"][0].tolist(),
            "mask": target["mask"][0].tolist(),
            "categories": target["cat"][0].tolist(),
        },
        "decoder": {
            "boxes": tensor_list(decoded["box3d_lidar"]),
            "scores": tensor_list(decoded["scores"]),
            "labels": tensor_list(decoded["label_preds"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("official_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.official_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
