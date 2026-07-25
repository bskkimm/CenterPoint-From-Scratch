import json
import math
from pathlib import Path

import torch

from centerpoint.data import CenterTargetAssigner, HardVoxelizer
from centerpoint.models.heads import CenterPointDecoder
from centerpoint.models.losses import FastFocalLoss, RegressionLoss
from centerpoint.models.readers import MeanVoxelFeatureEncoder
from centerpoint.utils.heatmap import draw_gaussian, gaussian_radius
from centerpoint.utils.geometry import boxes_to_corners_3d


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "official_3cf7d870.json"
OFFICIAL_COMMIT = "3cf7d870537e287c99b43b68636ea392a5e6f519"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text())


def test_fixture_is_tied_to_frozen_official_commit():
    fixture = load_fixture()

    assert fixture["metadata"]["official_commit"] == OFFICIAL_COMMIT
    assert len(fixture["metadata"]["sources"]) == 7


def test_box_corners_match_official_golden_output():
    fixture = load_fixture()["geometry"]

    corners = boxes_to_corners_3d(
        torch.tensor([[10.0, 20.0, 30.0]]),
        torch.tensor([[2.0, 4.0, 6.0]]),
        torch.tensor([math.pi / 2]),
    )

    torch.testing.assert_allclose(corners, torch.tensor(fixture["corners"]))


def test_heatmap_utilities_match_official_golden_output():
    fixture = load_fixture()["gaussian"]
    heatmap = torch.zeros((5, 5))

    draw_gaussian(heatmap, center=(0.0, 2.0), radius=1)

    assert gaussian_radius((2.5, 1.25), 0.1) == fixture["radius"]
    torch.testing.assert_allclose(heatmap, torch.tensor(fixture["heatmap"]))


def test_voxelization_and_vfe_match_official_golden_output():
    fixture = load_fixture()["voxelization"]
    points = torch.tensor(
        [
            [1.1, 0.1, 1.1, 10.0],
            [0.1, 1.1, 0.1, 20.0],
            [1.2, 0.2, 1.2, 30.0],
            [2.0, 1.0, 1.0, 40.0],
            [-0.1, 0.0, 0.0, 50.0],
        ]
    )
    result = HardVoxelizer((1, 1, 1), (0, 0, 0, 2, 2, 2), 2, 3)(points)
    encoded = MeanVoxelFeatureEncoder(4)(result.voxels, result.num_points)

    torch.testing.assert_allclose(result.voxels, torch.tensor(fixture["voxels"]))
    torch.testing.assert_allclose(
        result.coordinates, torch.tensor(fixture["coordinates"], dtype=torch.int32)
    )
    torch.testing.assert_allclose(
        result.num_points, torch.tensor(fixture["num_points"], dtype=torch.int32)
    )
    torch.testing.assert_allclose(encoded, torch.tensor(fixture["encoded"]))


def test_losses_match_official_golden_output():
    fixture = load_fixture()["losses"]
    focal = FastFocalLoss()(
        torch.tensor([[[[0.2, 0.7]], [[0.4, 0.1]]]]),
        torch.tensor([[[[0.0, 1.0]], [[0.5, 0.0]]]]),
        torch.tensor([[1]]),
        torch.tensor([[1]], dtype=torch.uint8),
        torch.tensor([[0]]),
    )
    regression = RegressionLoss()(
        torch.tensor([[[[1.0, 5.0]], [[2.0, 8.0]], [[3.0, 9.0]]]]),
        torch.tensor([[1, 1]], dtype=torch.uint8),
        torch.tensor([[0, 1]]),
        torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]]]),
    )

    torch.testing.assert_allclose(focal, torch.tensor(fixture["focal"]))
    torch.testing.assert_allclose(regression, torch.tensor(fixture["regression"]))


def test_target_assignment_matches_official_golden_output():
    fixture = load_fixture()["target"]
    boxes = torch.tensor(
        [
            [1.25, 2.75, 0.5, 2.0, 1.0, 1.5, 3.0, -2.0, math.pi / 2],
            [2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [-0.2, 1.8, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assigner = CenterTargetAssigner(
        tasks=(("car",),),
        voxel_size=(1, 1, 1),
        point_cloud_range=(0, 0, -5, 4, 4, 5),
        output_stride=1,
        gaussian_overlap=0.1,
        max_objects=5,
        min_radius=0,
    )

    target = assigner(boxes, torch.ones(3, dtype=torch.int64))[0]

    torch.testing.assert_allclose(target.heatmap, torch.tensor(fixture["heatmap"]))
    torch.testing.assert_allclose(target.annotation, torch.tensor(fixture["annotation"]))
    torch.testing.assert_allclose(
        target.indices, torch.tensor(fixture["indices"], dtype=torch.int64)
    )
    torch.testing.assert_allclose(target.mask, torch.tensor(fixture["mask"], dtype=torch.uint8))
    torch.testing.assert_allclose(
        target.categories, torch.tensor(fixture["categories"], dtype=torch.int64)
    )


def test_dense_decoder_matches_official_golden_output():
    fixture = load_fixture()["decoder"]
    predictions = {
        "hm": torch.tensor([[[[0.0, 1.0]], [[2.0, -1.0]]]]),
        "reg": torch.tensor([[[[0.25, 0.5]], [[0.75, 0.0]]]]),
        "height": torch.tensor([[[[1.5, -2.0]]]]),
        "dim": torch.tensor(
            [
                [
                    [[math.log(2.0), math.log(3.0)]],
                    [[math.log(4.0), math.log(5.0)]],
                    [[math.log(6.0), math.log(7.0)]],
                ]
            ],
            dtype=torch.float32,
        ),
        "vel": torch.tensor([[[[8.0, 9.0]], [[10.0, 11.0]]]]),
        "rot": torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]]),
    }
    decoded = CenterPointDecoder(
        voxel_size=(0.5, 1.0, 0.2),
        point_cloud_range=(-10.0, -20.0, -5.0, 10.0, 20.0, 3.0),
        output_stride=2,
    )(predictions)[0]

    torch.testing.assert_allclose(decoded.boxes, torch.tensor(fixture["boxes"]))
    torch.testing.assert_allclose(decoded.scores, torch.tensor(fixture["scores"]))
    torch.testing.assert_allclose(
        decoded.labels, torch.tensor(fixture["labels"], dtype=torch.int64)
    )
