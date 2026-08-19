import math

import pytest
import torch

from centerpoint.models.heads import CenterPointDecoder


def make_predictions():
    return {
        "hm": torch.tensor([[[[0.0, 1.0]], [[2.0, -1.0]]]]),
        "reg": torch.tensor([[[[0.25, 0.5]], [[0.75, 0.0]]]]),
        "height": torch.tensor([[[[1.5, -2.0]]]]),
        "dim": torch.tensor(
            [[[[math.log(2.0), math.log(3.0)]], [[math.log(4.0), math.log(5.0)]], [[math.log(6.0), math.log(7.0)]]]]
        ),
        "vel": torch.tensor([[[[8.0, 9.0]], [[10.0, 11.0]]]]),
        "rot": torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]]),
    }


def test_decoder_reconstructs_metric_boxes_and_one_class_per_cell():
    decoder = CenterPointDecoder(
        voxel_size=(0.5, 1.0, 0.2),
        point_cloud_range=(-10.0, -20.0, -5.0, 10.0, 20.0, 3.0),
        output_stride=2,
    )

    result = decoder(make_predictions(), label_offset=3)[0]

    expected_boxes = torch.tensor(
        [
            [-9.75, -18.5, 1.5, 2.0, 4.0, 6.0, 8.0, 10.0, 0.0],
            [-8.5, -20.0, -2.0, 3.0, 5.0, 7.0, 9.0, 11.0, math.pi / 2],
        ]
    )
    torch.testing.assert_allclose(result.boxes, expected_boxes, atol=1e-6, rtol=1e-6)
    torch.testing.assert_allclose(
        result.scores,
        torch.tensor([torch.sigmoid(torch.tensor(2.0)), torch.sigmoid(torch.tensor(1.0))]),
    )
    assert result.labels.tolist() == [4, 3]


def test_decoder_uses_strict_score_and_inclusive_center_range_filters():
    predictions = make_predictions()
    predictions["hm"] = torch.zeros_like(predictions["hm"])
    decoder = CenterPointDecoder(
        voxel_size=(1.0, 1.0, 1.0),
        point_cloud_range=(0.0, 0.0, -5.0, 4.0, 4.0, 5.0),
        output_stride=1,
        score_threshold=0.5,
        post_center_range=(0.25, 0.0, -2.0, 1.5, 0.75, 1.5),
    )

    empty = decoder(predictions)[0]
    assert empty.boxes.shape == (0, 9)

    predictions["hm"][:, 0, 0, 0] = 0.01
    boundary = decoder(predictions)[0]
    assert boundary.boxes.shape == (1, 9)
    torch.testing.assert_allclose(boundary.boxes[0, :3], torch.tensor([0.25, 0.75, 1.5]))


def test_decoder_supports_velocity_free_heads():
    predictions = make_predictions()
    del predictions["vel"]

    result = CenterPointDecoder((1, 1, 1), (0, 0, -5, 4, 4, 5), 1)(predictions)[0]

    assert result.boxes.shape == (2, 7)
    torch.testing.assert_allclose(result.boxes[:, -1], torch.tensor([0.0, math.pi / 2]))


def test_decoder_disables_autograd_like_official_predict():
    predictions = make_predictions()
    predictions = {name: value.requires_grad_() for name, value in predictions.items()}

    result = CenterPointDecoder((1, 1, 1), (0, 0, -5, 4, 4, 5), 1)(predictions)[0]

    assert not result.boxes.requires_grad
    assert not result.scores.requires_grad


def test_decoder_validates_head_shapes():
    predictions = make_predictions()
    predictions["reg"] = torch.zeros((1, 3, 1, 2))

    with pytest.raises(ValueError, match="reg"):
        CenterPointDecoder((1, 1, 1), (0, 0, 0, 1, 1, 1), 1)(predictions)


def test_decoder_maps_row_major_cells_to_metric_centers_on_a_non_square_grid():
    height, width = 2, 3
    zeros = torch.zeros((1, 1, height, width))
    predictions = {
        "hm": zeros.clone(),
        "reg": torch.zeros((1, 2, height, width)),
        "height": zeros.clone(),
        "dim": torch.zeros((1, 3, height, width)),
        "rot": torch.cat((zeros.clone(), torch.ones_like(zeros)), dim=1),
    }

    # Distinct x/y voxel sizes make a transposed grid observable in both coordinates.
    result = CenterPointDecoder(
        voxel_size=(1.0, 2.0, 0.2),
        point_cloud_range=(0.0, 0.0, -5.0, 10.0, 20.0, 3.0),
        output_stride=1,
    )(predictions)[0]

    expected_centers = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )
    assert result.boxes.shape == (height * width, 7)
    torch.testing.assert_allclose(result.boxes[:, :2], expected_centers)
