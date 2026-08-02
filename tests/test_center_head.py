import torch

from centerpoint.config import NUSCENES_VOXELNET_075
from centerpoint.contracts import TaskTargets
from centerpoint.models.heads import CenterHead


def make_head():
    config = NUSCENES_VOXELNET_075
    branches = {
        branch.name: (branch.output_channels, branch.num_convolutions)
        for branch in config.model.head.branches
    }
    return CenterHead(
        in_channels=config.model.head.input_channels,
        tasks=config.tasks,
        common_heads=branches,
        share_conv_channel=config.model.head.shared_channels,
        loss_weight=config.model.head.loss_weight,
        code_weights=config.model.head.code_weights,
    )


def make_targets(class_counts, batch_size=2, height=3, width=4):
    targets = []
    for class_count in class_counts:
        heatmap = torch.zeros((batch_size, class_count, height, width))
        annotation = torch.zeros((batch_size, 2, 10))
        indices = torch.zeros((batch_size, 2), dtype=torch.int64)
        mask = torch.zeros((batch_size, 2), dtype=torch.uint8)
        categories = torch.zeros((batch_size, 2), dtype=torch.int64)
        heatmap[:, 0, 0, 0] = 1
        mask[:, 0] = 1
        targets.append(TaskTargets(heatmap, annotation, indices, mask, categories))
    return targets


def test_center_head_matches_parameter_state_and_initialization_contracts():
    torch.manual_seed(7)
    head = make_head()

    assert sum(parameter.numel() for parameter in head.parameters()) == 1_669_510
    state = head.state_dict()
    assert state["shared_conv.0.weight"].shape == (64, 512, 3, 3)
    assert state["tasks.1.hm.3.weight"].shape == (2, 64, 3, 3)
    assert state["tasks.0.reg.3.weight"].shape == (2, 64, 3, 3)
    assert "code_weights" not in state
    assert torch.count_nonzero(state["tasks.0.reg.0.bias"]) == 0
    assert torch.count_nonzero(state["tasks.0.reg.3.bias"]) == 0
    torch.testing.assert_allclose(
        state["tasks.0.hm.3.bias"],
        torch.full((1,), -2.19),
    )

    expected = {
        "shared_conv.0.weight": (64, 512, 3, 3),
        "shared_conv.0.bias": (64,),
    }
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        expected[f"shared_conv.1.{suffix}"] = (64,)
    expected["shared_conv.1.num_batches_tracked"] = ()
    outputs = {"reg": 2, "height": 1, "dim": 3, "rot": 2, "vel": 2}
    for task_index, class_count in enumerate((1, 2, 2, 1, 2, 2)):
        for branch, output_channels in (*outputs.items(), ("hm", class_count)):
            prefix = f"tasks.{task_index}.{branch}"
            expected[f"{prefix}.0.weight"] = (64, 64, 3, 3)
            expected[f"{prefix}.0.bias"] = (64,)
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                expected[f"{prefix}.1.{suffix}"] = (64,)
            expected[f"{prefix}.1.num_batches_tracked"] = ()
            expected[f"{prefix}.3.weight"] = (output_channels, 64, 3, 3)
            expected[f"{prefix}.3.bias"] = (output_channels,)

    assert {name: tuple(value.shape) for name, value in state.items()} == expected


def test_center_head_emits_six_ordered_tasks_and_shared_features():
    head = make_head().eval()
    predictions, shared = head(torch.randn((2, 512, 3, 4)))

    assert shared.shape == (2, 64, 3, 4)
    assert len(predictions) == 6
    assert [prediction["hm"].shape[1] for prediction in predictions] == [1, 2, 2, 1, 2, 2]
    assert tuple(predictions[0]) == ("reg", "height", "dim", "rot", "vel", "hm")
    assert predictions[0]["reg"].shape == (2, 2, 3, 4)


def test_center_head_integrates_six_task_losses_and_gradients():
    torch.manual_seed(11)
    head = make_head().train()
    predictions, _ = head(torch.randn((2, 512, 3, 4)))
    targets = make_targets([1, 2, 2, 1, 2, 2])

    losses = head.loss(predictions, targets)
    total = sum(losses["loss"])
    total.backward()

    assert tuple(losses) == ("loss", "hm_loss", "loc_loss", "loc_loss_elem", "num_positive")
    assert all(len(values) == 6 for values in losses.values())
    assert all(values.shape == (10,) for values in losses["loc_loss_elem"])
    assert all(value.item() == 2 for value in losses["num_positive"])
    assert torch.isfinite(total)
    assert head.shared_conv[0].weight.grad is not None
    assert head.tasks[5].hm[3].weight.grad is not None
