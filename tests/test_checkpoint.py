import random

import numpy as np
import pytest
import torch
from torch import nn

from centerpoint.engine.checkpoint import (
    _torch_load,
    capture_rng_state,
    config_sha256,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


def test_config_hash_is_stable_across_mapping_order():
    assert config_sha256({"a": 1, "b": [2, 3]}) == config_sha256(
        {"b": [2, 3], "a": 1}
    )


def test_rng_state_round_trip_restores_all_cpu_generators():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))

    random.random()
    np.random.rand()
    torch.rand(1)
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_allclose(actual[2], expected[2])


def test_checkpoint_round_trip_restores_training_state(tmp_path):
    torch.manual_seed(3)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss = model(torch.ones((1, 2))).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_weight = model.weight.detach().clone()
    expected_optimizer = optimizer.state_dict()
    path = tmp_path / "nested" / "checkpoint.pth"
    config = {"model": {"name": "tiny"}, "seed": 3}

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        epoch=4,
        global_step=123,
        environment={"test": True},
    )
    with torch.no_grad():
        model.weight.zero_()
    optimizer.state.clear()

    metadata = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    torch.testing.assert_allclose(model.weight, expected_weight)
    assert optimizer.state_dict()["state"].keys() == expected_optimizer["state"].keys()
    assert scheduler.last_epoch == 1
    assert metadata.epoch == 4
    assert metadata.global_step == 123
    assert metadata.config_sha256 == config_sha256(config)
    assert metadata.environment == {"test": True}
    assert metadata.resume_scope == "single_process_epoch_boundary"


def test_checkpoint_rejects_tampered_config_hash(tmp_path):
    path = tmp_path / "checkpoint.pth"
    model = nn.Linear(1, 1)
    save_checkpoint(path, model=model, config={"value": 1}, epoch=0, global_step=0)
    payload = _torch_load(path, "cpu")
    payload["config"]["value"] = 2
    torch.save(payload, path)

    with pytest.raises(ValueError, match="config hash"):
        load_checkpoint(path, model=model)


def test_checkpoint_requires_requested_optional_state(tmp_path):
    path = tmp_path / "checkpoint.pth"
    model = nn.Linear(1, 1)
    save_checkpoint(path, model=model, config={}, epoch=0, global_step=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="optimizer"):
        load_checkpoint(path, model=model, optimizer=optimizer)


def test_checkpoint_rejects_unknown_model_state_version(tmp_path):
    path = tmp_path / "checkpoint.pth"
    model = nn.Linear(1, 1)
    save_checkpoint(path, model=model, config={}, epoch=0, global_step=0)
    payload = _torch_load(path, "cpu")
    payload["model_state_version"] = 2
    torch.save(payload, path)

    with pytest.raises(ValueError, match="model state version"):
        load_checkpoint(path, model=model)


def test_checkpoint_unwraps_data_parallel_state(tmp_path):
    path = tmp_path / "checkpoint.pth"
    wrapped = nn.DataParallel(nn.Linear(2, 1))
    expected = wrapped.module.weight.detach().clone()

    save_checkpoint(path, model=wrapped, config={}, epoch=0, global_step=0)
    with torch.no_grad():
        wrapped.module.weight.zero_()
    load_checkpoint(path, model=wrapped)

    torch.testing.assert_allclose(wrapped.module.weight, expected)


def test_checkpoint_rejects_distributed_save_without_rank_state(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    with pytest.raises(RuntimeError, match="single-process"):
        save_checkpoint(
            tmp_path / "checkpoint.pth",
            model=nn.Linear(1, 1),
            config={},
            epoch=0,
            global_step=0,
        )
