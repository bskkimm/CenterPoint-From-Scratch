"""Versioned, reproducible training checkpoints."""

import hashlib
import json
import os
import platform
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointMetadata:
    """Validated progress and provenance loaded from a checkpoint."""

    epoch: int
    global_step: int
    config_sha256: str
    environment: Mapping[str, Any]


def config_sha256(config: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible config independently of dictionary insertion order."""

    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def environment_manifest() -> Dict[str, Any]:
    """Capture runtime versions relevant to checkpoint interpretation."""

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "executable": sys.executable,
    }


def capture_rng_state() -> Dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and available CUDA generator states."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore generator states captured by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"RNG state is missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Union[str, Path],
    *,
    model: nn.Module,
    config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    environment: Optional[Mapping[str, Any]] = None,
) -> None:
    """Atomically write a complete schema-versioned training checkpoint."""

    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step must be non-negative")

    config_snapshot = json.loads(
        json.dumps(config, sort_keys=True, allow_nan=False)
    )
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "config_sha256": config_sha256(config_snapshot),
        "environment": dict(environment) if environment is not None else environment_manifest(),
    }
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata,
        "config": config_snapshot,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(str(temporary_path), str(destination))
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    path: Union[str, Path],
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Any = "cpu",
    restore_rng: bool = False,
) -> CheckpointMetadata:
    """Validate and restore a schema-version-1 checkpoint."""

    payload = torch.load(Path(path), map_location=map_location)
    required = {
        "schema_version",
        "metadata",
        "config",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng_state",
    }
    if not isinstance(payload, dict) or required.difference(payload):
        missing = sorted(required.difference(payload)) if isinstance(payload, dict) else sorted(required)
        raise ValueError(f"invalid checkpoint payload; missing keys: {missing}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema version {payload['schema_version']}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )

    metadata = payload["metadata"]
    metadata_keys = {"epoch", "global_step", "config_sha256", "environment"}
    if not isinstance(metadata, dict) or metadata_keys.difference(metadata):
        raise ValueError("checkpoint metadata is incomplete")
    if config_sha256(payload["config"]) != metadata["config_sha256"]:
        raise ValueError("checkpoint config hash does not match its config snapshot")

    model.load_state_dict(payload["model"], strict=True)
    _restore_optional_state("optimizer", optimizer, payload["optimizer"])
    _restore_optional_state("scheduler", scheduler, payload["scheduler"])
    _restore_optional_state("scaler", scaler, payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])

    return CheckpointMetadata(
        epoch=int(metadata["epoch"]),
        global_step=int(metadata["global_step"]),
        config_sha256=metadata["config_sha256"],
        environment=metadata["environment"],
    )


def _restore_optional_state(name: str, component: Optional[Any], state: Any) -> None:
    if component is None:
        return
    if state is None:
        raise ValueError(f"checkpoint does not contain requested {name} state")
    component.load_state_dict(state)
