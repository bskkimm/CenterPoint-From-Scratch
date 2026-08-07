"""Pinned class-balanced group sampling for nuScenes metadata."""

from typing import Any, Mapping, Sequence

import numpy as np


def class_balanced_infos(
    infos: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    rng: Any = np.random,
) -> list[Mapping[str, Any]]:
    """Resample frames so every configured class contributes equally.

    A frame contributes once to a class when that class is present, regardless of
    how many objects of that class it contains. Sampling is with replacement and
    class groups are concatenated in the configured class order.
    """

    if not class_names:
        raise ValueError("class_names must not be empty")
    class_infos = {name: [] for name in class_names}
    for info in infos:
        names = set(info["gt_names"])
        for name in class_names:
            if name in names:
                class_infos[name].append(info)

    missing = [name for name, values in class_infos.items() if not values]
    if missing:
        raise ValueError(f"no eligible infos for class(es): {', '.join(missing)}")

    total = sum(len(values) for values in class_infos.values())
    class_count = len(class_names)
    sampled = []
    for name in class_names:
        values = class_infos[name]
        ratio = (1.0 / class_count) / (len(values) / total)
        sampled.extend(rng.choice(values, int(len(values) * ratio)))
    return list(sampled)
