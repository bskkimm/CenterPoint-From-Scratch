import numpy as np
import pytest

from centerpoint.data import class_balanced_infos


class DeterministicRNG:
    def __init__(self):
        self.calls = []

    def choice(self, values, size):
        self.calls.append((list(values), size))
        return np.asarray(values[:size], dtype=object)


def test_class_balancing_samples_each_class_to_uniform_count_in_class_order():
    infos = [
        {"token": "car-only", "gt_names": np.array(["car"])},
        {"token": "car-bus", "gt_names": np.array(["car", "bus"])},
        {"token": "bus-only", "gt_names": np.array(["bus"])},
    ]
    rng = DeterministicRNG()

    sampled = class_balanced_infos(infos, ("car", "bus"), rng=rng)

    assert [info["token"] for info in sampled] == [
        "car-only",
        "car-bus",
        "car-bus",
        "bus-only",
    ]
    assert [call[1] for call in rng.calls] == [2, 2]
    assert [info["token"] for info in infos] == ["car-only", "car-bus", "bus-only"]


def test_class_balancing_uses_presence_not_object_count_and_allows_duplicates():
    infos = [
        {"token": "many-cars", "gt_names": np.array(["car", "car", "car"])},
        {"token": "one-car", "gt_names": np.array(["car"])},
    ]

    sampled = class_balanced_infos(infos, ("car",), rng=np.random.RandomState(4))

    assert len(sampled) == 2
    assert all(info["token"] in {"many-cars", "one-car"} for info in sampled)


def test_class_balancing_rejects_classes_without_eligible_infos():
    infos = [{"token": "car", "gt_names": np.array(["car"])}]

    with pytest.raises(ValueError, match="bus"):
        class_balanced_infos(infos, ("car", "bus"))
