import json
from pathlib import Path

from centerpoint.config import NUSCENES_VOXELNET_075


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "official_3cf7d870.json"


def test_canonical_manifest_matches_pinned_official_config():
    fixture = json.loads(FIXTURE_PATH.read_text())

    assert NUSCENES_VOXELNET_075.to_official_manifest() == fixture["config"]
