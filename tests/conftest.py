from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="session")
def discovery_config():
    path = Path(__file__).parents[1] / "configs" / "discovery_v1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
