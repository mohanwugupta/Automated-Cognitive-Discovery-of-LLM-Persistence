from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="session")
def discovery_config():
    path = Path(__file__).parents[1] / "configs" / "discovery_v1.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def theory_config():
    path = Path(__file__).parents[1] / "configs" / "theory_resolution_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["theory_resolution"]["allow_small_candidate_pool"] = True
    return config
