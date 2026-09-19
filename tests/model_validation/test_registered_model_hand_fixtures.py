from pathlib import Path

from cognitive_discovery.model_validation.specifications import validate_specifications
from cognitive_discovery.models.cognitive.registry import COGNITIVE_MODELS
import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_every_registered_model_has_a_complete_frozen_specification():
    hashes = validate_specifications(ROOT)
    assert set(hashes) == set(COGNITIVE_MODELS)
    assert all(len(value) == 64 for value in hashes.values())


def test_validation_tolerances_are_preregistered_before_recovery():
    config = yaml.safe_load((ROOT / "configs/validation/computational_models_v1.yaml").read_text())
    assert config["model_recovery"]["key_theories"] == ["dual_history", "latent_context", "outcome_history"]
    assert config["model_recovery"]["minimum_diagonal_recovery_probability"] == .80
