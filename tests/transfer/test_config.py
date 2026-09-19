from pathlib import Path

from cognitive_discovery.transfer.config import TASKS, load_transfer_config


ROOT = Path(__file__).resolve().parents[2]


def test_transfer_config_freezes_endpoint_metric_tasks_and_storage():
    config = load_transfer_config(ROOT / "configs/transfer/v1.yaml")
    assert tuple(config["tasks"]) == TASKS
    assert config["endpoint_id"] == "cognitive_counterfactual_recovery"
    assert config["metric_id"] == "global_cfr_v1"
    assert config["predictors"]["frozen_before_matrix"] is True
    assert config["artifact_policy"]["save_full_activations"] is False
