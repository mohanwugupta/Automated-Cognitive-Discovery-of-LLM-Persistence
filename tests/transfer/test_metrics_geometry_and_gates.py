import numpy as np
import pandas as pd
import pytest

from cognitive_discovery.transfer.geometry import controller_geometry
from cognitive_discovery.transfer.metrics import summarize_transfer
from cognitive_discovery.transfer.pipeline import gate_source_controllers, project_pilot_resources


def test_transfer_metric_identity_and_perfect_recovery():
    frame = pd.DataFrame({"predicted_counterfactual_effect": [-1., -.5, .5, 1.], "neural_counterfactual_effect": [-1., -.5, .5, 1.], "contrast_id": ["a", "b", "c", "d"]})
    result = summarize_transfer(frame, model="m", theory="t", source_task="bandit", target_task="effort", layer=3, rank=1, bootstrap_samples=100, random_cfrs=[-.2, .1])
    assert result["global_cfr"] == 1
    assert result["endpoint_id"] == "cognitive_counterfactual_recovery"
    assert result["metric_id"] == "global_cfr_v1"
    assert result["random_p"] == 1 / 3


def test_geometry_identical_and_orthogonal_subspaces():
    left = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
    identical = controller_geometry(left, left)
    orthogonal = controller_geometry(left, np.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]]))
    assert identical["subspace_overlap"] == 1
    assert orthogonal["subspace_overlap"] == 0


def test_failed_source_gate_marks_transfer_unavailable_not_zero():
    frame = pd.DataFrame([
        {"model": "m", "theory": "t", "source_task": "a", "target_task": "a", "global_cfr": -.1, "correlation": .1, "slope": .1, "bootstrap_low": -.2, "bootstrap_high": .1, "random_p": .5},
        {"model": "m", "theory": "t", "source_task": "a", "target_task": "b", "global_cfr": .4, "correlation": .4, "slope": .4, "bootstrap_low": .1, "bootstrap_high": .7, "random_p": .01},
    ])
    result = gate_source_controllers(frame)
    off = result[result.target_task == "b"].iloc[0]
    assert off.status == "unavailable"
    assert np.isnan(off.global_cfr)


def test_pilot_projection_scales_measured_gpu_time_and_storage():
    result = project_pilot_resources(
        train_seconds=600,
        diagonal_seconds=120,
        off_diagonal_seconds=480,
        pilot_storage_bytes=2_000_000_000,
        source_controllers=14,
    )
    assert result["pilot_gpu_hours"] == 1200 / 3600
    assert result["projected_full_gpu_hours"] == pytest.approx(14 * 1200 / 3600)
    assert result["projected_full_storage_gb"] == 28.0
    assert result["approval_required"] is True


def test_smoke_pilot_projection_accounts_for_full_grid_epochs_and_controls():
    result = project_pilot_resources(
        train_seconds=600,
        diagonal_seconds=120,
        off_diagonal_seconds=480,
        pilot_storage_bytes=2_000_000_000,
        source_controllers=14,
        train_scale=24,
        evaluation_scale=50,
        storage_scale=50,
    )
    expected_per_controller = (600 * 24 + (120 + 480) * 50) / 3600
    assert result["projected_gpu_hours_per_source_controller"] == pytest.approx(
        expected_per_controller
    )
    assert result["projected_full_gpu_hours"] == pytest.approx(
        expected_per_controller * 14
    )
    assert result["projected_full_storage_gb"] == 1400.0
