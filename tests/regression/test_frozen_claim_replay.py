from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_discovery.reproducibility.identities import EndpointID
from cognitive_discovery.reproducibility.replay import ReplayStatus, replay_all_claims


ROOT = Path(__file__).resolve().parents[2]


def test_all_c01_to_c13_claims_replay_without_gpu_or_network():
    results = replay_all_claims(ROOT)
    assert list(results) == [f"C{i:02d}" for i in range(1, 14)]
    assert all(result.frozen_value_reproduced for result in results.values())
    assert {result.replay_status for result in results.values()} <= set(ReplayStatus)


def test_qwen_confirmation_endpoints_remain_separate():
    results = replay_all_claims(ROOT)
    assert results["C07"].endpoint == EndpointID.NATURAL_EFFECT_RECOVERY
    assert results["C07"].values == pytest.approx(
        {"mech_pair_test": 0.778584, "mech_task_holdout": 0.815848}, abs=1e-6
    )
    assert results["C08"].endpoint == EndpointID.COGNITIVE_COUNTERFACTUAL_RECOVERY
    assert results["C08"].values == pytest.approx(
        {"mech_pair_test": 0.250536, "mech_task_holdout": -0.427508}, abs=1e-6
    )


def test_llama_behavioral_and_mechanistic_values_replay():
    results = replay_all_claims(ROOT)
    assert results["C10"].values == pytest.approx(
        {"dual_history_test_r2": 0.835387, "immediate_state_test_r2": 0.821201},
        abs=1e-6,
    )
    assert results["C11"].values == pytest.approx(
        {
            "cognitive_mech_pair_test": 0.648886,
            "cognitive_mech_task_holdout": 0.011162,
            "natural_mech_pair_test": 0.344851,
            "natural_mech_task_holdout": 0.318750,
        },
        abs=1e-6,
    )


def test_clean_clone_availability_is_explicit():
    results = replay_all_claims(ROOT)
    assert results["C07"].replay_status == ReplayStatus.COMPACT_REPLAY_ONLY
    assert "raw condition manifest" in results["C07"].remaining_dependency
    assert results["C11"].replay_status == ReplayStatus.COMPACT_REPLAY_ONLY
    assert "basis" in results["C11"].remaining_dependency
    assert results["C13"].replay_status == ReplayStatus.EXTERNAL_DEPENDENCY
