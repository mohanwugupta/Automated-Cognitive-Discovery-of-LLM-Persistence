import numpy as np
import pandas as pd

from cognitive_discovery.analysis.residual_discovery import discover_residuals
from cognitive_discovery.audits.calibration import calibration_tables
from cognitive_discovery.audits.information_sampling import audit_information_sampling
from cognitive_discovery.data.storage import records_frame
from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.experiments.collection import collect_conditions
from cognitive_discovery.models.flexible.static import fit_static_flexible
from cognitive_discovery.models.synthetic import (
    generate_hierarchical_teacher_data,
    generate_teacher_data,
)
from cognitive_discovery.participants.base import DeterministicParticipant


def test_flexible_task_specific_and_hierarchical_teachers_recover():
    for teacher, sharing in (
        ("M2", "task_specific"),
        ("M3", "hierarchical"),
        ("M4", "task_embedding"),
    ):
        train, test = generate_hierarchical_teacher_data(
            teacher, n_train=1400, n_test=350, seed=5, noise=0
        )
        fit = fit_static_flexible(train, test, kind="linear", sharing=sharing)
        assert fit.metrics["r2"] >= 0.98


def test_negative_heldout_residual_gain_is_never_a_discovery():
    train, test = generate_teacher_data(
        "dynamic_reevaluation", n_train=500, n_test=200, seed=24, noise=0.5
    )
    frame = pd.concat([train, test], ignore_index=True)
    result = discover_residuals(
        frame,
        base_model="dynamic_reevaluation",
        interactions=("history_action_1*factor_progress_evidence",),
        bootstraps=40,
        seed=91,
    )
    negative = result[result.heldout_r2_gain.fillna(-1) <= 0]
    assert not negative.discovered.any()
    assert set(result.selection_group_digest).isdisjoint(
        set(result.validation_group_digest)
    )


def test_information_sampling_audit_semantics_masks_and_normalization(discovery_config):
    conditions = compile_design(discovery_config, n_conditions=70, seed=51)
    observations = collect_conditions(conditions, DeterministicParticipant())
    audit = audit_information_sampling(records_frame(observations))
    target = audit[(audit.section == "semantics") & (audit.item == "target")]
    assert "gather more evidence" in target.iloc[0].value
    availability = audit[audit.section == "feature_availability"]
    assert set(availability.status) == {"consistent"}
    normalization = audit[audit.section == "normalization"]
    assert normalization.iloc[0].status == "reproducible"


def test_validation_calibration_is_diagnostic_not_recalibrated():
    observed = np.linspace(-2, 2, 100)
    predictions = pd.DataFrame(
        {
            "task_family": np.resize(["bandit", "effort"], 100),
            "persistence_logit": observed,
            "predicted_persistence_logit": 1.0 + 2.0 * observed,
        }
    )
    overall, deciles, per_task = calibration_tables(predictions)
    assert np.isclose(overall.iloc[0].calibration_intercept, 1.0)
    assert np.isclose(overall.iloc[0].calibration_slope, 2.0)
    assert len(deciles) == 10
    assert set(per_task.group) == {"bandit", "effort"}
