from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cognitive_discovery.reproducibility.identities import (
    EndpointID,
    MetricID,
    canonical_metric,
    persistence_logit,
    recovery_target,
    validate_binary_interface,
)
from cognitive_discovery.reproducibility.metrics import global_cfr_v1
from cognitive_discovery.causal_mechanistic.metrics import global_counterfactual_recovery


def test_global_cfr_v1_is_aggregate_not_mean_per_example():
    target = np.array([1.0, 10.0])
    observed = np.array([0.0, 10.0])
    expected = 1.0 - 1.0 / 101.0
    assert global_cfr_v1(target, observed) == pytest.approx(expected)
    assert global_cfr_v1(target, observed) == pytest.approx(
        global_counterfactual_recovery(target, observed)
    )
    legacy_mean = np.mean(1.0 - np.square(observed - target) / np.square(target))
    assert legacy_mean == pytest.approx(0.5)
    assert global_cfr_v1(target, observed) != pytest.approx(legacy_mean)
    assert canonical_metric(MetricID.GLOBAL_CFR_V1) is global_cfr_v1
    with pytest.raises(ValueError, match="not canonical"):
        canonical_metric("mean_per_example_cfr")


def test_global_cfr_v1_validates_alignment_and_degenerate_target():
    with pytest.raises(ValueError, match="align"):
        global_cfr_v1([1, 2], [1])
    assert global_cfr_v1([0, 0], [0, 0]) == 1.0
    assert math.isnan(global_cfr_v1([0, 0], [1, 0]))


def test_recovery_endpoints_select_different_columns_and_fail_closed():
    frame = pd.DataFrame(
        {
            "predicted_counterfactual_effect": [1.0, 2.0],
            "natural_effect": [3.0, 4.0],
        }
    )
    assert recovery_target(frame, EndpointID.COGNITIVE_COUNTERFACTUAL_RECOVERY).tolist() == [1, 2]
    assert recovery_target(frame, EndpointID.NATURAL_EFFECT_RECOVERY).tolist() == [3, 4]
    with pytest.raises(ValueError, match="unknown endpoint"):
        recovery_target(frame, "recovery")
    with pytest.raises(ValueError, match="natural_effect"):
        recovery_target(frame.drop(columns="natural_effect"), EndpointID.NATURAL_EFFECT_RECOVERY)


def test_persistence_target_and_interface_identity():
    assert persistence_logit({"Yes": 2.0, "No": 0.5}, "Yes", "No") == pytest.approx(1.5)
    assert persistence_logit({"Yes": 2.0, "No": 0.5}, "No", "Yes") == pytest.approx(-1.5)
    with pytest.raises(ValueError, match="distinct"):
        persistence_logit({"Yes": 2.0}, "Yes", "Yes")
    validate_binary_interface(("Yes", "No"), continue_label="Yes", interface_id="llama_yes_no_v2")
    validate_binary_interface(("X", "Y"), continue_label="X", interface_id="qwen_xy_v1")
    with pytest.raises(ValueError, match="interface"):
        validate_binary_interface(("A", "B"), continue_label="A", interface_id="llama_yes_no_v2")
