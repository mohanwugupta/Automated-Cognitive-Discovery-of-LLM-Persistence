"""Paper-facing parameter sign and ontology-variance summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGISTERED_PARAMETERS = (
    "factor_success_evidence",
    "factor_progress_evidence",
    "factor_continuation_cost",
    "factor_disengagement_value",
    "history_action_kernel",
    "history_outcome_kernel",
)


def parameter_sign_table(
    parameters: pd.DataFrame, *, near_zero: float = 0.05
) -> pd.DataFrame:
    rows = []
    for name in REGISTERED_PARAMETERS:
        part = parameters[parameters.parameter == name]
        values = part.estimate.to_numpy(dtype=float)
        if not len(values):
            rows.append(
                {
                    "parameter": name,
                    "tasks": 0,
                    "probability_positive": float("nan"),
                    "probability_negative": float("nan"),
                    "median_absolute_estimate": float("nan"),
                    "classification": "unavailable",
                }
            )
            continue
        median_abs = float(np.median(np.abs(values)))
        positive = float(np.mean(values > 0))
        negative = float(np.mean(values < 0))
        if median_abs <= float(near_zero):
            classification = "near_zero"
        elif positive >= 0.8 or negative >= 0.8:
            classification = "shared_sign"
        else:
            classification = "mixed_sign"
        rows.append(
            {
                "parameter": name,
                "tasks": len(values),
                "probability_positive": positive,
                "probability_negative": negative,
                "median_absolute_estimate": median_abs,
                "classification": classification,
            }
        )
    return pd.DataFrame(rows)
