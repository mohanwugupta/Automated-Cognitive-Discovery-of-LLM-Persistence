"""Surface and classify every flexible-teacher recovery check."""

from __future__ import annotations

import numpy as np
import pandas as pd


FAILURE_CATEGORIES = {
    "optimization_failure",
    "generalization_failure",
    "feature_mismatch",
    "sharing_mismatch",
    "numerical_or_data_bug",
    "expected_regularization_difference",
}


def _failure_category(row) -> tuple[str, str]:
    if bool(row.passed):
        return "passed", "student met the registered recovery threshold"
    train_r2 = getattr(row, "training_r2", np.nan)
    if np.isfinite(train_r2) and float(train_r2) < float(row.threshold):
        return (
            "optimization_failure",
            "student did not recover the frozen teacher on its training design",
        )
    teacher = str(row.teacher)
    sharing = str(row.sharing)
    if teacher.startswith("fitted_") and sharing == "fully_shared":
        return (
            "sharing_mismatch",
            "fully shared student cannot reproduce task-varying teacher parameters",
        )
    if teacher == "nonlinear_interaction" and str(row.model) in {"linear", "gam"}:
        return (
            "feature_mismatch",
            "student feature map does not contain the teacher interaction exactly",
        )
    if np.isfinite(train_r2):
        return (
            "generalization_failure",
            "training recovery passed but held-out teacher recovery did not",
        )
    return (
        "expected_regularization_difference",
        "regularized student did not meet the near-noiseless recovery criterion",
    )


def classify_teacher_checks(
    checks: pd.DataFrame, *, expected_checks: int | None = None
) -> pd.DataFrame:
    required = {"teacher", "model", "sharing", "r2", "threshold", "passed"}
    missing = required - set(checks)
    if missing:
        raise ValueError(f"teacher checks are missing: {sorted(missing)}")
    if expected_checks is not None and len(checks) != int(expected_checks):
        raise RuntimeError(
            f"teacher audit surfaced {len(checks)} checks; expected {expected_checks}"
        )
    rows = []
    for row in checks.itertuples(index=False):
        category, reason = _failure_category(row)
        teacher_r2 = float(getattr(row, "teacher_r2", 1.0))
        student_r2 = float(row.r2)
        rows.append(
            {
                "teacher_architecture": str(row.teacher),
                "student_architecture": str(row.model),
                "sharing_structure": str(row.sharing),
                "teacher_r2": teacher_r2,
                "student_r2": student_r2,
                "delta_r2": student_r2 - teacher_r2,
                "training_r2": float(getattr(row, "training_r2", np.nan)),
                "threshold": float(row.threshold),
                "passed": bool(row.passed),
                "failure_category": category,
                "failure_reason": reason,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(checks):
        raise RuntimeError("a teacher-recovery check was silently omitted")
    invalid = set(result.loc[~result.passed, "failure_category"]) - FAILURE_CATEGORIES
    if invalid:
        raise RuntimeError(f"unregistered failure categories: {sorted(invalid)}")
    return result
