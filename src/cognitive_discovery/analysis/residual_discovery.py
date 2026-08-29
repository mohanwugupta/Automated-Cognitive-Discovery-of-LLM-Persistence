"""Cross-task held-out search for interactions missing from a frozen cognitive model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from cognitive_discovery.models.features import FeatureEncoder
from cognitive_discovery.models.fitting import fit_model


CANDIDATE_INTERACTIONS = (
    "history_outcome_1*factor_progress_evidence",
    "history_outcome_kernel*factor_goal_continuity",
    "factor_continuation_cost*factor_progress_evidence",
    "factor_prior_investment*factor_goal_continuity",
    "history_action_1*factor_progress_evidence",
    "history_action_kernel*factor_environmental_stability",
)


def discover_residuals(
    frame: pd.DataFrame,
    *,
    base_model: str,
    interactions=CANDIDATE_INTERACTIONS,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    if "paired_condition_id" in frame:
        groups = np.asarray(sorted(frame.paired_condition_id.astype(str).unique()))
        rng.shuffle(groups)
        training_groups = set(groups[: max(1, int(0.7 * len(groups)))])
        train = frame[frame.paired_condition_id.astype(str).isin(training_groups)].copy()
        test = frame[~frame.paired_condition_id.astype(str).isin(training_groups)].copy()
    else:
        order = rng.permutation(len(frame))
        boundary = max(1, int(0.7 * len(frame)))
        train, test = frame.iloc[order[:boundary]].copy(), frame.iloc[order[boundary:]].copy()
    base = fit_model(train, base_model, sharing="fully_shared")
    train_residual = train.persistence_logit.to_numpy() - base.predict(train)
    test_residual = test.persistence_logit.to_numpy() - base.predict(test)
    null = float(np.mean((test_residual - train_residual.mean()) ** 2))
    rows = []
    for interaction in interactions:
        encoder = FeatureEncoder((interaction,))
        x_train, names = encoder.fit_transform(train)
        x_test, _ = encoder.transform(test)
        model = Ridge(alpha=0.01).fit(x_train, train_residual)
        prediction = model.predict(x_test)
        mse = float(np.mean((test_residual - prediction) ** 2))
        effect_tasks = 0
        for task, indices in test.groupby("task_family").groups.items():
            if len(indices) < 2:
                continue
            positions = test.index.get_indexer(indices)
            if np.std(prediction[positions]) > 1e-8:
                effect_tasks += 1
        rows.append(
            {
                "interaction": interaction,
                "heldout_r2_gain": 1.0 - mse / null if null > 1e-12 else float("nan"),
                "residual_mse": mse,
                "coefficient": float(model.coef_[0]),
                "families": effect_tasks,
                "feature_names": ";".join(names),
            }
        )
    return pd.DataFrame(rows).sort_values("heldout_r2_gain", ascending=False).reset_index(drop=True)

