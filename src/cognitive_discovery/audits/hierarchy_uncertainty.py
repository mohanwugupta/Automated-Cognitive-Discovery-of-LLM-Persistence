"""Paired M2/M3/M4 uncertainty with semantic-condition/task resampling."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from cognitive_discovery.models.fitting import task_macro_metrics


PAIRS = (("M2", "M3"), ("M2", "M4"), ("M3", "M4"))


def _split_digest(frame: pd.DataFrame) -> str:
    key = "paired_condition_id" if "paired_condition_id" in frame else "episode_id"
    payload = "\n".join(sorted(frame[key].astype(str).unique()))
    return hashlib.sha256(payload.encode()).hexdigest()


def paired_hierarchy_bootstrap(
    test: pd.DataFrame,
    fits: dict[tuple[str, str], object],
    *,
    architectures: tuple[str, ...],
    bootstraps: int = 500,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired draws and summaries for every architecture/variant pair."""

    if test.empty:
        raise ValueError("hierarchy bootstrap requires a non-empty fixed test split")
    group = "paired_condition_id" if "paired_condition_id" in test else "episode_id"
    rng = np.random.default_rng(int(seed))
    digest = _split_digest(test)
    draws = []
    for architecture in architectures:
        predictions = {
            variant: np.asarray(fits[(architecture, variant)].predict(test), dtype=float)
            for variant in ("M2", "M3", "M4")
        }
        task_groups = {
            task: np.asarray(sorted(part[group].astype(str).unique()))
            for task, part in test.groupby("task_family")
        }
        for bootstrap in range(int(bootstraps)):
            positions = []
            for task, groups in task_groups.items():
                sampled = rng.choice(groups, size=len(groups), replace=True)
                task_values = test.task_family.astype(str).to_numpy() == str(task)
                group_values = test[group].astype(str).to_numpy()
                positions.extend(
                    np.flatnonzero(task_values & (group_values == selected)).tolist()
                    for selected in sampled
                )
            indices = np.asarray(
                [index for block in positions for index in block], dtype=int
            )
            sampled_frame = test.iloc[indices].reset_index(drop=True)
            metrics = {
                variant: task_macro_metrics(sampled_frame, prediction[indices])["macro_r2"]
                for variant, prediction in predictions.items()
            }
            for left, right in PAIRS:
                draws.append(
                    {
                        "architecture": architecture,
                        "bootstrap": bootstrap,
                        "variant_a": left,
                        "variant_b": right,
                        "delta_macro_r2": metrics[left] - metrics[right],
                        "split_digest": digest,
                        "paired_semantic_bootstrap": True,
                    }
                )
    draw_frame = pd.DataFrame(draws)
    summaries = []
    for keys, part in draw_frame.groupby(["architecture", "variant_a", "variant_b"]):
        values = part.delta_macro_r2.to_numpy(dtype=float)
        summaries.append(
            {
                "architecture": keys[0],
                "variant_a": keys[1],
                "variant_b": keys[2],
                "mean_difference": float(np.mean(values)),
                "median_difference": float(np.median(values)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "probability_a_wins": float(np.mean(values > 0)),
                "probability_b_wins": float(np.mean(values < 0)),
                "split_digest": part.split_digest.iloc[0],
            }
        )
    return draw_frame, pd.DataFrame(summaries)
