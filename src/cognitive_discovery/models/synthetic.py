"""Synthetic teachers for cognitive recovery and flexible-ceiling validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


TASKS = (
    "bandit",
    "foraging",
    "solvability",
    "information_sampling",
    "waiting",
    "effort",
    "debugging",
)
LEVELS = {
    "factor_continuation_value": ("low", "medium", "high"),
    "factor_disengagement_value": ("low", "medium", "high"),
    "factor_continuation_cost": ("low", "medium", "high"),
    "factor_progress_evidence": ("negative", "neutral", "positive"),
    "factor_success_evidence": ("low", "medium", "high"),
    "factor_uncertainty": ("low", "high"),
    "factor_prior_investment": ("low", "high"),
    "factor_controllability": ("low", "high"),
    "factor_environmental_stability": ("stable", "changing"),
    "factor_goal_continuity": ("same", "new"),
}
VALUE = {
    "low": -1.0,
    "medium": 0.0,
    "high": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
    "stable": 1.0,
    "changing": -1.0,
    "same": 1.0,
    "new": -1.0,
}


def _make(n, rng):
    frame = pd.DataFrame(
        {
            name: rng.choice(levels, size=n)
            for name, levels in LEVELS.items()
        }
    )
    frame["task_family"] = np.resize(np.asarray(TASKS), n)
    rng.shuffle(frame["task_family"].values)
    frame["history_action_1"] = rng.choice((-1.0, 1.0), size=n)
    frame["history_action_kernel"] = 0.7 * frame.history_action_1 + rng.normal(0, 0.4, n)
    frame["history_outcome_1"] = rng.choice((-1.0, 0.0, 1.0), size=n)
    frame["history_outcome_kernel"] = 0.7 * frame.history_outcome_1 + rng.normal(0, 0.4, n)
    frame["episode_id"] = [f"episode-{index // 4}" for index in range(n)]
    frame["paired_condition_id"] = [f"pair-{index}" for index in range(n)]
    frame["step"] = np.arange(n) % 4
    return frame


def _v(frame, name):
    return frame[name].map(VALUE).to_numpy(dtype=float)


def _teacher(frame, name):
    advantage = (
        _v(frame, "factor_continuation_value")
        - _v(frame, "factor_disengagement_value")
        - _v(frame, "factor_continuation_cost")
    )
    progress = _v(frame, "factor_progress_evidence")
    success = _v(frame, "factor_success_evidence")
    if name == "dynamic_reevaluation":
        return 2.2 * advantage + 0.4 * progress + 0.3 * success
    if name == "choice_perseveration":
        return 2.4 * frame.history_action_kernel.to_numpy() + 0.15 * advantage
    if name == "outcome_history":
        return 2.4 * frame.history_outcome_kernel.to_numpy() + 0.15 * advantage
    if name == "dual_history":
        return (
            1.5 * frame.history_action_kernel.to_numpy()
            + 1.8 * frame.history_outcome_kernel.to_numpy()
            + 0.3 * advantage
        )
    if name == "latent_context":
        return (
            advantage
            + 1.5
            * frame.history_outcome_kernel.to_numpy()
            * _v(frame, "factor_goal_continuity")
        )
    raise ValueError(f"unsupported synthetic teacher: {name}")


def generate_teacher_data(
    teacher: str,
    *,
    n_train: int,
    n_test: int,
    seed: int = 0,
    noise: float = 0.05,
):
    rng = np.random.default_rng(int(seed))
    frame = _make(int(n_train) + int(n_test), rng)
    frame["persistence_logit"] = _teacher(frame, teacher) + rng.normal(
        0, float(noise), len(frame)
    )
    return (
        frame.iloc[:n_train].reset_index(drop=True),
        frame.iloc[n_train:].reset_index(drop=True),
    )

