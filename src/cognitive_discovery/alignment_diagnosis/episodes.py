"""Resumable bandit trajectories from controlled prefixes to STOP or decision 100."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from cognitive_discovery.cross_task.provenance import canonical_hash, file_hash
from .design import render, sampling_seed


RESPONSE_FIELDS = (
    "raw_response",
    "action_logits",
    "action_probabilities",
    "p_action_mass_raw",
    "top_token_is_action",
)
MEASUREMENT_FIELDS = RESPONSE_FIELDS + (
    "sampling_seed",
    "actual_choice",
    "valid_response",
    "choice_continue",
    "p_continue",
    "observed_failure",
    "comprehension_correct",
    "prompt_hash",
    "correction",
)


def run_episode(config, initial, score, directory):
    from .pipeline import _measurement, records

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{initial['trial_id']}.jsonl"
    saved = records(path) if path.exists() else []
    actions, rewards = list(initial["history_actions"]), list(initial["rewards"])
    first_decision = len(rewards) + 1
    total = float(sum(rewards))
    first = None
    end_reason = None
    count = 0
    with path.open("a") as handle:
        for decision in range(first_decision, config["episodes"]["max_decisions"] + 1):
            state = deepcopy(initial)
            state.update(
                history_actions=list(actions),
                rewards=list(rewards),
                history_outcomes=[1 if r == 3 else -1 for r in rewards],
                decision_index=decision,
            )
            state["messages"] = render(state)
            state["prompt_hash"] = canonical_hash(state["messages"])
            seed = sampling_seed(
                config,
                (
                    initial["trial_id"]
                    if decision == first_decision
                    else f"{initial['trial_id']}:decision:{decision}"
                ),
            )
            index = decision - first_decision
            response = (
                {key: saved[index][key] for key in RESPONSE_FIELDS}
                if index < len(saved)
                else score(state, seed)
            )
            measured = _measurement(state, response, seed)
            if first is None:
                first = measured
            action = measured["actual_choice"]
            reward = None
            if action in ("A", "B"):
                reward_seed = sampling_seed(
                    config,
                    f"reward:{initial['phase']}:{initial['variant']}:"
                    f"{first_decision}:{initial['history']}:{decision}:{action}",
                )
                p = config["episodes"]["arm_success_probabilities"][action]
                reward = 3 if np.random.default_rng(reward_seed).random() < p else -2
                total += reward + initial["incentive"]
            compact = {key: measured[key] for key in MEASUREMENT_FIELDS}
            compact.update(
                episode_id=initial["trial_id"],
                decision_index=decision,
                reward=reward,
                continuation_bonus=initial["incentive"] if reward is not None else 0.0,
                cumulative_reward=total,
            )
            if index < len(saved):
                if compact != saved[index]:
                    raise ValueError(
                        "episode checkpoint differs from its deterministic history"
                    )
            else:
                handle.write(
                    json.dumps(compact, sort_keys=True, allow_nan=False) + "\n"
                )
                handle.flush()
            count += 1
            if index >= len(saved) and decision % 20 == 0:
                print(
                    f"episode={initial['trial_id']} decision={decision}/100", flush=True
                )
            if action == "C":
                end_reason = "stop"
            elif action is None:
                end_reason = "invalid_response"
            elif decision == 100:
                end_reason = "decision_limit"
            if end_reason:
                if len(saved) > count:
                    raise ValueError(
                        "episode checkpoint contains decisions after termination"
                    )
                break
            actions.append(action)
            rewards.append(reward)
    return {
        **first,
        "episode_end_reason": end_reason,
        "episode_model_decisions": count,
        "episode_last_decision": decision,
        "episode_prefix_decisions": first_decision - 1,
        "episode_total_reward": total,
        "trajectory_sha256": file_hash(path),
    }


def write_episode_outputs(output, observations):
    import pandas as pd
    from .pipeline import records

    summaries = []
    for row in observations:
        if not row.get("episode_mode") or row["trial_type"] != "choice":
            continue
        path = output / row["trajectory_file"]
        steps = records(path)
        summary = {
            key: row[key]
            for key in (
                "trial_id",
                "pair_id",
                "variant",
                "phase",
                "history",
                "incentive",
                "correction",
                "scheduled_correction_decision",
                "episode_end_reason",
                "episode_model_decisions",
                "episode_last_decision",
                "episode_prefix_decisions",
                "episode_total_reward",
            )
        }
        summary["post_correction_violations"] = sum(
            r["observed_failure"] is True for r in steps
        )
        summary["any_observed_violation"] = summary["post_correction_violations"] > 0
        summaries.append(summary)
    frame = pd.DataFrame(summaries)
    frame.to_csv(output / "episode_summary.csv", index=False)
    groups = [
        "phase",
        "history",
        "incentive",
        "correction",
        "scheduled_correction_decision",
    ]
    grouped = (
        frame.groupby(groups)
        .agg(
            episodes=("trial_id", "size"),
            mean_model_decisions=("episode_model_decisions", "mean"),
            observed_violations=("post_correction_violations", "sum"),
            episodes_with_violations=("any_observed_violation", "sum"),
            stopped=("episode_end_reason", lambda v: int(v.eq("stop").sum())),
            invalid_responses=(
                "episode_end_reason",
                lambda v: int(v.eq("invalid_response").sum()),
            ),
            reached_limit=(
                "episode_end_reason",
                lambda v: int(v.eq("decision_limit").sum()),
            ),
        )
        .reset_index()
    )
    grouped.to_csv(output / "episode_conditions.csv", index=False)
    result = {
        "episodes": len(frame),
        "decision_limit": 100,
        "model_decisions": int(frame.episode_model_decisions.sum()),
        "stopped": int(frame.episode_end_reason.eq("stop").sum()),
        "invalid": int(frame.episode_end_reason.eq("invalid_response").sum()),
        "reached_limit": int(frame.episode_end_reason.eq("decision_limit").sum()),
        "correction_decisions": sorted(
            map(int, frame.scheduled_correction_decision.unique())
        ),
    }
    (output / "episode_report.md").write_text(
        "# Episode trajectories\n\n"
        f"{len(frame)} episodes, {result['model_decisions']} model decisions. "
        f"STOP ended {result['stopped']} episodes; {result['invalid']} ended on an invalid response; "
        f"{result['reached_limit']} reached decision 100.\n\n"
        f"Correction timing is fully crossed at decisions {result['correction_decisions']}. Each episode starts "
        "from its controlled replay prefix and runs until STOP, invalid response, or decision 100, "
        "including the prefix. Prefix actions are prescribed, not sampled model decisions. "
        "First critical decisions enter the matched model comparison. Subsequent decisions "
        "are dependent trajectory observations and are not pooled as independent fitting trials. "
        "Episode summaries include baseline, calibration, and held-out conditions.\n"
    )
    return result
