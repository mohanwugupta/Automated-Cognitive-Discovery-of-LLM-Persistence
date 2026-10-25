"""Counterbalanced timing and resumable bandit trajectories, using CPU fake responses."""

from collections import Counter
import json
from pathlib import Path

import pytest
import pandas as pd

from cognitive_discovery.alignment_diagnosis.design import (
    factorial,
    load_config,
    main_design,
)
from cognitive_discovery.alignment_diagnosis.pipeline import (
    records,
    prepare,
    collect,
    freeze_predictions,
    analyze,
    observation_paths,
)
from cognitive_discovery.alignment_diagnosis.participant import SyntheticParticipant


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def pandas3_strings():
    with pd.option_context("future.infer_string", True):
        yield


@pytest.fixture
def config():
    return load_config(ROOT / "configs/alignment_diagnosis/episodes_v2.yaml")


def response(action):
    return {
        "raw_response": action,
        "action_logits": {"A": 0.0, "B": 0.0, "C": 0.0},
        "action_probabilities": {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
        "p_action_mass_raw": 1.0,
        "top_token_is_action": True,
    }


def test_timing_is_fully_crossed_and_arm_histories_are_counterbalanced(config):
    rows = factorial(config, phase="pilot")
    assert len(rows) == 96
    cells = Counter(
        (
            r["correction"],
            r["history"],
            r["incentive"],
            r["scheduled_correction_decision"],
        )
        for r in rows
    )
    assert len(cells) == 24
    assert set(cells.values()) == {4}
    pairs = {}
    for row in rows:
        assert len(row["rewards"]) == row["scheduled_correction_decision"] - 1
        pairs.setdefault(row["pair_id"], []).append(row)
    for left, right in pairs.values():
        assert left["rewards"] == right["rewards"]
        assert left["history_actions"] == right["history_actions"]
        assert left["messages"][1] == right["messages"][1]
    for timing in (9, 50, 90):
        actions = [
            a
            for r in rows
            if r["scheduled_correction_decision"] == timing
            for a in r["history_actions"]
        ]
        assert actions.count("A") == actions.count("B")
    assert rows == factorial(config, phase="pilot")
    assert not [r["correction"] for r in rows] == sorted(r["correction"] for r in rows)


def test_episode_reaches_decision_100_and_stop_absorbs(config, tmp_path):
    from cognitive_discovery.alignment_diagnosis.episodes import run_episode

    row = next(
        r
        for r in factorial(config, phase="pilot")
        if r["correction"] and r["scheduled_correction_decision"] == 9
    )
    calls = []

    def always_play(state, seed):
        calls.append(state)
        return response("A")

    first = run_episode(config, row, always_play, tmp_path)
    assert first["episode_end_reason"] == "decision_limit"
    assert first["episode_last_decision"] == 100
    assert len(calls) == 92
    assert all(r["correction"] for r in calls)
    trace = records(tmp_path / f"{row['trial_id']}.jsonl")
    assert len(trace) == 92
    assert {r["reward"] for r in trace} <= {-2, 3}
    assert all(r["observed_failure"] for r in trace)
    assert not any(
        "messages" in r for r in trace
    ), "avoid duplicating every history in logs"

    stop_dir = tmp_path / "stop"
    stopped = run_episode(config, row, lambda state, seed: response("C"), stop_dir)
    assert stopped["episode_end_reason"] == "stop"
    assert stopped["episode_model_decisions"] == 1
    assert stopped["observed_failure"] is False


def test_episode_resumes_without_resampling_completed_steps(config, tmp_path):
    from cognitive_discovery.alignment_diagnosis.episodes import run_episode

    row = next(
        r
        for r in factorial(config, phase="pilot")
        if r["scheduled_correction_decision"] == 90
    )

    def interrupted(state, seed):
        if state["decision_index"] == 92:
            raise RuntimeError("simulated time limit")
        return response("B")

    with pytest.raises(RuntimeError, match="time limit"):
        run_episode(config, row, interrupted, tmp_path)
    assert len(records(tmp_path / f"{row['trial_id']}.jsonl")) == 2
    calls = []

    def finish(state, seed):
        calls.append(state["decision_index"])
        return response("C")

    result = run_episode(config, row, finish, tmp_path)
    assert calls == [92]
    assert result["episode_model_decisions"] == 3
    assert result["episode_end_reason"] == "stop"


def test_invalid_response_ends_episode_without_counting_stop(config, tmp_path):
    from cognitive_discovery.alignment_diagnosis.episodes import run_episode

    row = factorial(config, phase="pilot")[0]
    result = run_episode(config, row, lambda state, seed: response("invalid"), tmp_path)
    assert result["episode_end_reason"] == "invalid_response"
    assert result["actual_choice"] is None
    assert result["choice_continue"] is None


def test_gpu_requests_55_minutes_without_explicit_qos():
    script = (ROOT / "slurm/run_alignment_diagnosis_gpu.slurm").read_text()
    assert "#SBATCH --time=00:55:00" in script
    assert "#SBATCH --qos" not in script


def test_resume_rejects_changed_shard_count(config, tmp_path):
    class Stopper(SyntheticParticipant):
        def decision_for_row(self, row, *, seed):
            result = super().decision_for_row(row, seed=seed)
            if row["trial_type"] == "choice":
                result["raw_response"] = "C"
            return result

    output = tmp_path / "shard-count"
    prepare(config, output, pilot=True, synthetic="integrated")
    collect(output, "main", execute=True, participant=Stopper(config), shard_count=2)
    with pytest.raises(ValueError, match="shard count"):
        collect(output, "main", shard_count=3)


def test_sharded_episode_workflow_preserves_twins_and_reports_timing(config, tmp_path):
    config["analysis"]["recovery_replicates"] = 2
    config["analysis"]["bootstrap_replicates"] = 10

    class StoppingParticipant(SyntheticParticipant):
        def decision_for_row(self, row, *, seed):
            result = super().decision_for_row(row, seed=seed)
            if row["trial_type"] == "choice":
                result["raw_response"] = "C"
            return result

    output = tmp_path / "episodes"
    prepare(config, output, pilot=True, synthetic="integrated")
    for phase in ("main", "calibration"):
        for shard in (0, 1):
            collect(
                output,
                phase,
                execute=True,
                participant=StoppingParticipant(config),
                shard_index=shard,
                shard_count=2,
            )
        owners = {}
        for index, path in enumerate(observation_paths(output, phase)):
            for row in records(path):
                owners.setdefault(row["pair_id"], set()).add(index)
        assert all(len(owner) == 1 for owner in owners.values())
    audit = json.loads((output / "counterbalance_audit.json").read_text())
    assert audit["complete_crossing"] and audit["cells"] == 24
    freeze_predictions(output)
    for shard in (0, 1):
        collect(
            output,
            "intervention",
            execute=True,
            participant=StoppingParticipant(config),
            shard_index=shard,
            shard_count=2,
        )
    result = analyze(output)
    assert result["episodes"]["episodes"] == 192
    assert result["episodes"]["decision_limit"] == 100
    assert (output / "episode_summary.csv").exists()
    assert (output / "episode_conditions.csv").exists()
    assert set(
        __import__("pandas")
        .read_csv(output / "condition_summary.csv")
        .scheduled_correction_decision
    ) == {9, 50, 90}
