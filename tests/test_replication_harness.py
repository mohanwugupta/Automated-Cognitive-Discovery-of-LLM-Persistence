from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from cognitive_discovery.replication.adapters import (
    ReplicationModelAdapter,
    adapter_registry,
    resolve_relative_layers,
)
from cognitive_discovery.replication.config import (
    ReplicationConfigError,
    load_replication_config,
)
from cognitive_discovery.replication.controls import (
    EndpointMismatchError,
    audit_informative_target_shuffle,
    require_cognitive_endpoint,
)
from cognitive_discovery.replication.frozen_replay import replay_llama_phase_a
from cognitive_discovery.replication.harness import initialize_replication, plan_replication
from cognitive_discovery.replication.provenance import (
    ReplicationProvenanceError,
    validate_replication_provenance,
)
from cognitive_discovery.replication.report import write_replication_report
from cognitive_discovery.replication.splits import (
    assign_behavior_splits,
    assign_neural_splits,
    split_manifest_hash,
)
from cognitive_discovery.replication.state import ReplicationRunState, StageTransitionError
from cognitive_discovery.replication.workflow import AdapterParticipant


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/replication/default.yaml"
LLAMA_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


def test_adapter_contract_and_family_registry_are_model_agnostic():
    assert ReplicationModelAdapter.__abstractmethods__ == {
        "load_model",
        "load_tokenizer",
        "render_chat",
        "validate_response_tokens",
        "get_response_logits",
        "num_layers",
        "get_residual_state",
        "run_with_residual_intervention",
        "eos_token_ids",
    }
    assert set(adapter_registry.names()) == {"qwen", "llama", "gemma", "mistral"}
    for name in adapter_registry.names():
        adapter = adapter_registry.create(
            name,
            model_id=f"example/{name}",
            revision="a" * 40,
            local_files_only=True,
        )
        assert isinstance(adapter, ReplicationModelAdapter)
        assert adapter.adapter_name == name
        assert adapter.model is None
    with pytest.raises(ValueError, match="unknown replication adapter"):
        adapter_registry.create("qwen_or_llama_guess", model_id="x", revision="a" * 40)


def test_relative_depth_grid_is_not_qwen_layer_specific():
    assert resolve_relative_layers(32, [0.25, 0.5, 0.75, 0.9]) == [8, 16, 24, 28]
    assert resolve_relative_layers(40, [0.25, 0.5, 0.75, 0.9]) == [10, 20, 30, 36]
    with pytest.raises(ValueError, match="strictly between"):
        resolve_relative_layers(32, [0.0, 0.5])


def test_default_config_freezes_core_endpoint_grid_tasks_and_gates():
    config = load_replication_config(
        CONFIG,
        model_id="google/gemma-2-9b-it",
        revision="b" * 40,
        adapter="gemma",
    )
    assert config["endpoint_id"] == "cognitive_counterfactual_recovery"
    assert config["metric_id"] == "global_cfr_v1"
    assert config["mechanism"]["relative_depths"] == [0.25, 0.5, 0.75, 0.9]
    assert config["mechanism"]["ranks"] == [2, 8]
    assert config["mechanism"]["fitting_tasks"] == [
        "bandit",
        "debugging",
        "foraging",
        "solvability",
    ]
    assert config["mechanism"]["holdout_tasks"] == [
        "effort",
        "information_sampling",
        "waiting",
    ]
    assert config["behavior"]["contextual_conditions"] == 420
    assert config["optional"]["ood"]["requires_stage"] == "specificity"
    assert config["optional"]["ood"]["enabled"] is False
    assert config["specificity"]["maximum_response_mapping_cfr_gap"] == 0.25

    with pytest.raises(ReplicationConfigError, match="natural-effect"):
        load_replication_config(
            CONFIG,
            model_id="google/gemma-2-9b-it",
            revision="b" * 40,
            adapter="gemma",
            overrides={"endpoint_id": "natural_effect_recovery"},
        )
    with pytest.raises(ReplicationConfigError, match="immutable"):
        load_replication_config(
            CONFIG,
            model_id="google/gemma-2-9b-it",
            revision="main",
            adapter="gemma",
        )


def test_behavior_and_neural_splits_are_deterministic_group_safe_and_hashed():
    behavior = pd.DataFrame(
        [
            {"semantic_group": f"{task}:{index}", "task_family": task, "mapping": mapping}
            for task in ("bandit", "effort")
            for index in range(20)
            for mapping in ("continue_first", "continue_second")
        ]
    )
    left = assign_behavior_splits(
        behavior, group_column="semantic_group", task_column="task_family", seed=1001
    )
    right = assign_behavior_splits(
        behavior.sample(frac=1, random_state=5),
        group_column="semantic_group",
        task_column="task_family",
        seed=1001,
    )
    assert left.groupby("semantic_group").behavior_split.nunique().max() == 1
    assert split_manifest_hash(left, "semantic_group", "behavior_split") == split_manifest_hash(
        right, "semantic_group", "behavior_split"
    )
    assert set(left.behavior_split) == {"behavior_train", "behavior_selection", "behavior_test"}

    pairs = pd.DataFrame(
        [
            {"pair_id": f"{task}:{index}", "task_family": task}
            for task in (
                "bandit",
                "debugging",
                "foraging",
                "solvability",
                "effort",
                "information_sampling",
                "waiting",
            )
            for index in range(20)
        ]
    )
    neural = assign_neural_splits(
        pairs,
        group_column="pair_id",
        task_column="task_family",
        fitting_tasks={"bandit", "debugging", "foraging", "solvability"},
        holdout_tasks={"effort", "information_sampling", "waiting"},
        seed=2001,
    )
    assert set(neural[neural.task_family == "waiting"].neural_split) == {
        "neural_task_holdout"
    }
    assert set(neural[neural.task_family == "bandit"].neural_split) == {
        "neural_train",
        "neural_selection",
        "neural_test",
    }


def test_controls_cannot_switch_endpoint_and_target_preserving_shuffle_is_uninformative():
    require_cognitive_endpoint("cognitive_counterfactual_recovery")
    with pytest.raises(EndpointMismatchError, match="same cognitive"):
        require_cognitive_endpoint("natural_effect_recovery")
    unchanged = pd.DataFrame({"target": [1.0, 2.0], "shuffled_target": [1.0, 2.0]})
    changed = unchanged.assign(shuffled_target=[2.0, 1.0])
    assert audit_informative_target_shuffle(unchanged).status == "uninformative_control"
    assert audit_informative_target_shuffle(changed).status == "informative_control"


def test_phase_a_replays_only_approved_llama_cognitive_results():
    result = replay_llama_phase_a(ROOT)
    assert result["model"]["revision"] == LLAMA_REVISION
    assert result["interface"] == {
        "selected": "question",
        "passed": True,
        "approved_task_count": 7,
    }
    assert result["behavior"]["selected_model"] == "dual_history"
    assert result["behavior"]["immediate_state_test_r2"] == pytest.approx(0.8212009718)
    assert result["behavior"]["dual_history_test_r2"] == pytest.approx(0.8353870157)
    assert result["mechanism"]["layer"] == 24
    assert result["mechanism"]["rank"] == 2
    assert result["mechanism"]["endpoint_id"] == "cognitive_counterfactual_recovery"
    assert result["mechanism"]["neural_test_global_cfr"] == pytest.approx(0.6488862151)
    assert result["mechanism"]["neural_task_holdout_global_cfr"] == pytest.approx(
        0.0111618865
    )
    assert result["specificity"]["status"] == "not_run"
    assert "natural" not in json.dumps(result).lower()


def test_stage_state_is_restartable_and_enforces_measurement_gate():
    state = ReplicationRunState.new("test-run")
    with pytest.raises(StageTransitionError, match="dependency"):
        state.start("behavior")
    state.start("interface")
    state.complete("interface", outcome="measurement_failure")
    assert state.replication_status == "measurement_failure"
    with pytest.raises(StageTransitionError, match="measurement failure"):
        state.start("behavior")

    resumed = ReplicationRunState.from_dict(state.to_dict())
    assert resumed.to_dict() == state.to_dict()

    state.start("report")
    state.complete("report", outcome="complete")
    assert state.replication_status == "measurement_failure"


def test_adapter_participant_does_not_fabricate_full_vocabulary_validity_fields():
    class FakeAdapter:
        model_id = "fake/model"
        revision = "a" * 40

        def get_response_logits(self, messages, labels):
            del messages
            return {labels[0]: 2.0, labels[1]: -1.0}

    participant = AdapterParticipant(
        FakeAdapter(), {"id": "x_y", "labels": ["X", "Y"]}
    )
    result = participant.binary_decision(
        [{"role": "user", "content": "Choose one:\nX = continue\nY = stop"}],
        ["X", "Y"],
        positive_label="X",
    )
    assert "top_token_is_action" not in result
    assert "p_action_mass_raw" not in result


def test_initialization_is_dry_run_safe_and_writes_complete_initial_provenance(tmp_path):
    output = tmp_path / "replication"
    plan = plan_replication(
        root=ROOT,
        config_path=CONFIG,
        model_id="google/gemma-2-9b-it",
        revision="c" * 40,
        adapter="gemma",
        output=output,
    )
    assert plan["mode"] == "dry-run"
    assert plan["endpoint_id"] == "cognitive_counterfactual_recovery"
    assert not output.exists()

    initialized = initialize_replication(
        root=ROOT,
        config_path=CONFIG,
        model_id="google/gemma-2-9b-it",
        revision="c" * 40,
        adapter="gemma",
        output=output,
    )
    assert initialized["output"] == str(output)
    provenance = json.loads((output / "provenance.json").read_text())
    validate_replication_provenance(provenance, required_for_stage="interface")
    assert provenance["model"]["revision"] == "c" * 40
    assert provenance["tokenizer"]["revision"] == "c" * 40
    assert provenance["adapter"]["version"]
    assert len(provenance["environment"]["lockfile_sha256"]) == 64
    assert provenance["selected_controller_hash"] == "pending"
    with pytest.raises(ReplicationProvenanceError, match="frozen behavioral"):
        validate_replication_provenance(provenance, required_for_stage="counterfactuals")
    with pytest.raises(ReplicationProvenanceError, match="frozen behavioral"):
        validate_replication_provenance(provenance, required_for_stage="mechanism")
    with pytest.raises(FileExistsError, match="already exists"):
        initialize_replication(
            root=ROOT,
            config_path=CONFIG,
            model_id="google/gemma-2-9b-it",
            revision="c" * 40,
            adapter="gemma",
            output=output,
        )


def test_report_keeps_components_separate_and_records_exact_hashes(tmp_path):
    replay = replay_llama_phase_a(ROOT)
    path = write_replication_report(
        tmp_path / "REPLICATION_REPORT.md",
        replay,
        provenance={"schema_version": "replication-provenance-v1", "config_hash": "a" * 64},
    )
    text = path.read_text()
    assert "Measurement interface:" in text
    assert "Behavioral history-sensitive structure:" in text
    assert "Held-out task-family generalization:" in text
    assert "Far-OOD continuation generalization:" in text
    assert "Overall score" not in text
    assert "natural_effect_recovery" not in text
    assert "a" * 64 in text


def test_cli_supports_new_dry_run_and_frozen_llama_regression(tmp_path):
    env = {"PYTHONPATH": str(ROOT / "src")}
    new_output = tmp_path / "new-model"
    dry = subprocess.run(
        [
            sys.executable,
            "-m",
            "cognitive_discovery.replicate_model",
            "--model",
            "google/gemma-2-9b-it",
            "--revision",
            "d" * 40,
            "--adapter",
            "gemma",
            "--config",
            str(CONFIG),
            "--output",
            str(new_output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    assert not new_output.exists()
    assert json.loads(dry.stdout)["mode"] == "dry-run"

    replay_output = tmp_path / "llama-replay"
    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "cognitive_discovery.replicate_model",
            "--frozen-llama-replay",
            "--output",
            str(replay_output),
            "--execute",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert replay.returncode == 0, replay.stderr
    assert (replay_output / "REPLICATION_REPORT.md").is_file()
    assert (replay_output / "phase_a_replay.json").is_file()
