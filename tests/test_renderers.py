from dataclasses import replace

import pytest

from cognitive_discovery.design.sampling import compile_design
from cognitive_discovery.experiments.registry import get_renderer


@pytest.mark.parametrize(
    "task",
    [
        "bandit",
        "foraging",
        "solvability",
        "information_sampling",
        "waiting",
        "effort",
        "debugging",
    ],
)
def test_every_required_task_renders_natural_binary_choice(discovery_config, task):
    row = next(
        item
        for item in compile_design(discovery_config, n_conditions=70, seed=8)
        if item.task_family == task
    )
    trial = get_renderer(task).render(row)
    assert trial.messages[-1]["role"] == "user"
    prompt = trial.messages[-1]["content"]
    assert "measures persistence" not in prompt.lower()
    assert row.response_mapping.continue_label in prompt
    assert row.response_mapping.disengage_label in prompt
    assert trial.semantic_state["environment_seed"] == row.environment_seed


def test_paired_label_mapping_preserves_latent_semantics(discovery_config):
    left, right = compile_design(discovery_config, n_conditions=1, seed=1)
    left_trial = get_renderer(left.task_family).render(left)
    right_trial = get_renderer(right.task_family).render(right)
    assert left_trial.semantic_state == right_trial.semantic_state
    assert left_trial.prompt_hash != right_trial.prompt_hash


def test_changing_one_factor_preserves_other_semantic_fields(discovery_config):
    row = next(
        item
        for item in compile_design(discovery_config, n_conditions=140, seed=4)[::2]
        if item.factor_available["continuation_cost"]
    )
    changed_factors = dict(row.semantic_factors)
    changed_factors["continuation_cost"] = (
        "high" if changed_factors["continuation_cost"] != "high" else "low"
    )
    changed = replace(row, semantic_factors=changed_factors)
    before = get_renderer(row.task_family).render(row).semantic_state
    after = get_renderer(changed.task_family).render(changed).semantic_state
    differing = {key for key in before if before[key] != after[key]}
    assert differing <= {"continuation_cost", "continuation_advantage"}

