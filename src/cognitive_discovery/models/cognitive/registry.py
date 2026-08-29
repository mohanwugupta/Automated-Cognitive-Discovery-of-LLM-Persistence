"""Grounded, modular operationalizations of the PRD hypothesis bank."""

from __future__ import annotations

from dataclasses import dataclass


IMMEDIATE_FEATURES = (
    "factor_continuation_value",
    "factor_disengagement_value",
    "factor_continuation_cost",
    "factor_progress_evidence",
    "factor_success_evidence",
    "factor_uncertainty",
    "factor_prior_investment",
    "factor_controllability",
    "factor_environmental_stability",
    "factor_goal_continuity",
)


@dataclass(frozen=True)
class CognitiveModelSpec:
    name: str
    hypothesis: str
    features: tuple[str, ...]
    description: str


COGNITIVE_MODELS = {
    "intercept": CognitiveModelSpec("intercept", "null", (), "Task-agnostic mean policy."),
    "immediate_state": CognitiveModelSpec(
        "immediate_state", "baseline", IMMEDIATE_FEATURES, "All observable current-state main effects."
    ),
    "dynamic_reevaluation": CognitiveModelSpec(
        "dynamic_reevaluation",
        "H1",
        (
            "continuation_advantage",
            "factor_progress_evidence",
            "factor_success_evidence",
            "factor_uncertainty",
        ),
        "Prospective continue-versus-disengage evidence is recomputed at each decision.",
    ),
    "choice_perseveration": CognitiveModelSpec(
        "choice_perseveration",
        "H2",
        (*IMMEDIATE_FEATURES, "history_action_1", "history_action_kernel"),
        "Recent semantic actions directly bias repetition.",
    ),
    "outcome_history": CognitiveModelSpec(
        "outcome_history",
        "H3",
        (*IMMEDIATE_FEATURES, "history_outcome_1", "history_outcome_kernel"),
        "Recent successes and failures are integrated.",
    ),
    "dual_history": CognitiveModelSpec(
        "dual_history",
        "H2+H3",
        (
            *IMMEDIATE_FEATURES,
            "history_action_1",
            "history_action_kernel",
            "history_outcome_1",
            "history_outcome_kernel",
        ),
        "Action and outcome traces contribute separately.",
    ),
    "latent_motivation": CognitiveModelSpec(
        "latent_motivation",
        "H4",
        (
            "latent_state",
            "factor_continuation_value",
            "factor_continuation_cost",
            "factor_progress_evidence",
        ),
        "A slowly filtered commitment state summarizes current and historical evidence.",
    ),
    "latent_context": CognitiveModelSpec(
        "latent_context",
        "H5",
        (
            *IMMEDIATE_FEATURES,
            "context_relevant_outcome_kernel",
            "context_relevant_outcome_kernel*factor_goal_continuity",
            "context_relevant_outcome_kernel*factor_environmental_stability",
        ),
        "History is weighted by inferred continuity and environmental stability.",
    ),
    "option_termination": CognitiveModelSpec(
        "option_termination",
        "H6",
        (
            "continuation_advantage",
            "factor_progress_evidence",
            "factor_goal_continuity",
            "history_action_kernel",
        ),
        "Disengagement is a state-dependent termination policy for an extended option.",
    ),
    "task_set_reinstatement": CognitiveModelSpec(
        "task_set_reinstatement",
        "H7",
        (
            *IMMEDIATE_FEATURES,
            "factor_goal_continuity",
            "history_action_kernel",
            "history_action_kernel*factor_goal_continuity",
        ),
        "Repeated context reinstates a previous task policy.",
    ),
    "meta_control": CognitiveModelSpec(
        "meta_control",
        "H8",
        (
            "factor_continuation_value",
            "factor_continuation_cost",
            "factor_controllability",
            "factor_uncertainty",
            "factor_success_evidence",
            "factor_continuation_value*factor_controllability",
            "factor_continuation_cost*factor_uncertainty",
        ),
        "Expected benefit of control is traded against control cost.",
    ),
}


GENERIC_SEQUENTIAL_MODEL = CognitiveModelSpec(
    "generic_sequential_choice",
    "H9",
    (
        "history_action_1",
        "history_action_kernel",
        "history_outcome_1",
        "history_outcome_kernel",
        "factor_environmental_stability",
    ),
    "Ordinary history-sensitive sequential choice without persistence-specific constructs.",
)
COGNITIVE_MODELS[GENERIC_SEQUENTIAL_MODEL.name] = GENERIC_SEQUENTIAL_MODEL
