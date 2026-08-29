"""SweetBean-style boundary from abstract conditions to concrete interactions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from cognitive_discovery.ontology.task_schema import ConditionSpec


LEVEL_TEXT = {
    "continuation_value": {
        "low": "The payoff for completing the current course is small.",
        "medium": "The payoff for completing the current course is moderate.",
        "high": "The payoff for completing the current course is large.",
    },
    "disengagement_value": {
        "low": "The alternative available now has little value.",
        "medium": "The alternative available now has moderate value.",
        "high": "The alternative available now has high value.",
    },
    "continuation_cost": {
        "low": "Taking one more step has a small cost.",
        "medium": "Taking one more step has a moderate cost.",
        "high": "Taking one more step has a large cost.",
    },
    "progress_evidence": {
        "negative": "Recent evidence suggests movement away from completion.",
        "neutral": "Recent evidence does not indicate clear progress or decline.",
        "positive": "Recent evidence suggests movement toward completion.",
    },
    "success_evidence": {
        "low": "The available evidence makes eventual success unlikely.",
        "medium": "The available evidence gives an even chance of eventual success.",
        "high": "The available evidence makes eventual success likely.",
    },
    "uncertainty": {
        "low": "These estimates are reliable.",
        "high": "These estimates are uncertain.",
    },
    "prior_investment": {
        "low": "Little effort has already been spent.",
        "high": "A great deal of effort has already been spent; it cannot be recovered.",
    },
    "controllability": {
        "low": "Your next action has little influence over the outcome.",
        "high": "Your next action strongly influences the outcome.",
    },
    "environmental_stability": {
        "stable": "The environment has been stable, so earlier observations remain relevant.",
        "changing": "The environment has been changing, so earlier observations may be less relevant.",
    },
    "goal_continuity": {
        "same": "This is the same objective and policy as in the preceding decisions.",
        "new": "This is a new objective, though the decision format is similar.",
    },
}


@dataclass(frozen=True)
class RenderedTrial:
    condition_id: str
    paired_condition_id: str
    task_family: str
    messages: tuple[dict[str, str], ...]
    semantic_state: dict
    continue_label: str
    disengage_label: str

    @property
    def prompt_hash(self) -> str:
        payload = "\n".join(f"{message['role']}:{message['content']}" for message in self.messages)
        return hashlib.sha256(payload.encode()).hexdigest()


class BaseTaskRenderer:
    """A deterministic renderer whose only input is an abstract condition."""

    task_family = "abstract"
    scenario = "An ongoing course of action is available."
    continue_text = "continue the current course"
    disengage_text = "stop and take the alternative"

    def concrete_context(self, condition: ConditionSpec) -> str:
        return ""

    def semantic_state(self, condition: ConditionSpec) -> dict:
        return {
            "task_family": condition.task_family,
            "environment_seed": condition.environment_seed,
            **condition.semantic_factors,
            "continuation_advantage": condition.continuation_advantage,
            "history_length": condition.history.length,
            "history_valence": condition.history.valence,
            "history_actions": condition.history.actions,
            "history_outcomes": condition.history.outcomes,
        }

    def _factor_text(self, condition: ConditionSpec) -> str:
        return " ".join(
            LEVEL_TEXT[name][value]
            for name, value in condition.semantic_factors.items()
            if value is not None
        )

    def _history_text(self, condition: ConditionSpec) -> str:
        if condition.contextual_history is not None:
            from .contextual_history.renderers import contextual_history_text

            return contextual_history_text(condition)
        if condition.history.length == 0:
            return "There are no preceding decisions in this episode."
        return (
            f"The preceding semantic choices were: {condition.history.action_text}. "
            f"Their outcomes were: {condition.history.outcome_text}."
        )

    def render(self, condition: ConditionSpec) -> RenderedTrial:
        if condition.task_family != self.task_family:
            raise ValueError(
                f"{self.__class__.__name__} cannot render {condition.task_family}"
            )
        mapping = condition.response_mapping
        prompt = (
            f"{self.scenario}\n\n"
            f"{self.concrete_context(condition)}\n\n"
            f"{self._factor_text(condition)}\n\n"
            f"{self._history_text(condition)}\n\n"
            "Choose one:\n"
            f"{mapping.continue_label} = {self.continue_text}\n"
            f"{mapping.disengage_label} = {self.disengage_text}\n\n"
            f"Respond with only {mapping.labels[0]} or {mapping.labels[1]}."
        )
        return RenderedTrial(
            condition_id=condition.condition_id,
            paired_condition_id=condition.paired_condition_id,
            task_family=condition.task_family,
            messages=({"role": "user", "content": prompt},),
            semantic_state=self.semantic_state(condition),
            continue_label=mapping.continue_label,
            disengage_label=mapping.disengage_label,
        )
