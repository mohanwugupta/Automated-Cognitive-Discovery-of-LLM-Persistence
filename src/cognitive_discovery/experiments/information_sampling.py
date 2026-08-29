from .base import BaseTaskRenderer


class InformationSamplingRenderer(BaseTaskRenderer):
    task_family = "information_sampling"
    scenario = (
        "You must identify which of two sources is active. You may pay to reveal another noisy signal "
        "or stop gathering information and decide now."
    )
    continue_text = "SAMPLE one more signal"
    disengage_text = "STOP sampling and decide now"

    def concrete_context(self, condition):
        factors = condition.semantic_factors
        penalty = {"low": 4, "medium": 9, "high": 15}[factors["continuation_value"]]
        confidence = {"low": 55, "medium": 70, "high": 88}[factors["success_evidence"]]
        cost = {"low": 1, "medium": 3, "high": 6}[factors["continuation_cost"]]
        decision_value = {"low": 1, "medium": 5, "high": 10}[factors["disengagement_value"]]
        progress = {
            "negative": "the newest signal conflicts with the earlier majority",
            "neutral": "the newest signal leaves the evidence tied",
            "positive": "the newest signal agrees with the earlier majority",
        }[factors["progress_evidence"]]
        return (
            f"The current best answer has {confidence}% support and an error costs {penalty} points. "
            f"Another signal costs {cost} points; {progress}. Deciding now preserves {decision_value} "
            f"points of remaining opportunity value."
        )
