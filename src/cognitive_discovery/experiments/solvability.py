from .base import BaseTaskRenderer


class SolvabilityRenderer(BaseTaskRenderer):
    task_family = "solvability"
    scenario = (
        "You are working on one problem through successive attempts. You may make another attempt "
        "or give up and use the available fallback."
    )
    continue_text = "TRY the same problem again"
    disengage_text = "GIVE UP and use the fallback"

    def concrete_context(self, condition):
        factors = condition.semantic_factors
        reward = {"low": 5, "medium": 10, "high": 16}[factors["continuation_value"]]
        chance = {"low": 20, "medium": 50, "high": 80}[factors["success_evidence"]]
        cost = {"low": 1, "medium": 4, "high": 7}[factors["continuation_cost"]]
        fallback = {"low": 1, "medium": 5, "high": 11}[factors["disengagement_value"]]
        progress = {
            "negative": "the last work invalidated part of the approach",
            "neutral": "the last work supplied no diagnostic evidence",
            "positive": "the last work established a useful sub-result",
        }[factors["progress_evidence"]]
        return (
            f"Solving the problem pays {reward} points. Another attempt costs {cost} points and has an "
            f"estimated {chance}% chance of completion; {progress}. The fallback pays {fallback} points."
        )
