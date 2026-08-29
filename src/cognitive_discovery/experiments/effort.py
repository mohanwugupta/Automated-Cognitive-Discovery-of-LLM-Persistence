from .base import BaseTaskRenderer


class EffortRenderer(BaseTaskRenderer):
    task_family = "effort"
    scenario = (
        "You can complete another unit of increasingly demanding work toward the current reward, or "
        "end the work sequence and take an outside option."
    )
    continue_text = "perform one more WORK unit"
    disengage_text = "QUIT the work sequence"

    def concrete_context(self, condition):
        factors = condition.semantic_factors
        reward = {"low": 5, "medium": 10, "high": 17}[factors["continuation_value"]]
        cost = {"low": 1, "medium": 4, "high": 8}[factors["continuation_cost"]]
        outside = {"low": 1, "medium": 6, "high": 12}[factors["disengagement_value"]]
        remaining = {"negative": 7, "neutral": 4, "positive": 2}[factors["progress_evidence"]]
        chance = {"low": 30, "medium": 60, "high": 90}[factors["success_evidence"]]
        return (
            f"The next reward pays {reward} points, needs about {remaining} more work units, and has a "
            f"{chance}% completion chance. Each unit costs {cost} points. Quitting pays {outside} points."
        )
