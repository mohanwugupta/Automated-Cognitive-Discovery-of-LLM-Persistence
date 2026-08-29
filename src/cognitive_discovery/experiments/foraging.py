from .base import BaseTaskRenderer


class ForagingRenderer(BaseTaskRenderer):
    task_family = "foraging"
    scenario = (
        "You are gathering food from a patch that can change in quality. You may search the current "
        "patch again or travel to the outside patch."
    )
    continue_text = "SEARCH the current patch again"
    disengage_text = "LEAVE for the outside patch"

    def concrete_context(self, condition):
        factors = condition.semantic_factors
        food = {"low": 3, "medium": 7, "high": 12}[factors["continuation_value"]]
        chance = {"low": 20, "medium": 50, "high": 80}[factors["success_evidence"]]
        cost = {"low": 1, "medium": 3, "high": 5}[factors["continuation_cost"]]
        outside = {"low": 1, "medium": 5, "high": 9}[factors["disengagement_value"]]
        trend = {
            "negative": "recent searches depleted the patch",
            "neutral": "recent searches left its estimated quality unchanged",
            "positive": "recent searches revealed signs of renewal",
        }[factors["progress_evidence"]]
        return (
            f"Food here is worth {food} points, has an estimated {chance}% chance per search, and a "
            f"search costs {cost} points; {trend}. Traveling to the outside patch is worth "
            f"{outside} points."
        )
