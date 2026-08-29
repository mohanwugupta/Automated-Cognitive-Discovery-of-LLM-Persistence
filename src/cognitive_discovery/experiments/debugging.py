from .base import BaseTaskRenderer


class DebuggingRenderer(BaseTaskRenderer):
    task_family = "debugging"
    scenario = (
        "One software bug blocks a project. Another repair attempt may fix it and a failed attempt can "
        "reveal a clue; alternatively you can abandon it for a fresh project."
    )
    continue_text = "DEBUG the same bug for one more attempt"
    disengage_text = "ABANDON this bug for the fresh project"

    def concrete_context(self, condition):
        factors = condition.semantic_factors
        reward = {"low": 6, "medium": 12, "high": 20}[factors["continuation_value"]]
        chance = {"low": 20, "medium": 50, "high": 80}[factors["success_evidence"]]
        cost = {"low": 1, "medium": 4, "high": 7}[factors["continuation_cost"]]
        restart = {"low": 1, "medium": 6, "high": 12}[factors["disengagement_value"]]
        clue = {
            "negative": "the newest diagnostic contradicted the working theory",
            "neutral": "the newest diagnostic was inconclusive",
            "positive": "the newest diagnostic narrowed the likely cause",
        }[factors["progress_evidence"]]
        return (
            f"Fixing the bug pays {reward} points. A repair attempt costs {cost} points and currently "
            f"has a {chance}% success chance; {clue}. Restarting elsewhere is worth {restart} points."
        )
