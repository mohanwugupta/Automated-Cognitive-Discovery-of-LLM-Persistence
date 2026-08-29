from .base import BaseTaskRenderer


class BanditRenderer(BaseTaskRenderer):
    task_family = "bandit"
    scenario = (
        "You are repeatedly choosing whether to keep playing the current reward source or leave it "
        "for another available opportunity."
    )
    continue_text = "PLAY the current reward source once more"
    disengage_text = "LEAVE for the alternative opportunity"

    def concrete_context(self, condition):
        factors = condition.semantic_factors
        reward = {"low": 4, "medium": 8, "high": 14}[factors["continuation_value"]]
        success = {"low": 25, "medium": 50, "high": 75}[factors["success_evidence"]]
        fee = {"low": 1, "medium": 3, "high": 6}[factors["continuation_cost"]]
        alternative = {"low": 1, "medium": 5, "high": 10}[factors["disengagement_value"]]
        return (
            f"A successful play of the current source pays {reward} points and the current evidence "
            f"suggests a {success}% success chance. One play costs {fee} points. Leaving secures an "
            f"alternative worth {alternative} points."
        )
