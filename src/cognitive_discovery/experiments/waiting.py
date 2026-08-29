from .base import BaseTaskRenderer


class WaitingRenderer(BaseTaskRenderer):
    task_family = "waiting"
    scenario = (
        "A delayed reward may arrive. Each decision represents one simulated time step; no real-time "
        "waiting is required."
    )
    continue_text = "WAIT one more simulated step"
    disengage_text = "QUIT waiting and take the available option"

    def concrete_context(self, condition):
        factors = condition.semantic_factors
        reward = {"low": 4, "medium": 9, "high": 15}[factors["continuation_value"]]
        arrival = {"low": 15, "medium": 45, "high": 75}[factors["success_evidence"]]
        cost = {"low": 1, "medium": 3, "high": 6}[factors["continuation_cost"]]
        quit_payoff = {"low": 0, "medium": 4, "high": 9}[factors["disengagement_value"]]
        return (
            f"The delayed reward is {reward} points and currently has an estimated {arrival}% chance "
            f"of arriving in the next simulated step. Waiting costs {cost} points; quitting pays "
            f"{quit_payoff} points now."
        )
