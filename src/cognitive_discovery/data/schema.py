from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class Observation:
    design_id: str
    condition_id: str
    paired_condition_id: str
    task_family: str
    episode_id: str
    step: int
    factors: dict[str, str | None]
    factor_available: dict[str, bool]
    history_actions: tuple[str, ...]
    history_outcomes: tuple[int, ...]
    semantic_continue_token: str
    semantic_disengage_token: str
    response_mapping: str
    p_continue: float
    p_disengage: float
    persistence_logit: float
    sampled_action: str
    terminated: bool
    prompt_hash: str
    environment_seed: int
    model: str
    model_revision: str | None
    split: str
    sampling_strategy: str
    p_action_mass_raw: float | None = None
    top_token_is_action: bool | None = None

    def to_dict(self) -> dict:
        row = asdict(self)
        row["history_actions"] = json.dumps(self.history_actions)
        row["history_outcomes"] = json.dumps(self.history_outcomes)
        factors = row.pop("factors")
        available = row.pop("factor_available")
        row.update({f"factor_{name}": value for name, value in factors.items()})
        row.update(
            {f"factor_available_{name}": bool(value) for name, value in available.items()}
        )
        return row

