"""Restartable, dependency-gated replication stage state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


STAGE_DEPENDENCIES = {
    "interface": (),
    "behavior": ("interface",),
    "model_comparison": ("behavior",),
    "freeze_theory": ("model_comparison",),
    "counterfactuals": ("freeze_theory",),
    "mechanism": ("counterfactuals",),
    "generalization": ("mechanism",),
    "specificity": ("generalization",),
    "abstraction": ("specificity",),
    "ood": ("specificity",),
    "report": ("specificity",),
}


class StageTransitionError(RuntimeError):
    pass


@dataclass
class ReplicationRunState:
    run_id: str
    replication_status: str = "initialized"
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def new(cls, run_id: str) -> "ReplicationRunState":
        return cls(
            run_id=str(run_id),
            stages={
                name: {"status": "pending", "outcome": None, "artifacts": {}}
                for name in STAGE_DEPENDENCIES
            },
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReplicationRunState":
        if value.get("schema_version") != "replication-state-v1":
            raise StageTransitionError("unsupported replication state schema")
        state = cls(
            run_id=value["run_id"],
            replication_status=value["replication_status"],
            stages=deepcopy(value["stages"]),
        )
        if set(state.stages) != set(STAGE_DEPENDENCIES):
            raise StageTransitionError("replication state stage inventory is incomplete")
        return state

    @classmethod
    def load(cls, path: str | Path) -> "ReplicationRunState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "replication-state-v1",
            "run_id": self.run_id,
            "replication_status": self.replication_status,
            "stages": deepcopy(self.stages),
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        return path

    def start(self, stage: str) -> None:
        if stage not in STAGE_DEPENDENCIES:
            raise StageTransitionError(f"unknown replication stage {stage!r}")
        measurement_failure_report = (
            self.replication_status == "measurement_failure" and stage == "report"
        )
        if self.replication_status == "measurement_failure" and stage != "report":
            raise StageTransitionError("cannot continue after a measurement failure")
        current = self.stages[stage]["status"]
        if current == "complete":
            raise StageTransitionError(f"stage {stage} is already complete")
        if current == "running":
            raise StageTransitionError(f"stage {stage} is already running")
        missing = (
            []
            if measurement_failure_report
            else [
                dependency
                for dependency in STAGE_DEPENDENCIES[stage]
                if self.stages[dependency]["status"] != "complete"
            ]
        )
        if missing:
            raise StageTransitionError(f"stage {stage} dependency is incomplete: {missing}")
        self.stages[stage]["status"] = "running"
        if not measurement_failure_report:
            self.replication_status = "running"

    def complete(self, stage: str, *, outcome: str, artifacts=None) -> None:
        if self.stages.get(stage, {}).get("status") != "running":
            raise StageTransitionError(f"stage {stage} must be running before completion")
        self.stages[stage] = {
            "status": "complete",
            "outcome": str(outcome),
            "artifacts": dict(artifacts or {}),
        }
        if stage == "interface" and outcome == "measurement_failure":
            self.replication_status = "measurement_failure"
        elif stage == "report":
            if self.replication_status != "measurement_failure":
                self.replication_status = "complete"
        else:
            self.replication_status = "running"

    def fail(self, stage: str, *, reason: str) -> None:
        if stage not in self.stages:
            raise StageTransitionError(f"unknown replication stage {stage!r}")
        self.stages[stage] = {
            "status": "blocked",
            "outcome": str(reason),
            "artifacts": {},
        }
        self.replication_status = "blocked"
