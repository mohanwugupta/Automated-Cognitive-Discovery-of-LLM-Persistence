from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from cognitive_discovery.reproduce import inspect_stage, replay_claim_records


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_drives_all_claim_navigation():
    records = replay_claim_records(ROOT)
    assert [record["claim_id"] for record in records] == [f"C{i:02d}" for i in range(1, 14)]
    assert all(record["frozen_value_reproduced"] for record in records)
    assert records[6]["endpoint_id"] == "natural_effect_recovery"
    assert records[7]["endpoint_id"] == "cognitive_counterfactual_recovery"
    assert records[6]["navigation"]["config_path"] == (
        "configs/canonical/qwen_confirmation_v2.yaml"
    )


def test_stage_inspection_uses_manifest_dependencies_and_hashes():
    record = inspect_stage(ROOT, "causal_specificity_v2")
    assert record["dependency_order"][-1] == "causal_specificity_v2"
    assert "causal_mech_v1" in record["dependency_order"]
    assert record["claims"] == ["C02", "C03"]
    assert len(record["verified_artifacts"]) == 3


def test_documented_module_commands_work_without_gpu_or_network():
    commands = [
        ["claims", "--json"],
        ["claim", "C07", "--json"],
        ["stage", "causal_specificity_v2", "--json"],
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    for arguments in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "cognitive_discovery.reproduce", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr
