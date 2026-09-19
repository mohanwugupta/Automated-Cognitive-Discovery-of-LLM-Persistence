from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_qwen_run_record_uses_pinned_followup_revision_and_fresh_inputs():
    record = yaml.safe_load((ROOT / "configs/replication/qwen_prospective_run.yaml").read_text())
    assert record["interpretation"] == "pipeline_self_replication"
    assert record["status"] == "ready_for_prospective_run"
    assert set(record["freshness"].values()) == {False}
    assert record["model"]["id"] == "Qwen/Qwen3.5-4B"
    assert record["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert record["model"]["tokenizer_revision"] == record["model"]["revision"]
