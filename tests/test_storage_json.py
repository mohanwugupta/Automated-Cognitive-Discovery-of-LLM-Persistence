"""JSON artifacts must retain nulls under current and Pandas 3 string inference."""

import json

import pandas as pd
import pytest

from cognitive_discovery.data.storage import write_records


@pytest.mark.parametrize("infer_strings", [False, True])
def test_json_writer_emits_null_for_nullable_fields(tmp_path, infer_strings):
    rows = [
        {
            "expected_action": None,
            "optional_count": None,
            "history": [1, -1, None],
            "probability": 0.12345678901234567,
        },
        {
            "expected_action": "C",
            "optional_count": 1,
            "history": [None, 1],
            "probability": 0.8765432109876543,
        },
    ]
    with pd.option_context("future.infer_string", infer_strings):
        path = write_records(rows, tmp_path / "rows.jsonl")

    def reject_constant(value):
        raise ValueError(f"non-JSON numeric constant: {value}")

    saved = [
        json.loads(line, parse_constant=reject_constant)
        for line in path.read_text().splitlines()
    ]
    assert saved[0]["expected_action"] is None
    assert saved[0]["optional_count"] is None
    for observed, original in zip(saved, rows):
        assert observed["history"] == original["history"]
        assert observed["probability"] == original["probability"]
