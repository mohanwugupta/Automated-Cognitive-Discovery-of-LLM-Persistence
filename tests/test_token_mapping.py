import math

import pytest

from cognitive_discovery.participants.token_mapping import (
    binary_choice_metrics,
    verify_choice_tokens,
)


class Tokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, value, add_special_tokens=False):
        return self.mapping[value]


def test_semantic_logit_is_oriented_by_mapping():
    forward = binary_choice_metrics({"X": 3.0, "Y": 1.0}, positive_label="X")
    reverse = binary_choice_metrics({"X": 3.0, "Y": 1.0}, positive_label="Y")
    assert math.isclose(forward["choice_logit"], 2.0)
    assert math.isclose(reverse["choice_logit"], -2.0)
    assert math.isclose(forward["p_positive"], reverse["p_negative"])


def test_response_tokens_must_be_single_and_distinct():
    assert verify_choice_tokens(Tokenizer({"X": [4], "Y": [8]}), ("X", "Y")) == {
        "X": 4,
        "Y": 8,
    }
    with pytest.raises(ValueError, match="single token"):
        verify_choice_tokens(Tokenizer({"X": [4, 5], "Y": [8]}), ("X", "Y"))
    with pytest.raises(ValueError, match="distinct tokens"):
        verify_choice_tokens(Tokenizer({"X": [4], "Y": [4]}), ("X", "Y"))

