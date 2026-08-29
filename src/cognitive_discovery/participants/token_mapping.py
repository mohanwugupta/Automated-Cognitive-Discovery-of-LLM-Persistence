"""Semantic binary-token validation and logit extraction ported from the sprint runner."""

from __future__ import annotations

from collections.abc import Iterable
import math


def verify_choice_tokens(tokenizer, labels: Iterable[str]) -> dict[str, int]:
    labels = tuple(labels)
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError("binary choice requires two distinct labels")
    token_ids = {}
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"choice {label!r} is not a single token: {encoded}")
        token_ids[label] = int(encoded[0])
    if len(set(token_ids.values())) != 2:
        raise ValueError("choice labels do not map to distinct tokens")
    return token_ids


def verify_chat_choice_tokens(tokenizer, messages, labels: Iterable[str]) -> dict[str, int]:
    labels = tuple(labels)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    token_ids = {}
    for label in labels:
        completed = tokenizer.encode(prompt + label, add_special_tokens=False)
        if completed[: len(prefix)] != prefix or len(completed) != len(prefix) + 1:
            raise ValueError(
                f"choice {label!r} is not a clean one-token chat continuation"
            )
        token_ids[label] = int(completed[-1])
    if len(set(token_ids.values())) != 2:
        raise ValueError("chat choice labels do not map to distinct tokens")
    return token_ids


def binary_choice_metrics(logits: dict[str, float], positive_label: str) -> dict[str, float]:
    if len(logits) != 2 or positive_label not in logits:
        raise ValueError("binary logits must include the semantic-positive label")
    negative_label = next(label for label in logits if label != positive_label)
    maximum = max(float(value) for value in logits.values())
    weights = {label: math.exp(float(value) - maximum) for label, value in logits.items()}
    total = sum(weights.values())
    return {
        "positive_label": positive_label,
        "negative_label": negative_label,
        "p_positive": weights[positive_label] / total,
        "p_negative": weights[negative_label] / total,
        "choice_logit": float(logits[positive_label]) - float(logits[negative_label]),
        **{f"logit_{label}": float(value) for label, value in logits.items()},
        **{f"p_{label}": value / total for label, value in weights.items()},
    }

