from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
import math
import re


class BaseParticipant(ABC):
    model_id = "abstract"

    @abstractmethod
    def binary_decision(self, messages, labels, *, positive_label) -> dict:
        """Return probabilities oriented toward the supplied semantic-positive label."""


class DeterministicParticipant(BaseParticipant):
    """Model-free plumbing participant; never use its output as scientific data."""

    def __init__(self, model_id: str = "deterministic/model-free-smoke"):
        self.model_id = model_id

    def binary_decision(self, messages, labels, *, positive_label) -> dict:
        labels = tuple(labels)
        negative_label = next(label for label in labels if label != positive_label)
        canonical_messages = [dict(message) for message in messages]
        # Canonicalize nuisance labels so mapping reversal tests semantics rather
        # than a deliberately random response-token hash.
        for message in canonical_messages:
            content = message.get("content", "")
            for label in labels:
                content = re.sub(rf"\b{re.escape(label)}\b", "<LABEL>", content)
            message["content"] = content
        payload = json.dumps(canonical_messages, sort_keys=True)
        unit = int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        logit = -1.5 + 3.0 * unit
        probability = 1.0 / (1.0 + math.exp(-logit))
        return {
            "positive_label": positive_label,
            "negative_label": negative_label,
            "p_positive": probability,
            "p_negative": 1.0 - probability,
            "choice_logit": logit,
            "p_action_mass_raw": 1.0,
            "top_token_is_action": True,
        }
