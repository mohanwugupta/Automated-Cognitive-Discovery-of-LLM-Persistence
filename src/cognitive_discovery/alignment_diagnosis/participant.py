"""Three-action probabilities and unconstrained sampled responses via existing Qwen loader."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from cognitive_discovery.participants.qwen import QwenParticipant
from cognitive_discovery.participants.token_mapping import verify_chat_action_tokens


def parse_response(response, labels):
    value = response.strip()
    return value if value in labels else None


class BanditQwenParticipant(QwenParticipant):
    def decision(self, messages, labels, *, seed):
        import torch

        token_ids = verify_chat_action_tokens(self.tokenizer, messages, labels)
        inputs = self._tokenize(messages)
        with torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=False, use_cache=False)
            if getattr(output, "hidden_states", None) is not None:
                raise RuntimeError(
                    "behavioral collection must not retain hidden states"
                )
            logits = output.logits[0, -1].float()
            selected = [float(logits[token_ids[label]].cpu()) for label in labels]
            action_probs = softmax(selected)
            mass = float(
                torch.exp(
                    torch.logsumexp(logits[list(token_ids.values())], 0)
                    - torch.logsumexp(logits, 0)
                ).cpu()
            )
            devices = [logits.device.index] if logits.device.type == "cuda" else []
            # Seed per trial so resuming does not change subsequent choices.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                generated = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    typical_p=1.0,
                    min_p=None,
                    repetition_penalty=1.0,
                    max_new_tokens=8,
                    use_cache=True,
                    return_dict_in_generate=False,
                    output_scores=False,
                    output_hidden_states=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        raw = self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return {
            "raw_response": raw,
            "action_logits": dict(zip(labels, selected)),
            "action_probabilities": dict(zip(labels, map(float, action_probs))),
            "p_action_mass_raw": mass,
            "top_token_is_action": int(logits.argmax()) in token_ids.values(),
        }


class SyntheticParticipant:
    """Explicitly synthetic smoke data; never a substitute for Qwen observations."""

    model_id = "synthetic/bandit-diagnosis"
    revision = "synthetic"

    def __init__(self, config, generator="integrated"):
        self.config, self.generator = config, generator

    def decision_for_row(self, row, *, seed):
        from .models import predict

        rng = np.random.default_rng(seed)
        if row["trial_type"] == "comprehension":
            labels = ("X", "Y")
            p = 0.99 if row["correction"] else 0.01
            probabilities = [1 - p, p]
        else:
            labels = ("A", "B", "C")
            p = float(
                predict(
                    [row],
                    self.generator,
                    [-0.5, 0.7, 0.3],
                    2.0 if self.generator == "integrated" else 0.8,
                    self.config["analysis"]["epsilon"],
                )[0]
            )
            if row["trial_type"] == "isolated" and row["correction"]:
                p = 0.001
            probabilities = [p / 2, p / 2, 1 - p]
        return {
            "raw_response": str(rng.choice(labels, p=probabilities)),
            "action_logits": dict(zip(labels, np.log(probabilities).tolist())),
            "action_probabilities": dict(zip(labels, probabilities)),
            "p_action_mass_raw": 1.0,
            "top_token_is_action": True,
        }
