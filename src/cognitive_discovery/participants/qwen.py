"""Behavior-only Qwen participant; no hidden states are requested or retained."""

from __future__ import annotations

from collections.abc import Mapping

from .base import BaseParticipant
from .token_mapping import (
    binary_choice_metrics,
    verify_chat_choice_tokens,
    verify_choice_tokens,
)


class QwenParticipant(BaseParticipant):
    def __init__(self, model, tokenizer, model_id: str, revision: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision or getattr(getattr(model, "config", None), "_commit_hash", None)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        local_files_only: bool = True,
    ):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if getattr(config, "model_type", None) == "qwen3_5" and hasattr(config, "text_config"):
            try:
                from transformers import Qwen3_5ForConditionalGeneration

                model_class = Qwen3_5ForConditionalGeneration
            except ImportError:
                from transformers import AutoModelForMultimodalLM

                model_class = AutoModelForMultimodalLM
        else:
            model_class = AutoModelForCausalLM
        model = model_class.from_pretrained(
            model_name_or_path,
            config=config,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
            dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        model.eval()
        return cls(model, tokenizer, model_name_or_path, revision)

    def _tokenize(self, messages):
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        inputs = dict(encoded) if isinstance(encoded, Mapping) else {"input_ids": encoded}
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in inputs.items()}

    def binary_decision(self, messages, labels, *, positive_label) -> dict:
        import torch

        labels = tuple(labels)
        if hasattr(self.tokenizer, "apply_chat_template"):
            token_ids = verify_chat_choice_tokens(self.tokenizer, messages, labels)
        else:
            token_ids = verify_choice_tokens(self.tokenizer, labels)
        with torch.inference_mode():
            outputs = self.model(
                **self._tokenize(messages), output_hidden_states=False, use_cache=False
            )
        if getattr(outputs, "hidden_states", None) is not None:
            raise RuntimeError("behavior-only participant unexpectedly received hidden states")
        logits = outputs.logits[0, -1]
        selected = {
            label: float(logits[token_id].detach().float().cpu())
            for label, token_id in token_ids.items()
        }
        result = binary_choice_metrics(selected, positive_label)
        selected_ids = torch.tensor(list(token_ids.values()), device=logits.device)
        result["p_action_mass_raw"] = float(
            torch.exp(
                torch.logsumexp(logits[selected_ids].float(), dim=0)
                - torch.logsumexp(logits.float(), dim=0)
            ).cpu()
        )
        result["top_token_is_action"] = int(logits.argmax()) in token_ids.values()
        return result

