"""Architecture adapters for behavioral logits and residual interventions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
import math
from typing import Any


class ReplicationModelAdapter(ABC):
    """The only model-specific boundary used by the replication pipeline."""

    adapter_name = "abstract"
    adapter_version = "replication-adapter-v1"

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
        local_files_only: bool = True,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.model_id = str(model_id)
        self.revision = str(revision)
        self.tokenizer_id = str(tokenizer_id or model_id)
        self.tokenizer_revision = str(tokenizer_revision or revision)
        self.local_files_only = bool(local_files_only)
        self.device_map = str(device_map)
        self.dtype = str(dtype)
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """Load and return the causal language model."""

    @abstractmethod
    def load_tokenizer(self):
        """Load and return the matching tokenizer."""

    @abstractmethod
    def render_chat(self, messages):
        """Render chat messages with the model's native template."""

    @abstractmethod
    def validate_response_tokens(self, labels):
        """Return the unique single-token IDs for the response labels."""

    @abstractmethod
    def get_response_logits(self, messages, labels):
        """Return next-token logits for each response label."""

    @abstractmethod
    def get_response_metrics(self, messages, labels, *, positive_label):
        """Return binary metrics plus measured full-vocabulary validity fields."""

    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of residual-stream transformer blocks."""

    @abstractmethod
    def get_residual_state(self, messages, layer, position):
        """Return one residual state at a named block and token position."""

    @abstractmethod
    def run_with_residual_intervention(self, messages, layer, intervention):
        """Run one prompt while editing the selected residual state."""

    @abstractmethod
    def eos_token_ids(self):
        """Return all tokenizer/model EOS IDs."""

    def load(self) -> "ReplicationModelAdapter":
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.model is None:
            self.load_model()
        return self

    def descriptor(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "adapter_version": self.adapter_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
        }


class HuggingFaceResidualAdapter(ReplicationModelAdapter):
    """Shared implementation for decoder-only Hugging Face chat models."""

    def load_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=self.tokenizer_revision,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )
        return self.tokenizer

    def load_model(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            config=config,
            revision=self.revision,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
            dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
        )
        self.model.eval()
        return self.model

    def render_chat(self, messages):
        if self.tokenizer is None:
            self.load_tokenizer()
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _tokenize(self, messages):
        if self.tokenizer is None:
            self.load_tokenizer()
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        values = dict(encoded) if isinstance(encoded, Mapping) else {"input_ids": encoded}
        if self.model is None:
            self.load_model()
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in values.items()}

    def validate_response_tokens(self, labels):
        from cognitive_discovery.participants.token_mapping import verify_choice_tokens

        if self.tokenizer is None:
            self.load_tokenizer()
        return verify_choice_tokens(self.tokenizer, tuple(labels))

    def _chat_token_ids(self, messages, labels):
        from cognitive_discovery.participants.token_mapping import verify_chat_choice_tokens

        if self.tokenizer is None:
            self.load_tokenizer()
        return verify_chat_choice_tokens(self.tokenizer, messages, tuple(labels))

    def _next_token_logits(self, messages, labels):
        import torch

        token_ids = self._chat_token_ids(messages, labels)
        with torch.inference_mode():
            output = self.model(
                **self._tokenize(messages),
                output_hidden_states=False,
                use_cache=False,
            )
        if getattr(output, "hidden_states", None) is not None:
            raise RuntimeError("behavior-only adapter unexpectedly received hidden states")
        return token_ids, output.logits[0, -1]

    def get_response_logits(self, messages, labels):
        token_ids, logits = self._next_token_logits(messages, labels)
        return {
            label: float(logits[token_id].detach().float().cpu())
            for label, token_id in token_ids.items()
        }

    def get_response_metrics(self, messages, labels, *, positive_label):
        """Measure action logits and their mass/rank in the full vocabulary."""

        import torch

        from cognitive_discovery.participants.token_mapping import binary_choice_metrics

        labels = tuple(labels)
        token_ids, logits = self._next_token_logits(messages, labels)
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
        result["top_token_is_action"] = int(logits.argmax().item()) in set(
            token_ids.values()
        )
        return result

    def _runner(self):
        from cognitive_discovery.mechanistic.activations.qwen_runner import MechanisticQwenRunner
        from cognitive_discovery.participants.qwen import QwenParticipant

        self.load()
        return MechanisticQwenRunner(
            QwenParticipant(self.model, self.tokenizer, self.model_id, self.revision)
        )

    def num_layers(self) -> int:
        return self._runner().layer_count

    def get_residual_state(self, messages, layer, position):
        if position not in {"final_prompt_token", -1}:
            raise ValueError("replication harness supports final_prompt_token residual states")
        labels = getattr(self, "active_labels", None)
        positive = getattr(self, "active_positive_label", None)
        if not labels or positive is None:
            raise RuntimeError("active response interface must be bound before residual capture")
        result = self._runner().forward(
            messages,
            labels,
            positive_label=positive,
            capture_layers=(int(layer),),
        )
        return result.states[int(layer)]

    def bind_response_interface(self, labels, *, positive_label: str) -> None:
        if positive_label not in labels:
            raise ValueError("positive label must be part of the response interface")
        self.active_labels = tuple(labels)
        self.active_positive_label = str(positive_label)

    def run_with_residual_intervention(self, messages, layer, intervention):
        labels = getattr(self, "active_labels", None)
        positive = getattr(self, "active_positive_label", None)
        if not labels or positive is None:
            raise RuntimeError("active response interface must be bound before intervention")
        result = self._runner().forward(
            messages,
            labels,
            positive_label=positive,
            editors={int(layer): intervention},
            capture_layers=(),
        )
        return {
            "persistence_logit": result.persistence_logit,
            "p_continue": result.p_continue,
        }

    def eos_token_ids(self):
        self.load()
        values = set()
        for source in (self.tokenizer, getattr(self.model, "config", None)):
            value = getattr(source, "eos_token_id", None)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values.update(int(item) for item in value)
            elif value is not None:
                values.add(int(value))
        return tuple(sorted(values))


class QwenAdapter(HuggingFaceResidualAdapter):
    adapter_name = "qwen"
    adapter_version = "qwen-hf-residual-v2"

    def load_model(self):
        """Load Qwen text or hybrid checkpoints behind the Qwen adapter boundary."""

        import torch
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )
        if getattr(config, "model_type", None) == "qwen3_5" and hasattr(
            config, "text_config"
        ):
            try:
                from transformers import Qwen3_5ForConditionalGeneration as model_class
            except ImportError:
                from transformers import AutoModelForMultimodalLM as model_class
        else:
            from transformers import AutoModelForCausalLM as model_class
        self.model = model_class.from_pretrained(
            self.model_id,
            config=config,
            revision=self.revision,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
            dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
        )
        self.model.eval()
        return self.model


class LlamaAdapter(HuggingFaceResidualAdapter):
    adapter_name = "llama"
    adapter_version = "llama-hf-residual-v2"


class GemmaAdapter(HuggingFaceResidualAdapter):
    adapter_name = "gemma"
    adapter_version = "gemma-hf-residual-v2"


class MistralAdapter(HuggingFaceResidualAdapter):
    adapter_name = "mistral"
    adapter_version = "mistral-hf-residual-v2"


class AdapterRegistry:
    def __init__(self):
        self._adapters: dict[str, type[ReplicationModelAdapter]] = {}

    def register(self, adapter_class: type[ReplicationModelAdapter]) -> None:
        name = adapter_class.adapter_name
        if not name or name == "abstract" or name in self._adapters:
            raise ValueError(f"invalid or duplicate replication adapter: {name!r}")
        self._adapters[name] = adapter_class

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def create(self, name: str, **kwargs) -> ReplicationModelAdapter:
        try:
            adapter_class = self._adapters[name]
        except KeyError as error:
            raise ValueError(
                f"unknown replication adapter {name!r}; choose from {list(self.names())}"
            ) from error
        return adapter_class(**kwargs)


adapter_registry = AdapterRegistry()
for _adapter in (QwenAdapter, LlamaAdapter, GemmaAdapter, MistralAdapter):
    adapter_registry.register(_adapter)


def resolve_relative_layers(num_layers: int, fractions) -> list[int]:
    """Resolve preregistered depth fractions using zero-based block indices."""

    num_layers = int(num_layers)
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    values = [float(value) for value in fractions]
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("relative depths must be strictly between zero and one")
    layers = [min(num_layers - 1, max(0, math.floor(value * num_layers))) for value in values]
    if len(set(layers)) != len(layers):
        raise ValueError("relative depths collapse to duplicate layers for this model")
    return layers
