"""Qwen residual-stream runner isolated from behavior-only collection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cognitive_discovery.participants.qwen import QwenParticipant
from cognitive_discovery.participants.token_mapping import (
    binary_choice_metrics,
    verify_chat_choice_tokens,
)

from .hooks import (
    decision_token_positions,
    hidden_tensor,
    locate_transformer_layers,
    replace_hidden_state,
    stream_layer_states,
)


@dataclass
class MechanisticForward:
    persistence_logit: float
    p_continue: float
    states: dict[int, np.ndarray]


@dataclass
class MechanisticGradient:
    """One final-prompt residual state and its semantic-logit gradient."""

    persistence_logit: float
    state: np.ndarray
    gradient: np.ndarray
    layer: int


class MechanisticQwenRunner:
    activation_position = "final_prompt_token"
    layer_convention = "zero_based_block_output"

    def __init__(self, participant: QwenParticipant):
        self.participant = participant
        self.model = participant.model
        self.tokenizer = participant.tokenizer
        self.layers = locate_transformer_layers(self.model)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(QwenParticipant.from_pretrained(*args, **kwargs))

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def forward(
        self,
        messages,
        labels,
        *,
        positive_label: str,
        editors=None,
        capture_layers=None,
    ) -> MechanisticForward:
        import torch

        labels = tuple(labels)
        token_ids = verify_chat_choice_tokens(self.tokenizer, messages, labels)
        inputs = self.participant._tokenize(messages)
        if "attention_mask" in inputs:
            positions = decision_token_positions(inputs["attention_mask"])
        else:
            positions = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[1] - 1,
                dtype=torch.long,
                device=inputs["input_ids"].device,
            )
        selected_layers = (
            set(range(self.layer_count))
            if capture_layers is None
            else set(capture_layers)
        )
        states: dict[int, np.ndarray] = {}

        def capture(index, state):
            if index in selected_layers:
                states[index] = state.float().cpu().numpy()[0].copy()

        handles = stream_layer_states(
            self.layers, positions, capture, editors=dict(editors or {})
        )
        try:
            with torch.inference_mode():
                outputs = self.model(
                    **inputs, output_hidden_states=False, use_cache=False
                )
        finally:
            for handle in handles:
                handle.remove()
        if getattr(outputs, "hidden_states", None) is not None:
            raise RuntimeError(
                "mechanistic runner requested streamed hooks, not hidden-state outputs"
            )
        logits = outputs.logits[0, int(positions[0].item())]
        selected = {
            label: float(logits[token_id].detach().float().cpu())
            for label, token_id in token_ids.items()
        }
        metrics = binary_choice_metrics(selected, positive_label)
        return MechanisticForward(
            persistence_logit=float(metrics["choice_logit"]),
            p_continue=float(metrics["p_positive"]),
            states=states,
        )

    def choice_output_direction(
        self, messages, labels, *, positive_label: str
    ) -> np.ndarray:
        """Direct positive-versus-negative unembedding geometry control."""

        token_ids = verify_chat_choice_tokens(self.tokenizer, messages, tuple(labels))
        negative_label = next(label for label in labels if label != positive_label)
        output = self.model.get_output_embeddings()
        if output is None or not hasattr(output, "weight"):
            raise RuntimeError("model does not expose output-embedding weights")
        vector = (
            output.weight[token_ids[positive_label]]
            - output.weight[token_ids[negative_label]]
        )
        return vector.detach().float().cpu().numpy().copy()

    def state_and_persistence_gradient(
        self,
        messages,
        labels,
        *,
        positive_label: str,
        layer: int,
    ) -> MechanisticGradient:
        """Differentiate ``CONTINUE - DISENGAGE`` at one residual-stream state.

        The hook detaches the selected block output from upstream computation, then
        differentiates through every downstream block. Only the final prompt-token
        row is returned or persisted. Calling this method separately for each layer
        avoids one layer's detached leaf cutting the gradient path of another.
        """

        import torch

        layer = int(layer)
        if layer < 0 or layer >= self.layer_count:
            raise ValueError(f"layer {layer} is outside [0, {self.layer_count})")
        labels = tuple(labels)
        token_ids = verify_chat_choice_tokens(self.tokenizer, messages, labels)
        negative_label = next(label for label in labels if label != positive_label)
        inputs = self.participant._tokenize(messages)
        if "attention_mask" in inputs:
            positions = decision_token_positions(inputs["attention_mask"])
        else:
            positions = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[1] - 1,
                dtype=torch.long,
                device=inputs["input_ids"].device,
            )
        captured: dict[str, object] = {}

        def make_leaf(_module, _inputs, output):
            hidden = hidden_tensor(output)
            leaf = hidden.detach().requires_grad_(True)
            captured["leaf"] = leaf
            return replace_hidden_state(output, leaf)

        handle = self.layers[layer].register_forward_hook(make_leaf)
        try:
            with torch.enable_grad():
                outputs = self.model(
                    **inputs, output_hidden_states=False, use_cache=False
                )
                leaf = captured.get("leaf")
                if leaf is None:
                    raise RuntimeError(
                        "gradient hook did not observe the requested layer"
                    )
                position = int(positions[0].item())
                logits = outputs.logits[0, position]
                persistence = (
                    logits[token_ids[positive_label]]
                    - logits[token_ids[negative_label]]
                )
                gradient = torch.autograd.grad(
                    persistence, leaf, retain_graph=False, create_graph=False
                )[0]
                state = leaf[0, position]
                selected_gradient = gradient[0, position]
        finally:
            handle.remove()
        if not torch.isfinite(selected_gradient).all():
            raise RuntimeError("persistence-logit gradient contains a non-finite value")
        return MechanisticGradient(
            persistence_logit=float(persistence.detach().float().cpu()),
            state=state.detach().float().cpu().numpy().copy(),
            gradient=selected_gradient.detach().float().cpu().numpy().copy(),
            layer=layer,
        )
