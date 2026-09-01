"""Custom EOS-terminated autoregressive generation with a frozen residual edit."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Callable

import numpy as np

from cognitive_discovery.mechanistic.activations.hooks import (
    hidden_tensor,
    locate_transformer_layers,
    replace_hidden_state,
)
from cognitive_discovery.participants.qwen import QwenParticipant


@dataclass
class GenerationStep:
    logits: object
    past_key_values: object = None
    baseline_logits: object = None


@dataclass
class DecodedGeneration:
    generated_token_ids: list[int]
    token_events: list[dict]
    termination_reason: str
    eos_emitted: bool
    context_censored: bool
    elapsed_seconds: float


def resolve_context_limit(model_config, tokenizer) -> int:
    """Resolve the finite architectural context without trusting sentinel values."""

    candidates = []
    objects = [model_config, getattr(model_config, "text_config", None)]
    for value in objects:
        if value is None:
            continue
        for name in (
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "seq_length",
        ):
            observed = getattr(value, name, None)
            if isinstance(observed, (int, np.integer)) and 1 < int(observed) < 10**9:
                candidates.append(int(observed))
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(tokenizer_limit, (int, np.integer))
        and 1 < int(tokenizer_limit) < 10**9
    ):
        candidates.append(int(tokenizer_limit))
    if not candidates:
        raise RuntimeError(
            "could not resolve the model's architectural context capacity"
        )
    return min(candidates)


def resolve_eos_token_ids(model, tokenizer) -> tuple[int, ...]:
    values = []
    for owner in (
        getattr(model, "generation_config", None),
        getattr(model, "config", None),
        tokenizer,
    ):
        value = getattr(owner, "eos_token_id", None) if owner is not None else None
        if value is None:
            continue
        values.extend(value if isinstance(value, (tuple, list, set)) else (value,))
    eos_ids = tuple(sorted({int(value) for value in values if value is not None}))
    if not eos_ids:
        raise RuntimeError("no canonical EOS token is defined by the model/tokenizer")
    return eos_ids


def edit_final_token(module_output, direction, magnitude: float):
    """Add one direction only at the newly processed final-token position."""

    import torch

    hidden = hidden_tensor(module_output)
    if hidden.ndim != 3:
        raise ValueError(
            "transformer residual output must have [batch, sequence, hidden] shape"
        )
    vector = torch.as_tensor(direction, device=hidden.device, dtype=hidden.dtype)
    if vector.ndim != 1 or vector.shape[0] != hidden.shape[-1]:
        raise ValueError("steering direction does not match the residual width")
    changed = hidden.clone()
    changed[:, -1, :] = changed[:, -1, :] + float(magnitude) * vector
    return replace_hidden_state(module_output, changed)


def _eos_stats(logits, eos_token_ids: tuple[int, ...]) -> tuple[float, float]:
    import torch

    indices = torch.as_tensor(eos_token_ids, device=logits.device, dtype=torch.long)
    eos_values = logits.float().index_select(0, indices)
    if not torch.isfinite(eos_values).all():
        raise RuntimeError("EOS was masked or made non-sampleable by the intervention")
    log_probability = torch.logsumexp(eos_values, dim=0) - torch.logsumexp(
        logits.float(), dim=0
    )
    probability = float(torch.exp(log_probability).detach().cpu())
    probability = min(max(probability, np.finfo(float).tiny), 1.0 - np.finfo(float).eps)
    log_odds = float(math.log(probability) - math.log1p(-probability))
    return probability, log_odds


def sample_token(logits, sampling: dict, generator) -> int:
    import torch

    values = logits.float() / float(sampling.get("temperature", 1.0))
    top_p = float(sampling.get("top_p", 1.0))
    if top_p < 1.0:
        sorted_values, sorted_indices = torch.sort(values, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_values, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_values = sorted_values.masked_fill(remove, float("-inf"))
        values = torch.full_like(values, float("-inf")).scatter(
            0, sorted_indices, sorted_values
        )
    probabilities = torch.softmax(values, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def decode_autoregressive(
    initial_token_ids,
    *,
    step: Callable[[list[int], object, int], GenerationStep],
    eos_token_ids: tuple[int, ...],
    context_limit: int,
    context_safety_margin: int,
    sampling: dict,
    seed: int,
    infrastructure_timeout_seconds: float | None = None,
) -> DecodedGeneration:
    """Decode until canonical EOS or remaining architectural capacity is consumed."""

    import torch

    tokens = [int(value) for value in initial_token_ids]
    initial_length = len(tokens)
    usable_limit = int(context_limit) - int(context_safety_margin)
    if initial_length >= usable_limit:
        raise ValueError("prompt leaves no architectural capacity for generation")
    generated = []
    events = []
    past = None
    generator = None
    started = time.monotonic()
    termination_reason = "context_limit"
    eos_emitted = False
    while len(tokens) < usable_limit:
        if (
            infrastructure_timeout_seconds is not None
            and time.monotonic() - started >= float(infrastructure_timeout_seconds)
        ):
            termination_reason = "infrastructure_timeout"
            break
        try:
            result = step(tokens, past, len(events))
        except torch.cuda.OutOfMemoryError:
            termination_reason = "infrastructure_oom"
            past = None
            result = None
            torch.cuda.empty_cache()
            break
        logits = result.logits
        if generator is None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(int(seed))
        p_eos, eos_logit = _eos_stats(logits, eos_token_ids)
        baseline_p = baseline_logit = None
        if result.baseline_logits is not None:
            baseline_p, baseline_logit = _eos_stats(
                result.baseline_logits, eos_token_ids
            )
        selected = sample_token(logits, sampling, generator)
        is_eos = selected in eos_token_ids
        events.append(
            {
                "token_index": len(events) + 1,
                "generated_before": len(generated),
                "sampled_token_id": selected,
                "sampled_eos": is_eos,
                "p_eos": p_eos,
                "eos_logit": eos_logit,
                "continuation_evidence": float(math.log1p(-p_eos) - math.log(p_eos)),
                "baseline_p_eos": baseline_p,
                "baseline_eos_logit": baseline_logit,
            }
        )
        past = result.past_key_values
        if is_eos:
            termination_reason = "eos"
            eos_emitted = True
            break
        generated.append(selected)
        tokens.append(selected)
    elapsed = time.monotonic() - started
    return DecodedGeneration(
        generated_token_ids=generated,
        token_events=events,
        termination_reason=termination_reason,
        eos_emitted=eos_emitted,
        context_censored=termination_reason
        in {"context_limit", "infrastructure_timeout", "infrastructure_oom"},
        elapsed_seconds=float(elapsed),
    )


def quality_metrics(token_ids: list[int], text: str) -> dict:
    from collections import Counter

    tokens = list(map(int, token_ids))

    def repeated_ngram_fraction(n):
        if len(tokens) < n:
            return 0.0
        grams = [
            tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)
        ]
        return float(1.0 - len(set(grams)) / len(grams))

    sentences = [
        value.strip().lower() for value in re.split(r"[.!?]+", text) if value.strip()
    ]
    repeated_sentence_fraction = (
        float(1.0 - len(set(sentences)) / len(sentences)) if sentences else 0.0
    )
    punctuation = sum(
        not character.isalnum() and not character.isspace() for character in text
    )
    punctuation_fraction = float(punctuation / max(1, len(text)))
    lexical_diversity = float(len(set(tokens)) / max(1, len(tokens)))
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    stopwords = {
        "and",
        "are",
        "for",
        "from",
        "that",
        "the",
        "this",
        "was",
        "were",
        "with",
    }
    topic_terms = [
        value for value, _ in Counter(words).most_common(8) if value not in stopwords
    ][:3]
    severe = bool(
        repeated_ngram_fraction(4) > 0.5
        or repeated_ngram_fraction(8) > 0.35
        or repeated_sentence_fraction > 0.5
        or punctuation_fraction > 0.5
    )
    return {
        "repeated_4gram_fraction": repeated_ngram_fraction(4),
        "repeated_8gram_fraction": repeated_ngram_fraction(8),
        "repeated_sentence_fraction": repeated_sentence_fraction,
        "lexical_diversity": lexical_diversity,
        "punctuation_fraction": punctuation_fraction,
        "severe_degeneration": severe,
        "coarse_topic_terms": ",".join(topic_terms),
    }


class QwenFreeGenerationRunner:
    """Qwen custom decoding runner that never applies an output-token cap."""

    def __init__(self, participant: QwenParticipant):
        self.participant = participant
        self.model = participant.model
        self.tokenizer = participant.tokenizer
        self.layers = locate_transformer_layers(self.model)
        self.context_limit = resolve_context_limit(self.model.config, self.tokenizer)
        self.eos_token_ids = resolve_eos_token_ids(self.model, self.tokenizer)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(QwenParticipant.from_pretrained(*args, **kwargs))

    def _messages(self, prompt: str):
        return [{"role": "user", "content": str(prompt)}]

    def _encode(self, prompt: str) -> list[int]:
        inputs = self.participant._tokenize(self._messages(prompt))
        if int(inputs["input_ids"].shape[0]) != 1:
            raise ValueError("OOD generation currently requires one prompt per run")
        return [int(value) for value in inputs["input_ids"][0].detach().cpu().tolist()]

    def _model_step(
        self,
        token_ids: list[int],
        past_key_values,
        *,
        layer: int,
        direction,
        magnitude: float,
        apply_intervention: bool,
        force_hook: bool = False,
        use_cache: bool = True,
    ) -> GenerationStep:
        import torch

        device = next(self.model.parameters()).device
        full_ids = torch.as_tensor([token_ids], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(full_ids)
        if use_cache and hasattr(self.model, "prepare_inputs_for_generation"):
            cache_position = torch.arange(
                0 if past_key_values is None else len(token_ids) - 1,
                len(token_ids),
                device=device,
                dtype=torch.long,
            )
            try:
                model_inputs = self.model.prepare_inputs_for_generation(
                    full_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    use_cache=True,
                )
            except TypeError:
                model_inputs = self.model.prepare_inputs_for_generation(
                    full_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
        elif use_cache and past_key_values is not None:
            model_inputs = {
                "input_ids": full_ids[:, -1:],
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
        else:
            model_inputs = {
                "input_ids": full_ids,
                "attention_mask": attention_mask,
                "use_cache": bool(use_cache),
            }
        handle = None
        if apply_intervention or force_hook:
            if int(layer) < 0 or int(layer) >= len(self.layers):
                raise ValueError(
                    "frozen intervention layer is outside the loaded model"
                )

            def hook(_module, _inputs, output):
                return edit_final_token(output, direction, magnitude)

            handle = self.layers[int(layer)].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            if handle is not None:
                handle.remove()
        return GenerationStep(
            logits=outputs.logits[0, -1].float(),
            past_key_values=getattr(outputs, "past_key_values", None),
        )

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        alpha: float,
        layer: int,
        direction,
        sigma_E: float,
        regime: str,
        sampling: dict,
        infrastructure_timeout_seconds: float | None,
        eos_logit_scale: float | None = None,
    ) -> tuple[dict, list[dict]]:
        import torch

        if regime not in {"continuous", "pulse", "eos_control"}:
            raise ValueError(f"unknown OOD intervention regime: {regime}")
        initial_ids = self._encode(prompt)
        magnitude = float(alpha) * float(sigma_E)

        def step(tokens, past, step_index):
            apply = regime == "continuous" or (regime == "pulse" and step_index == 0)
            result = self._model_step(
                tokens,
                past,
                layer=layer,
                direction=direction,
                magnitude=magnitude,
                apply_intervention=apply and alpha != 0,
                use_cache=True,
            )
            if regime == "eos_control" and alpha != 0:
                values = result.logits.clone()
                indices = torch.as_tensor(self.eos_token_ids, device=values.device)
                values[indices] -= float(alpha) * float(eos_logit_scale or 1.0)
                result.logits = values
            return result

        decoded = decode_autoregressive(
            initial_ids,
            step=step,
            eos_token_ids=self.eos_token_ids,
            context_limit=self.context_limit,
            context_safety_margin=int(sampling.get("context_safety_margin", 1)),
            sampling=sampling,
            seed=seed,
            infrastructure_timeout_seconds=infrastructure_timeout_seconds,
        )
        text = self.tokenizer.decode(
            decoded.generated_token_ids, skip_special_tokens=True
        )
        summary = {
            "seed": int(seed),
            "alpha": float(alpha),
            "regime": regime,
            "prompt_tokens": len(initial_ids),
            "generated_tokens": len(decoded.generated_token_ids),
            "generated_words": len(re.findall(r"\b\w+\b", text)),
            "decision_steps": len(decoded.token_events),
            "survival_time": max(1, len(decoded.token_events)),
            "eos_emitted": decoded.eos_emitted,
            "termination_reason": decoded.termination_reason,
            "context_censored": decoded.context_censored,
            "context_limit": self.context_limit,
            "elapsed_seconds": decoded.elapsed_seconds,
            "generated_text": text,
            **quality_metrics(decoded.generated_token_ids, text),
        }
        return summary, decoded.token_events

    def immediate_eos_effect(
        self,
        prompt: str,
        *,
        layer: int,
        direction,
        sigma_E: float,
        alpha: float,
    ) -> dict:
        token_ids = self._encode(prompt)
        result = self._model_step(
            token_ids,
            None,
            layer=layer,
            direction=direction,
            magnitude=float(alpha) * float(sigma_E),
            apply_intervention=alpha != 0,
            use_cache=False,
        )
        p_eos, eos_logit = _eos_stats(result.logits, self.eos_token_ids)
        return {"alpha": float(alpha), "p_eos": p_eos, "eos_logit": eos_logit}

    def numerical_checks(self, prompt: str, *, layer: int, direction) -> dict:
        import torch

        token_ids = self._encode(prompt)
        baseline = self._model_step(
            token_ids,
            None,
            layer=layer,
            direction=direction,
            magnitude=0.0,
            apply_intervention=False,
            use_cache=False,
        )
        zero = self._model_step(
            token_ids,
            None,
            layer=layer,
            direction=direction,
            magnitude=0.0,
            apply_intervention=False,
            force_hook=True,
            use_cache=False,
        )
        alpha_zero_max_abs = float(
            torch.max(torch.abs(baseline.logits - zero.logits)).detach().cpu()
        )
        cached_first = self._model_step(
            token_ids,
            None,
            layer=layer,
            direction=direction,
            magnitude=0.0,
            apply_intervention=False,
            use_cache=True,
        )
        selectable = cached_first.logits.clone()
        selectable[list(self.eos_token_ids)] = float("-inf")
        next_token = int(selectable.argmax().item())
        extended = [*token_ids, next_token]
        cached_second = self._model_step(
            extended,
            cached_first.past_key_values,
            layer=layer,
            direction=direction,
            magnitude=0.0,
            apply_intervention=False,
            use_cache=True,
        )
        uncached_second = self._model_step(
            extended,
            None,
            layer=layer,
            direction=direction,
            magnitude=0.0,
            apply_intervention=False,
            use_cache=False,
        )
        cache_max_abs = float(
            torch.max(torch.abs(cached_second.logits - uncached_second.logits))
            .detach()
            .cpu()
        )
        scale = float(torch.max(torch.abs(uncached_second.logits)).detach().cpu())
        return {
            "alpha_zero_max_abs_logit_error": alpha_zero_max_abs,
            "alpha_zero_passed": bool(alpha_zero_max_abs <= 1e-5),
            "cache_max_abs_logit_error": cache_max_abs,
            "cache_relative_error": cache_max_abs / max(scale, 1e-12),
            "cache_equivalent": bool(
                cache_max_abs <= 2e-2 or cache_max_abs / max(scale, 1e-12) <= 2e-3
            ),
            "eos_token_ids": list(self.eos_token_ids),
            "context_limit": self.context_limit,
        }
