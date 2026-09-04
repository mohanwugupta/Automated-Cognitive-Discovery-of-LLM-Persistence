"""Custom EOS-terminated autoregressive generation with a frozen residual edit."""

from __future__ import annotations

import inspect
import math
import re
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cognitive_discovery.mechanistic.activations.hooks import (
    hidden_tensor,
    locate_transformer_layers,
    replace_hidden_state,
)
from cognitive_discovery.participants.qwen import QwenParticipant


GENERATION_ENGINE_VERSION = "qwen_incremental_single_token_v2"


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


# These bounds are defined in probability space because BF16 cached and
# uncached forwards may use different kernel shapes.  A vocabulary-wide
# maximum logit difference can therefore be dominated by a token with
# effectively zero probability even when the two sampling distributions are
# indistinguishable.  Total variation directly bounds the probability change
# of any next-token event; the separate EOS bound protects the scientific
# outcome even when EOS is very unlikely.
CACHE_EQUIVALENCE_CRITERIA = {
    "total_variation_max": 0.02,
    "eos_log_odds_abs_error_max": 0.05,
}


def compare_cached_logits(
    cached_logits,
    uncached_logits,
    eos_token_ids: tuple[int, ...],
) -> dict:
    """Compare cache paths by sampling distribution and EOS evidence.

    Raw maximum logit error is retained as a diagnostic, but is deliberately
    not a pass/fail criterion: softmax is invariant to a common logit shift and
    large errors on negligible-probability tail tokens do not affect sampling.
    """

    import torch

    cached = cached_logits.detach().float().reshape(-1)
    uncached = uncached_logits.detach().float().reshape(-1)
    if cached.shape != uncached.shape:
        raise ValueError("cached and uncached logits have different shapes")
    if not torch.isfinite(cached).all() or not torch.isfinite(uncached).all():
        raise RuntimeError("cached or uncached logits contain non-finite values")

    difference = cached - uncached
    max_abs = float(difference.abs().max().cpu())
    mean_abs = float(difference.abs().mean().cpu())
    rmse = float(torch.sqrt(torch.mean(difference.square())).cpu())
    scale = float(uncached.abs().max().cpu())

    cached_log_prob = torch.log_softmax(cached, dim=0)
    uncached_log_prob = torch.log_softmax(uncached, dim=0)
    cached_probability = torch.exp(cached_log_prob)
    uncached_probability = torch.exp(uncached_log_prob)
    probability_difference = (cached_probability - uncached_probability).abs()
    total_variation = float((0.5 * probability_difference.sum()).cpu())
    max_probability_error = float(probability_difference.max().cpu())

    log_midpoint = torch.logaddexp(cached_log_prob, uncached_log_prob) - math.log(2.0)
    js_value = 0.5 * (
        torch.sum(cached_probability * (cached_log_prob - log_midpoint))
        + torch.sum(uncached_probability * (uncached_log_prob - log_midpoint))
    )
    js_divergence = max(0.0, float(js_value.cpu()))

    cached_eos_probability, cached_eos_log_odds = _eos_stats(cached, eos_token_ids)
    uncached_eos_probability, uncached_eos_log_odds = _eos_stats(
        uncached, eos_token_ids
    )
    eos_probability_error = abs(cached_eos_probability - uncached_eos_probability)
    eos_log_odds_error = abs(cached_eos_log_odds - uncached_eos_log_odds)

    top_k = min(10, int(cached.numel()))
    cached_top = set(torch.topk(cached_probability, top_k).indices.cpu().tolist())
    uncached_top = set(torch.topk(uncached_probability, top_k).indices.cpu().tolist())
    top_overlap = len(cached_top & uncached_top) / max(1, top_k)
    criteria = dict(CACHE_EQUIVALENCE_CRITERIA)
    equivalent = bool(
        total_variation <= criteria["total_variation_max"]
        and eos_log_odds_error <= criteria["eos_log_odds_abs_error_max"]
    )
    return {
        "cache_validation_version": "next_token_distribution_v1",
        "cache_equivalence_criteria": criteria,
        "cache_max_abs_logit_error": max_abs,
        "cache_mean_abs_logit_error": mean_abs,
        "cache_rmse_logit_error": rmse,
        "cache_relative_error": max_abs / max(scale, 1e-12),
        "cache_total_variation_distance": total_variation,
        "cache_jensen_shannon_divergence": js_divergence,
        "cache_max_probability_abs_error": max_probability_error,
        "cache_eos_probability_abs_error": eos_probability_error,
        "cache_eos_log_odds_abs_error": eos_log_odds_error,
        "cache_top1_token_match": bool(
            cached_probability.argmax().item() == uncached_probability.argmax().item()
        ),
        "cache_top10_overlap_fraction": float(top_overlap),
        "cache_equivalent": equivalent,
    }


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

    @property
    def uses_hybrid_linear_attention(self) -> bool:
        """Whether decoding crosses Qwen-style chunked/recurrent mixer paths."""

        config = getattr(self.model, "config", None)
        text_config = getattr(config, "text_config", None) or config
        layer_types = getattr(text_config, "layer_types", ()) or ()
        return any(str(value) == "linear_attention" for value in layer_types)

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
        step_ids = full_ids if past_key_values is None else full_ids[:, -1:]
        if use_cache and hasattr(self.model, "prepare_inputs_for_generation"):
            # Transformers 5.x no longer infers which tokens are new from the
            # cache. Passing the full growing prefix without
            # ``next_sequence_length`` reprocesses old tokens on top of the
            # existing cache. Always provide only the new token after prefill,
            # and pass version-specific hints only when explicitly supported.
            parameters = inspect.signature(
                self.model.prepare_inputs_for_generation
            ).parameters
            prepare_kwargs = {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if "next_sequence_length" in parameters:
                prepare_kwargs["next_sequence_length"] = int(step_ids.shape[1])
            if "is_first_iteration" in parameters:
                prepare_kwargs["is_first_iteration"] = past_key_values is None
            if "cache_position" in parameters:
                if past_key_values is None:
                    past_seen_tokens = 0
                elif hasattr(past_key_values, "get_seq_length"):
                    past_seen_tokens = int(past_key_values.get_seq_length())
                else:
                    past_seen_tokens = len(token_ids) - int(step_ids.shape[1])
                prepare_kwargs["cache_position"] = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + int(step_ids.shape[1]),
                    device=device,
                    dtype=torch.long,
                )
            model_inputs = self.model.prepare_inputs_for_generation(
                step_ids, **prepare_kwargs
            )
        elif use_cache:
            model_inputs = {
                "input_ids": step_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values
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
        if cached_first.past_key_values is None:
            raise RuntimeError("model did not return a cache when use_cache=True")
        prefill_comparison = compare_cached_logits(
            cached_first.logits, baseline.logits, self.eos_token_ids
        )

        # Exercise retained cache state across several decode steps. The token
        # path is fixed by the retained-cache run, then replayed from a fresh
        # prompt prefill. This isolates state retention, token slicing, and
        # cache positions without conflating Qwen3.5's distinct Gated DeltaNet
        # chunked and recurrent numerical algorithms.
        replay_tokens = []
        cached_step = cached_first
        for _ in range(3):
            selectable = cached_step.logits.clone()
            selectable[list(self.eos_token_ids)] = float("-inf")
            replay_tokens.append(int(selectable.argmax().item()))
            cached_step = self._model_step(
                [*token_ids, *replay_tokens],
                cached_step.past_key_values,
                layer=layer,
                direction=direction,
                magnitude=0.0,
                apply_intervention=False,
                use_cache=True,
            )

        fresh_step = self._model_step(
            token_ids,
            None,
            layer=layer,
            direction=direction,
            magnitude=0.0,
            apply_intervention=False,
            use_cache=True,
        )
        for index in range(len(replay_tokens)):
            fresh_step = self._model_step(
                [*token_ids, *replay_tokens[: index + 1]],
                fresh_step.past_key_values,
                layer=layer,
                direction=direction,
                magnitude=0.0,
                apply_intervention=False,
                use_cache=True,
            )
        state_replay_comparison = compare_cached_logits(
            cached_step.logits, fresh_step.logits, self.eos_token_ids
        )

        # Keep the literal PRD comparison as a separate, strict result. On
        # hybrid Qwen3.5 this crosses the chunked-prefill and recurrent-decode
        # Gated DeltaNet paths, for which Transformers currently has a known
        # upstream divergence. It must be reported, not relaxed or hidden.
        native_uncached = self._model_step(
            [*token_ids, *replay_tokens],
            None,
            layer=layer,
            direction=direction,
            magnitude=0.0,
            apply_intervention=False,
            use_cache=False,
        )
        native_comparison = compare_cached_logits(
            cached_step.logits, native_uncached.logits, self.eos_token_ids
        )
        cache_state_reuse_equivalent = bool(
            prefill_comparison["cache_equivalent"]
            and state_replay_comparison["cache_equivalent"]
        )
        native_limitation = bool(
            self.uses_hybrid_linear_attention
            and not native_comparison["cache_equivalent"]
            and cache_state_reuse_equivalent
        )
        return {
            "alpha_zero_max_abs_logit_error": alpha_zero_max_abs,
            "alpha_zero_passed": bool(alpha_zero_max_abs <= 1e-5),
            # The top-level fields remain the literal native cache/no-cache
            # result consumed by the strict preregistered aggregate gate.
            **native_comparison,
            "cache_prefill_comparison": prefill_comparison,
            "cache_state_replay_comparison": state_replay_comparison,
            "cache_state_reuse_equivalent": cache_state_reuse_equivalent,
            "cache_replay_tokens": len(replay_tokens),
            "model_uses_hybrid_linear_attention": self.uses_hybrid_linear_attention,
            "native_no_cache_limitation": native_limitation,
            "cache_validation_disposition": (
                "native_qwen35_chunk_recurrent_divergence"
                if native_limitation
                else (
                    "passed"
                    if native_comparison["cache_equivalent"]
                    else "unexplained_native_cache_divergence"
                )
            ),
            "eos_token_ids": list(self.eos_token_ids),
            "context_limit": self.context_limit,
        }
