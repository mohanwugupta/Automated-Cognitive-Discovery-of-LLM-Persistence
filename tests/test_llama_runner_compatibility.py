"""Tiny random Llama integration test. No gated weights or scientific data.

The existing Qwen-named wrappers already use AutoModelForCausalLM and generic
decoder hooks. Test that boundary before introducing an unnecessary adapter.
"""
import pytest
import numpy as np

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cognitive_discovery.participants.qwen import QwenParticipant
from cognitive_discovery.mechanistic.activations.qwen_runner import MechanisticQwenRunner


def test_llama_patch_identity_effect_and_alignment_gradient(monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM
    import cognitive_discovery.mechanistic.activations.qwen_runner as runner_module
    from cognitive_discovery.causal_mechanistic.das import DASAlignment

    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=32)).eval()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 4, int(messages[0]["content"])]]),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long)}

    monkeypatch.setattr(runner_module, "verify_chat_choice_tokens", lambda *a: {"X": 7, "Y": 9})
    runner = MechanisticQwenRunner(QwenParticipant(model, Tokenizer(), "tiny-random-llama"))
    assert runner.layer_count == 2
    base = [{"role": "user", "content": "5"}]
    source = [{"role": "user", "content": "6"}]
    call = lambda msg, **kw: runner.forward(msg, ["X", "Y"], positive_label="X", capture_layers=[0], **kw)
    natural = call(base)
    identity = call(base, editors={0: lambda state: state})
    assert identity.persistence_logit == pytest.approx(natural.persistence_logit, abs=1e-7)
    src = call(source).states[0]
    changed = call(base, editors={0: lambda state: torch.as_tensor(src, dtype=state.dtype).unsqueeze(0)})
    assert abs(changed.persistence_logit - natural.persistence_logit) > 1e-7
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    alignment = DASAlignment(16, 2, seed=3)
    value = runner.differentiable_persistence_logit(base, ["X", "Y"], positive_label="X",
        editors={0: lambda state: alignment.edit(state, src)})
    value.backward()
    assert torch.isfinite(alignment.parameter.grad).all()
    assert torch.linalg.vector_norm(alignment.parameter.grad) > 0
    assert all(not layer._forward_hooks for layer in runner.layers)
    assert np.isfinite(changed.states[0]).all()
