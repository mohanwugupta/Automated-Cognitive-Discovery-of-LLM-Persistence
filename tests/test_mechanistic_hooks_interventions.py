import numpy as np
import torch
from torch import nn

from cognitive_discovery.mechanistic.activations.hooks import (
    decision_token_positions,
    stream_layer_states,
)
from cognitive_discovery.mechanistic.calibration.projection_to_computation import (
    DirectionCalibration,
)
from cognitive_discovery.mechanistic.patching.projection_patch import (
    orthogonal_residual,
    projection_patch,
)
from cognitive_discovery.mechanistic.representations.ridge_probe import RidgeProbe
from cognitive_discovery.mechanistic.steering.intervene import add_direction


class AddOne(nn.Module):
    def forward(self, hidden):
        return (hidden + 1.0, "cache")


def test_final_prompt_position_is_padding_invariant():
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    assert decision_token_positions(mask).tolist() == [1, 3]


def test_hook_extracts_and_edits_only_decision_position():
    layer = AddOne()
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    positions = torch.tensor([1, 3])
    captured = {}
    editor_calls = []

    def edit_once(state):
        editor_calls.append(state.clone())
        return state + torch.tensor([10.0, 0.0, 0.0])

    handles = stream_layer_states(
        [layer],
        positions,
        lambda index, state: captured.setdefault(index, state.clone()),
        editors={0: edit_once},
    )
    output = layer(hidden)[0]
    for handle in handles:
        handle.remove()
    expected = hidden + 1.0
    expected[0, 1, 0] += 10.0
    expected[1, 3, 0] += 10.0
    assert torch.equal(output, expected)
    assert torch.equal(captured[0], expected[torch.arange(2), positions])
    assert len(editor_calls) == 1


def test_steering_zero_and_reverse_change_projection_exactly():
    state = np.array([1.0, -2.0, 3.0])
    direction = np.array([2.0, 0.0, 0.0])
    assert np.array_equal(add_direction(state, direction, 0.0), state)
    plus = add_direction(state, direction, 1.5)
    minus = add_direction(state, direction, -1.5)
    unit = direction / np.linalg.norm(direction)
    assert np.isclose((plus - state) @ unit, 1.5)
    assert np.isclose((minus - state) @ unit, -1.5)


def test_projection_patch_is_exact_and_preserves_orthogonal_residual():
    source = np.array([2.0, 4.0, -1.0])
    target = np.array([-3.0, 1.0, 7.0])
    direction = np.array([1.0, 2.0, 0.0])
    patched = projection_patch(target, source, direction)
    unit = direction / np.linalg.norm(direction)
    assert np.isclose(patched @ unit, source @ unit)
    assert np.allclose(
        orthogonal_residual(patched, direction),
        orthogonal_residual(target, direction),
    )
    assert np.array_equal(projection_patch(target, target, direction), target)


def test_probe_and_calibration_use_training_statistics_only():
    train_x = np.arange(8, dtype=float)[:, None]
    train_y = 3 * train_x[:, 0] - 2
    probe = RidgeProbe(alpha=1e-9).fit(train_x, train_y)
    original_mean = probe.mean_.copy()
    probe.predict(np.array([[1000.0], [2000.0]]))
    assert np.array_equal(probe.mean_, original_mean)
    calibration = DirectionCalibration.fit(train_x[:, 0], train_y)
    dose = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(calibration.slope * calibration.alpha_for_delta(dose), dose)
