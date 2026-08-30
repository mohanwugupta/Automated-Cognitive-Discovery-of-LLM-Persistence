"""Residual-stream hooks at one fixed pre-generation token position."""

from __future__ import annotations

from collections.abc import Callable


def decision_token_positions(attention_mask):
    """Return the last unmasked token for either left or right padding."""

    import torch

    mask = torch.as_tensor(attention_mask)
    if mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    positions = (
        torch.where(mask.bool(), indices, torch.full_like(indices, -1))
        .max(dim=1)
        .values
    )
    if (positions < 0).any():
        raise ValueError("every prompt must contain at least one unmasked token")
    return positions


def locate_transformer_layers(model):
    """Locate the text decoder blocks across supported Hugging Face wrappers."""

    candidates = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
    )
    for path in candidates:
        value = model
        for name in path:
            value = getattr(value, name, None)
            if value is None:
                break
        if value is not None and hasattr(value, "__len__") and len(value):
            return value
    raise ValueError("could not locate transformer layers on the supplied model")


def hidden_tensor(module_output):
    if hasattr(module_output, "last_hidden_state"):
        return module_output.last_hidden_state
    if isinstance(module_output, tuple):
        return module_output[0]
    return module_output


def replace_hidden_state(module_output, replacement):
    """Replace only the residual tensor while preserving a layer's output type."""

    if isinstance(module_output, tuple):
        return (replacement, *module_output[1:])
    if hasattr(module_output, "last_hidden_state"):
        module_output.last_hidden_state = replacement
        return module_output
    return replacement


def stream_layer_states(
    layers,
    positions,
    callback: Callable[[int, object], None],
    *,
    editors: dict[int, Callable[[object], object]] | None = None,
):
    """Register hooks that callback with only ``[batch, hidden]`` states.

    Returned handles must be removed by the caller. Editors receive and return
    the selected residual states; all other sequence positions are untouched.
    """

    import torch

    editors = dict(editors or {})
    positions = torch.as_tensor(positions)
    handles = []
    for layer_index, layer in enumerate(layers):

        def hook(_module, _inputs, output, *, index=layer_index):
            hidden = hidden_tensor(output)
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            selected_positions = positions.to(hidden.device)
            selected = hidden[batch, selected_positions]
            if index in editors:
                edited = editors[index](selected)
                if edited.shape != selected.shape:
                    raise ValueError("residual editor changed the selected-state shape")
                changed = hidden.clone()
                changed[batch, selected_positions] = edited.to(hidden.dtype)
                output = replace_hidden_state(output, changed)
                selected = edited
            callback(index, selected.detach())
            return output

        handles.append(layer.register_forward_hook(hook))
    return handles
