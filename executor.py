from __future__ import annotations

import torch

from ..mono.runner import FfnRunner
from .weights import LayerWeights


class FfnExecutor:
    """Explicit layer execution; never monkey-patches layer.forward."""

    def __init__(self, layers, layers_weights: list[LayerWeights], runner: FfnRunner):
        self.layers = tuple(layers)
        self.weights = tuple(layers_weights)
        self.runner = runner
        self._identities = [self._identity(layer) for layer in self.layers]

    @staticmethod
    def _identity(layer):
        return tuple((id(t), t.data_ptr()) for t in (
            layer.post_attention_layernorm.weight,
            layer.mlp.gate_up_proj.weight, layer.mlp.down_proj.weight))

    @staticmethod
    def _scale(hidden, scale):
        if hidden.ndim != 2:
            raise ValueError("Specialized FFN requires rank-2 decoder activations")
        value = scale.to(hidden.dtype)
        if value.ndim == 3:
            value = value[0]
        return hidden * (value + 1)

    def run_layer(self, index, positions, hidden_states, residual, *, active):
        layer = self.layers[index]
        if not active:
            return layer(positions=positions, hidden_states=hidden_states, residual=residual)
        if self._identity(layer) != self._identities[index]:
            raise RuntimeError("Weights were replaced after binding; rebuild decode controller")
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        if layer.layer_type == "linear_attention":
            part = layer.linear_attn(hidden_states=hidden_states)
        elif layer.layer_type == "full_attention":
            part = layer.self_attn(hidden_states=hidden_states, positions=positions)
        else:
            raise ValueError(f"Unsupported layer type {layer.layer_type}")
        if getattr(layer, "layer_scale", False):
            part = self._scale(part, layer.attn_layer_scale)
        hidden_states, residual = self.runner.launch_layer(self.weights[index], part, residual)
        if getattr(layer, "layer_scale", False):
            hidden_states = self._scale(hidden_states, layer.ffn_layer_scale)
        return hidden_states, residual
