"""Adapter called by the exact-source vLLM patch in install_vllm.py.

It leaves original attention, cache handling, embeddings, final norm, and
auxiliary-hidden-state collection in place. Use --enforce-eager with this
adapter; standalone HIP graph qualification is provided by validate.py.
"""
from __future__ import annotations

import os

from .runtime.decode import SpecializedDecode
from .runtime.spec import DecodeConfig


def get_or_build_decode(inner):
    config = getattr(inner, "_qwen38_vllm_config", None)
    if config is None or os.environ.get("QWEN38_SPECIALIZED_DECODE", "0") != "1":
        return None
    controller = getattr(inner, "_qwen38_decode_controller", None)
    if controller is None:
        device = next(inner.parameters()).device
        decode_config = DecodeConfig(
            backend=os.environ.get("QWEN38_FFN_BACKEND", "torch"),
            cache_dir=os.environ.get("QWEN38_FFN_CACHE_DIR"),
        )
        controller = SpecializedDecode(inner, config, device, decode_config)
        inner._qwen38_decode_controller = controller
    return controller


def run_decoder_layers(inner, positions, hidden_states, residual,
                       intermediate_tensors, aux_hidden_states):
    controller = get_or_build_decode(inner)
    decision = controller.plan(positions, intermediate_tensors) if controller else None
    failure = None
    try:
        for layer_idx in range(inner.start_layer, inner.end_layer):
            if decision is not None and decision.use_mono:
                hidden_states, residual = controller.executor.run_layer(
                    layer_idx, positions, hidden_states, residual, active=True)
            else:
                hidden_states, residual = inner.layers[layer_idx](
                    positions=positions, hidden_states=hidden_states, residual=residual)
            inner._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual)
    except Exception as error:
        failure = error
    if controller is not None:
        controller.after_step(failure)
    elif failure is not None:
        raise failure
    return hidden_states, residual
