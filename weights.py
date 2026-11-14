from __future__ import annotations

from dataclasses import dataclass

import torch

from .spec import DecodeConfig


@dataclass(frozen=True)
class LayerWeights:
    layer_idx: int
    post_norm: torch.Tensor
    w13: torch.Tensor
    w2: torch.Tensor
    eps: float

    @property
    def hidden_size(self):
        return self.w13.shape[1]

    @property
    def intermediate_size(self):
        return self.w2.shape[1]


def inner_model(model):
    lm = getattr(model, "language_model", model)
    return getattr(lm, "model", lm)


def check_tensor(t, shape, name, device, *, require_cuda=True):
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"{name} is not a tensor")
    if t.dtype != torch.bfloat16 or tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name}: expected contiguous BF16 {shape}, got {t.dtype} {tuple(t.shape)}")
    if t.device != device or not t.is_contiguous():
        raise ValueError(f"{name}: wrong device or noncontiguous storage")
    if require_cuda and not t.is_cuda:
        raise ValueError(f"{name}: GPU storage required")
    if t.data_ptr() % 16:
        raise ValueError(f"{name}: storage must be 16-byte aligned")


def bind_model_weights(model, config: DecodeConfig, *, require_cuda=True):
    layers = inner_model(model).layers
    if len(layers) != config.num_layers:
        raise ValueError(f"Expected {config.num_layers} local decoder layers")
    result = []
    for i, layer in enumerate(layers):
        if getattr(layer, "layer_type", None) not in ("linear_attention", "full_attention"):
            raise ValueError(f"Layer {i}: unsupported attention type")
        if getattr(layer, "use_attn_reduce_scatter_for_moe", False):
            raise ValueError("Sequence-parallel MoE is unsupported")
        norm = layer.post_attention_layernorm
        mlp = layer.mlp
        if "Gemma" not in type(norm).__name__ and "Qwen3_5RMSNorm" not in type(norm).__name__:
            raise ValueError(f"Layer {i}: expected Gemma-style (1+w) RMSNorm")
        w13 = mlp.gate_up_proj.weight
        w2 = mlp.down_proj.weight
        device = w13.device
        for proj in (mlp.gate_up_proj, mlp.down_proj):
            if getattr(proj, "bias", None) is not None:
                raise ValueError(f"Layer {i}: FFN bias is unsupported")
            method = getattr(proj, "quant_method", None)
            if method is not None and type(method).__name__ != "UnquantizedLinearMethod":
                raise ValueError(f"Layer {i}: unsupported projection method {type(method).__name__}")
        if type(mlp.act_fn).__name__ != "SiluAndMul":
            raise ValueError(f"Layer {i}: expected SiluAndMul")
        eps = getattr(norm, "variance_epsilon", None)
        if eps != config.eps:
            raise ValueError(f"Layer {i}: normalization epsilon mismatch")
        for t, shape, name in (
            (norm.weight, (config.hidden_size,), "post_norm"),
            (w13, (2 * config.intermediate_size, config.hidden_size), "w13"),
            (w2, (config.hidden_size, config.intermediate_size), "w2"),
        ):
            check_tensor(t, shape, f"layer[{i}].{name}", device,
                         require_cuda=require_cuda)
        # Keep original BF16 storage; the HIP kernel reinterprets its bits.
        result.append(LayerWeights(i, norm.weight, w13, w2, eps))
    return result
