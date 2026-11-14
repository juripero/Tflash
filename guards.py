from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .spec import DecodeConfig, EXPECTED_ARCH


def check_static_support(vllm_config: Any, device: torch.device,
                         config: DecodeConfig) -> list[str]:
    reasons = []
    if torch.version.hip is None or not torch.cuda.is_available():
        reasons.append("A ROCm device is required for vLLM integration.")
    else:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch != EXPECTED_ARCH:
            reasons.append(f"Expected {EXPECTED_ARCH}, got {arch}.")
    parallel = getattr(vllm_config, "parallel_config", None)
    for name in ("tensor_parallel_size", "pipeline_parallel_size",
                 "decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            reasons.append(f"{name} must equal 1.")
    model = getattr(vllm_config, "model_config", None)
    if getattr(model, "dtype", None) != torch.bfloat16:
        reasons.append("model_config.dtype must be torch.bfloat16.")
    if getattr(vllm_config, "quant_config", None) is not None:
        reasons.append("Quantized models are unsupported.")
    if getattr(model, "quantization", None) is not None:
        reasons.append("Quantization must be disabled.")
    if getattr(vllm_config, "lora_config", None) is not None:
        reasons.append("LoRA is unsupported: direct weight reads bypass its updates.")
    if getattr(vllm_config, "speculative_config", None) is not None:
        reasons.append("Speculative decoding requires a separately validated adapter.")
    hf = getattr(model, "hf_text_config", None)
    if hf is None:
        hf = getattr(model, "hf_config", None)
        hf = getattr(hf, "text_config", hf)
    if hf is None:
        reasons.append("Missing text configuration.")
    else:
        expected = {"hidden_size": config.hidden_size,
                    "intermediate_size": config.intermediate_size,
                    "num_hidden_layers": config.num_layers,
                    "hidden_act": "silu", "model_type": "qwen3_5_text"}
        for name, value in expected.items():
            if getattr(hf, name, None) != value:
                reasons.append(f"Expected {name}={value!r}.")
        if getattr(hf, "rms_norm_eps", None) != config.eps:
            reasons.append("Normalization epsilon differs from DecodeConfig.eps.")
    # This release integrates an eager Python model loop. Direct HIP graph
    # capture/replay is supported, but Dynamo/Inductor integration is not.
    if not getattr(model, "enforce_eager", False):
        reasons.append("The supplied vLLM adapter requires --enforce-eager.")
    return reasons


@dataclass(frozen=True)
class StepDecision:
    use_mono: bool
    rows: int = 0
    reason: str = ""


def decide_step(num_tokens: int, max_rows: int = 16,
                is_decode_only: bool = False,
                is_piecewise_cudagraph: bool = False) -> StepDecision:
    if not is_decode_only or is_piecewise_cudagraph:
        return StepDecision(False, reason="not an eligible decode step")
    if not 1 <= num_tokens <= max_rows:
        return StepDecision(False, reason="physical row count outside supported range")
    # Exact physical rows, never a rounded bucket that can exceed the input.
    return StepDecision(True, num_tokens)


def metadata_is_decode_only(metadata: Any) -> bool:
    """Require explicit GDN decode counts; missing evidence means stock.

    A full-attention max_query_len==1 alone cannot distinguish one-token
    prefills. The GDN metadata supplies the prefill/decode classification.
    """
    if not isinstance(metadata, dict):
        return False
    candidates = [m for m in metadata.values()
                  if hasattr(m, "num_prefills") and hasattr(m, "num_decodes")]
    if not candidates:
        return False
    for m in candidates:
        if m.num_prefills != 0 or m.num_decodes <= 0:
            return False
        if getattr(m, "num_prefill_tokens", 0) != 0:
            return False
        if getattr(m, "num_spec_decodes", 0) != 0:
            return False
    return True
