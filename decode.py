from __future__ import annotations

import logging

import torch

from ..mono.runner import FfnRunner
from .executor import FfnExecutor
from .guards import check_static_support, decide_step, metadata_is_decode_only
from .spec import DecodeConfig
from .weights import bind_model_weights, inner_model

logger = logging.getLogger(__name__)


class SpecializedDecode:
    """Post-load, pre-capture controller. Static refusal leaves stock intact."""

    def __init__(self, model, vllm_config, device, config=None):
        self.config = config or DecodeConfig()
        self.model = inner_model(model)
        self.enabled = False
        self.runner = None
        self.executor = None
        self.refusal_reasons = []
        self._closed = False
        if not self.config.enabled:
            return
        self.refusal_reasons = check_static_support(vllm_config, torch.device(device), self.config)
        if self.refusal_reasons:
            logger.warning("Qwen specialized decode remains stock: %s", "; ".join(self.refusal_reasons))
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Build decode controller after weight loading and before capture")
        weights = bind_model_weights(self.model, self.config)
        self.runner = FfnRunner(device, self.config)
        self.runner.warmup()
        self.executor = FfnExecutor(self.model.layers, weights, self.runner)
        self.enabled = True
        logger.info("Qwen specialized FFN backend=%s rows<=%d", self.config.backend, self.config.max_rows)

    def plan(self, positions, intermediate_tensors=None, *, metadata=None,
             graph_mode="NONE"):
        if not self.enabled or self._closed or intermediate_tensors is not None:
            return decide_step(0)
        if metadata is None:
            from vllm.forward_context import get_forward_context, is_forward_context_available
            if not is_forward_context_available():
                return decide_step(0)
            context = get_forward_context()
            metadata = context.attn_metadata
            graph_mode = getattr(context.cudagraph_runtime_mode, "name", str(context.cudagraph_runtime_mode))
        return decide_step(
            positions.shape[-1], self.config.max_rows,
            is_decode_only=metadata_is_decode_only(metadata),
            is_piecewise_cudagraph=graph_mode not in ("NONE", "FULL"),
        )

    def after_step(self, failure=None):
        if self.runner is not None:
            if failure is not None:
                self.runner.abort()
            # Check every forward, including subsequent stock-only forwards.
            if failure is None:
                self.runner.check_health()
        if failure is not None:
            raise failure

    def close(self):
        if self.runner is not None and self.runner.device.type == "cuda":
            torch.cuda.synchronize(self.runner.device)
        self._closed = True
        self.enabled = False
        # Keep tensors alive while externally owned HIP graphs may reference them.
