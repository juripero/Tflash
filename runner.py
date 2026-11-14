from __future__ import annotations

import torch

from ..runtime.spec import DecodeConfig
from ..runtime.weights import LayerWeights, check_tensor
from .a16w16 import build_ffn
from .ffn import MAX_ROWS, reference_ffn, scratch_bytes
from .residency import checked_residency
from .sync import CONTROL_WORDS, ERROR_WORDS


class FfnRunner:
    """One workspace, one serialized HIP stream, no host epoch counters.

    Call warmup before any capture. A graph may replay only with this runner
    kept alive and without overlapping another graph using this workspace.
    Physical row counts are captured constants; graph padding is processed
    independently, and cannot affect valid rows or any attention cache.
    """

    def __init__(self, device: torch.device, config: DecodeConfig):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if config.backend == "hip" and self.device.type != "cuda":
            raise ValueError("HIP backend requires a GPU")
        self.config = config
        self.module = None
        self.residency = None
        self.scratch = None
        self.control = None
        self.fault_word = torch.zeros(ERROR_WORDS, dtype=torch.int32, device=self.device)
        self._stream = None
        self._ready = False

    def warmup(self):
        if self._ready:
            return
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warmup cannot run during graph capture")
        if self.config.backend == "hip":
            with torch.cuda.device(self.device):
                self.module = build_ffn(self.config.hidden_size,
                                        self.config.intermediate_size,
                                        self.config.cache_dir)
                self.residency = checked_residency(self.module, self.config.blocks)
                count = scratch_bytes(MAX_ROWS, self.config.hidden_size,
                                      self.config.intermediate_size) // 2
                self.scratch = torch.empty(count, dtype=torch.bfloat16, device=self.device)
                self.control = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=self.device)
                # Complete initialization before binding an execution stream.
                torch.cuda.synchronize(self.device)
        self._ready = True

    def _check(self, layer, attn_out, residual):
        rows = attn_out.shape[0] if attn_out.ndim == 2 else 0
        if not 1 <= rows <= self.config.max_rows:
            raise ValueError("Expected 1..max_rows physical rows in a rank-2 activation")
        h, inter = self.config.hidden_size, self.config.intermediate_size
        gpu = self.config.backend == "hip"
        for tensor, shape, name in (
            (attn_out, (rows, h), "attn_out"), (residual, (rows, h), "residual"),
            (layer.post_norm, (h,), "post_norm"), (layer.w13, (2 * inter, h), "w13"),
            (layer.w2, (h, inter), "w2"),
        ):
            check_tensor(tensor, shape, name, self.device, require_cuda=gpu)
        if layer.eps != self.config.eps:
            raise ValueError("Layer epsilon differs from runner epsilon")
        return rows

    @torch.no_grad()
    def launch_layer(self, layer: LayerWeights, attn_out: torch.Tensor,
                     residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._ready:
            raise RuntimeError("Call runner.warmup() before launching")
        rows = self._check(layer, attn_out, residual)
        if self.config.backend == "torch":
            return reference_ffn(attn_out, residual, layer)
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device).cuda_stream
            if self._stream is None:
                self._stream = stream
            elif self._stream != stream:
                raise RuntimeError("Workspace is bound to a different HIP stream; use a separate runner")
            out = torch.empty_like(attn_out)
            resout = torch.empty_like(residual)
            self._raw_launch(layer, attn_out, residual, out, resout, rows, stream)
        return out, resout

    def _raw_launch(self, layer, a, residual, out, resout, rows, stream,
                    *, drop_cta=-1, max_spins=None):
        """Internal entry used by fault-injection qualification, never fallback."""
        self.module.ffn_launch(
            a.data_ptr(), residual.data_ptr(), layer.post_norm.data_ptr(),
            layer.w13.data_ptr(), layer.w2.data_ptr(), out.data_ptr(),
            resout.data_ptr(), self.scratch.data_ptr(), self.control.data_ptr(),
            self.fault_word.data_ptr(), rows, layer.eps, self.config.blocks,
            self.config.max_spins if max_spins is None else max_spins,
            drop_cta, stream,
        )

    def check_health(self):
        # Exactly one element, and no per-layer bitwise_or launch.
        torch._assert_async(self.fault_word[0] == 0,
                            "Qwen BF16 FFN failed; discard this step and stop the worker")

    def abort(self):
        self.fault_word[0].fill_(1)

    def health(self):
        """Explicit host synchronization for diagnostics, never in hot dispatch."""
        return tuple(self.fault_word.tolist())
