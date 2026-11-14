from __future__ import annotations

import torch
import torch.nn.functional as F

THREADS = 128
WAVE_SIZE = 32
MAX_ROWS = 16


def scratch_layout(max_rows=16, hidden_size=5120, intermediate_size=17408):
    """Byte offsets and lengths; scratch contains only BF16 data.

    Barrier control and the sticky fault word are separate int32 tensors.
    """
    if not 1 <= max_rows <= MAX_ROWS:
        raise ValueError("max_rows must be in [1, 16]")
    x_bytes = max_rows * hidden_size * 2
    mid_bytes = max_rows * intermediate_size * 2
    return {"x_norm": (0, x_bytes), "act_mid": (x_bytes, mid_bytes),
            "total": (0, x_bytes + mid_bytes)}


def scratch_bytes(max_rows=16, hidden_size=5120, intermediate_size=17408):
    return scratch_layout(max_rows, hidden_size, intermediate_size)["total"][1]


def reference_ffn(attn_out, residual, layer, *, return_stages=False):
    """BF16 boundaries match stock Gemma norm -> BF16 linear -> SiluAndMul.

    This also runs on CPU for mathematical and integration tests.
    """
    summed = attn_out.float() + residual.float()
    residual_out = summed.to(torch.bfloat16)
    inv = torch.rsqrt(summed.square().mean(-1, keepdim=True) + layer.eps)
    x = (summed * inv * (layer.post_norm.float() + 1)).to(torch.bfloat16)
    gate_up = F.linear(x, layer.w13)
    gate, up = gate_up.chunk(2, dim=-1)
    mid = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    out = F.linear(mid, layer.w2)
    if return_stages:
        return out, residual_out, x, mid
    return out, residual_out
