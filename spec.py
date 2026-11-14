from __future__ import annotations

import math
from dataclasses import dataclass

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
NUM_LAYERS = 64
MAX_DECODE_TOKENS = 16
EXPECTED_ARCH = "gfx1201"


@dataclass(frozen=True)
class DecodeConfig:
    """Use `hip` explicitly after running validate.py on the target machine.

    `torch` is a staged parity baseline, not a claimed optimization. Both
    backends leave attention and its caches with the stock vLLM modules.
    """

    enabled: bool = True
    backend: str = "torch"
    max_rows: int = 16
    hidden_size: int = HIDDEN_SIZE
    intermediate_size: int = INTERMEDIATE_SIZE
    num_layers: int = NUM_LAYERS
    eps: float = 1e-6
    blocks: int = 64
    max_spins: int = 10_000_000
    cache_dir: str | None = None

    def __post_init__(self):
        if self.backend not in ("torch", "hip"):
            raise ValueError("backend must be 'torch' or 'hip'")
        if not 1 <= self.max_rows <= MAX_DECODE_TOKENS:
            raise ValueError("max_rows must be in [1, 16]")
        if any(x <= 0 or x % 16 for x in (self.hidden_size, self.intermediate_size)):
            raise ValueError("hidden/intermediate dimensions must be positive multiples of 16")
        if self.num_layers <= 0 or not 1 <= self.blocks <= 64:
            raise ValueError("num_layers must be positive and blocks in [1, 64]")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not 1 <= self.max_spins < 2**31:
            raise ValueError("max_spins must be in [1, 2**31)")
