"""TP1 BF16 post-attention FFN for the Qwen3.5-compatible 27B layout."""

from .runtime.decode import SpecializedDecode
from .runtime.spec import DecodeConfig

__all__ = ["DecodeConfig", "SpecializedDecode"]
