from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Residency:
    blocks_per_processor: int
    processor_count: int
    wave_size: int

    @property
    def capacity(self):
        return self.blocks_per_processor * self.processor_count


def checked_residency(module, blocks: int) -> Residency:
    """HIP queries the actual compiled function; unknown occupancy is fatal.

    Use the runtime's processor-count units consistently. Do not assume that
    an RDNA driver reports CUs rather than WGPs in multiProcessorCount.
    """
    values = module.occupancy()
    if len(values) != 3:
        raise RuntimeError("Malformed HIP occupancy response")
    result = Residency(*map(int, values))
    if result.blocks_per_processor <= 0 or result.processor_count <= 0:
        raise RuntimeError("HIP could not establish persistent grid residency")
    if result.wave_size != 32 or result.capacity < blocks:
        raise RuntimeError(f"Grid {blocks} cannot reside: {result}")
    return result
