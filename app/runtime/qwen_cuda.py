"""CUDA prefill tuning for the pinned Qwen ONNX export.

The current graph produces every present KV tensor with ``Concat``. It cannot
use in-place past/present buffer sharing or CUDA Graph capture as-is. CUDA
prefill tuning leaves decoding, weights, admission rules, and CPU execution
untouched. Each mode can be turned off when measuring the target hardware.
"""

from __future__ import annotations

import os


PREFILL_CHUNK_VARIABLE = "TRENDANALIZER_QWEN_PREFILL_CHUNK"
SPLIT_PREFILL_VARIABLE = "TRENDANALIZER_QWEN_SPLIT_PREFILL"
_CANDIDATE_CHUNKS = (128, 256, 512)


def benchmarkable_prefill_chunks(pinned_chunk: int) -> tuple[int, ...]:
    """Candidate prefill sizes bounded by the current ONNX prompt contract.

    These are candidates for measurement, not measured speedups.  We do not
    offer larger chunks here because the 8 GB RTX 3070 deployment target has
    not yet supplied a memory or latency profile for this model.
    """
    if type(pinned_chunk) is not int or not 1 <= pinned_chunk <= 2048:
        raise ValueError("Invalid pinned Qwen prefill chunk")
    return (pinned_chunk, *(size for size in _CANDIDATE_CHUNKS
                            if size > pinned_chunk and size % pinned_chunk == 0))


def qwen_prefill_chunk(pinned_chunk: int, provider: str) -> int:
    """Use 256 tokens on CUDA by default; preserve the pinned value on CPU.

    An explicit supported override selects a benchmark mode. An invalid value
    safely falls back to the pinned size. The benchmark checks answer quality
    for every mode before recommending one for production.
    """
    choices = benchmarkable_prefill_chunks(pinned_chunk)
    if provider != "CUDAExecutionProvider":
        return pinned_chunk
    raw = os.environ.get(PREFILL_CHUNK_VARIABLE)
    if raw is None:
        return 256 if 256 in choices else pinned_chunk
    raw = raw.strip()
    if not raw.isascii() or not raw.isdigit():
        return pinned_chunk
    proposed = int(raw)
    return proposed if proposed in choices else pinned_chunk


def qwen_split_prefill(provider: str) -> bool:
    """Project only the last prompt token on CUDA unless explicitly disabled."""
    if provider != "CUDAExecutionProvider":
        return False
    raw = os.environ.get(SPLIT_PREFILL_VARIABLE)
    return raw is None or raw.strip() == "1"
