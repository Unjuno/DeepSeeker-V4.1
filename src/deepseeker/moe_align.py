"""Issue #18 minimum test: sort-and-pad alignment port (Apple CPU/numpy).

Provenance: contract ported from FreeToken (Apache-2.0, Copyright 2026
FreeToken Authors / FlashML Team),
`python/freetoken/moe/fused.py::moe_align_block_size` docstring. Original
delegates to sgl_kernel/Triton at runtime; this port implements the same
observable contract in numpy for the M1 Max grouped-GEMM front end
(P6 #31 / P7). Expert IDs are 0-based here.

Contract: flatten topk_ids[M, K] row-major; group flat positions by
expert; pad each expert's run with the sentinel M*K up to a multiple of
block_size; concatenate experts in id order. Returns (sorted_ids,
expert_ids per block, num_tokens_post_pad).
"""

from __future__ import annotations

import numpy as np


def align_block_size(
    topk_ids: np.ndarray, block_size: int, num_experts: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sort flat (token, top-k) positions by expert with block padding."""
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be 2-D, got shape {topk_ids.shape}")
    if block_size < 1 or num_experts < 1:
        raise ValueError("block_size and num_experts must be >= 1")
    flat = topk_ids.reshape(-1)
    if flat.size and (flat.min() < 0 or flat.max() >= num_experts):
        raise ValueError("expert ids out of range")
    sentinel = flat.size
    order = np.argsort(flat, kind="stable")
    sorted_ids: list[int] = []
    expert_ids: list[int] = []
    for expert in range(num_experts):
        run = [int(i) for i in order if flat[i] == expert]
        pad = (-len(run)) % block_size
        run += [sentinel] * pad
        sorted_ids.extend(run)
        expert_ids.extend([expert] * (len(run) // block_size))
    return (
        np.asarray(sorted_ids, dtype=np.int64),
        np.asarray(expert_ids, dtype=np.int64),
        len(sorted_ids),
    )
