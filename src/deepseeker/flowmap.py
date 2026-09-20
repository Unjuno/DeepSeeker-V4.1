"""Analytic decode flow-map: bytes/token as functions of the cheats.

Closed form, no experiments. Reproduces the MLX bench (109.9ms vs 109.8ms
measured) then maximizes over:

  batch B        weight traffic shared across the batch (occupancy union)
  sparsity s     fraction of expert neurons skipped per token (row-select
                 on gate/up + column-select on down; needs a neuron predictor)
  mtp_accept a   expected extra accepted tokens per forward from the 3
                 built-in MTP heads (nearly-free tokens)
  residency      experts already in 64GB pay zero SSD/UW traffic here;
                 modeled as effective byte cost multiplier (default 1.0)

tok/s = B * (1 + a) / (bytes_per_forward / bandwidth + launch_latency)

Bytes per forward (fp4 bytes from the Issue #2 manifest):
  dense_fixed  attention+shared+head+router+norm weights, paid once per
               forward regardless of batch (broadcast).
  moe          union over the batch of routed experts x expert bytes x (1-s).
               Union via occupancy: n*(1-(1-k/n)^B) per layer.
  mtp_verify   per-draft verify cost (OPEN measurement, default 0).
"""

from __future__ import annotations

SCHEMA = "deepseeker.flowmap/v1"


def expert_union_per_layer(n_routed: int, top_k: int, batch: int) -> float:
    """Expected distinct experts touched in one layer across the batch."""
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch!r}")
    return n_routed * (1.0 - (1.0 - top_k / n_routed) ** batch)


def bytes_per_forward(
    batch: int,
    dense_bytes: float,
    n_layers: int,
    n_routed: int,
    top_k: int,
    expert_bytes: float,
    sparsity: float = 0.0,
    mtp_drafts: int = 0,
    mtp_verify_bytes: float = 0.0,
) -> dict:
    """Split traffic per forward pass."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity!r}")
    union = expert_union_per_layer(n_routed, top_k, batch)
    moe = n_layers * union * expert_bytes * (1.0 - sparsity)
    mtp = batch * mtp_drafts * mtp_verify_bytes
    total = dense_bytes + moe + mtp
    return {
        "dense_bytes": dense_bytes,
        "moe_bytes": moe,
        "mtp_verify_bytes": mtp,
        "total_bytes": total,
        "experts_touched_per_layer": union,
    }


def tokens_per_second(
    batch: int,
    mtp_accept: float,
    flow: dict,
    bandwidth_bps: float,
    launch_latency_s: float = 0.005,
) -> float:
    """Throughput from the flow-map: tokens confirmed per second."""
    if batch < 1 or mtp_accept < 0:
        raise ValueError("batch >= 1 and mtp_accept >= 0 required")
    seconds = flow["total_bytes"] / bandwidth_bps + launch_latency_s
    return batch * (1.0 + mtp_accept) / seconds


def required_batch(
    target: float,
    mtp_accept: float,
    sparsity: float,
    dense_bytes: float,
    n_layers: int,
    n_routed: int,
    top_k: int,
    expert_bytes: float,
    bandwidth_bps: float,
    launch_latency_s: float = 0.005,
    mtp_drafts: int = 0,
    mtp_verify_bytes: float = 0.0,
    ceiling: int = 4096,
) -> int | None:
    """Smallest batch hitting target tok/s (None beyond ceiling)."""
    for batch in range(1, ceiling + 1):
        flow = bytes_per_forward(
            batch, dense_bytes, n_layers, n_routed, top_k, expert_bytes,
            sparsity, mtp_drafts, mtp_verify_bytes,
        )
        if tokens_per_second(batch, mtp_accept, flow, bandwidth_bps, launch_latency_s) >= target:
            return batch
    return None


def required_sparsity(
    target: float,
    batch: int,
    mtp_accept: float,
    dense_bytes: float,
    n_layers: int,
    n_routed: int,
    top_k: int,
    expert_bytes: float,
    bandwidth_bps: float,
    launch_latency_s: float = 0.005,
    step: float = 0.01,
) -> float | None:
    """Smallest neuron sparsity hitting target (None if even s->1 fails)."""
    s = 0.0
    while s < 1.0:
        flow = bytes_per_forward(
            batch, dense_bytes, n_layers, n_routed, top_k, expert_bytes, s
        )
        if tokens_per_second(batch, mtp_accept, flow, bandwidth_bps, launch_latency_s) >= target:
            return round(s, 2)
        s = round(s + step, 10)
    return None
