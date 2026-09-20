"""Analytic decode flow-map for bandwidth-bound inference.

This module is a hypothesis model, not an end-to-end benchmark.

Important semantics:

* batch means concurrent sequences processed together. tokens_per_second
  therefore returns aggregate accepted-token throughput across the batch.
  It is not the generation rate of one conversation when batch > 1.
* sparsity models skipping expert neurons. Any non-zero value is an
  unverified, potentially quality-changing counterfactual unless exact
  equivalence has separately been proved. DeepSeeker's lossless/default
  path uses sparsity=0.
* mtp_accept is expected extra accepted tokens per sequence/forward.
  If mtp_verify_bytes is left at zero, MTP results are optimistic upper
  bounds until candidate/verification cost is measured.

The traffic model uses exact encoded routed-expert bytes from the Issue #2
manifest. Batch expert union is approximated by independent occupancy:

    n * (1 - (1 - k/n) ** B)

Aggregate throughput:

    B * (1 + a) / (bytes_per_forward / bandwidth + launch_latency)

Per-sequence throughput for the same synchronized batch:

    (1 + a) / (bytes_per_forward / bandwidth + launch_latency)
"""

from __future__ import annotations

SCHEMA = "deepseeker.flowmap/v2"


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
    """Split modeled encoded-weight traffic per synchronized forward.

    sparsity is retained only for counterfactual research. A non-zero value
    is marked unverified because skipping neurons is not lossless by default
    and must not be used as evidence for the project quality target.
    """
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
        "quality_status": (
            "lossless-compatible-model"
            if sparsity == 0.0
            else "UNVERIFIED_QUALITY_CHANGING_SPARSITY"
        ),
    }


def per_sequence_tokens_per_second(
    mtp_accept: float,
    flow: dict,
    bandwidth_bps: float,
    launch_latency_s: float = 0.005,
) -> float:
    """Accepted-token rate for one sequence in a synchronized batch.

    With non-zero mtp_accept and zero supplied verification traffic this is
    an optimistic upper bound.
    """
    if mtp_accept < 0:
        raise ValueError("mtp_accept >= 0 required")
    seconds = flow["total_bytes"] / bandwidth_bps + launch_latency_s
    return (1.0 + mtp_accept) / seconds


def tokens_per_second(
    batch: int,
    mtp_accept: float,
    flow: dict,
    bandwidth_bps: float,
    launch_latency_s: float = 0.005,
) -> float:
    """Aggregate accepted-token throughput across batch sequences.

    For batch > 1 this MUST NOT be reported as the generation speed of a
    single conversation. Use per_sequence_tokens_per_second for that.
    """
    if batch < 1:
        raise ValueError("batch >= 1 required")
    return batch * per_sequence_tokens_per_second(
        mtp_accept, flow, bandwidth_bps, launch_latency_s
    )


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
    """Smallest concurrent batch hitting target aggregate tok/s.

    A non-zero sparsity is quality-unverified. Zero mtp_verify_bytes with
    non-zero mtp_accept is optimistic until verification cost is measured.
    """
    for batch in range(1, ceiling + 1):
        flow = bytes_per_forward(
            batch,
            dense_bytes,
            n_layers,
            n_routed,
            top_k,
            expert_bytes,
            sparsity,
            mtp_drafts,
            mtp_verify_bytes,
        )
        if (
            tokens_per_second(
                batch, mtp_accept, flow, bandwidth_bps, launch_latency_s
            )
            >= target
        ):
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
    """Counterfactual sparsity for target aggregate throughput.

    This function does NOT establish a lossless path. Any returned non-zero
    sparsity is outside the project quality invariant until exact equivalence
    is independently demonstrated.
    """
    s = 0.0
    while s < 1.0:
        flow = bytes_per_forward(
            batch, dense_bytes, n_layers, n_routed, top_k, expert_bytes, s
        )
        if (
            tokens_per_second(
                batch, mtp_accept, flow, bandwidth_bps, launch_latency_s
            )
            >= target
        ):
            return round(s, 2)
        s = round(s + step, 10)
    return None
