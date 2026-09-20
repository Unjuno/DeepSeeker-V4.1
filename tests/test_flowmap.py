"""Tests for the analytic flow-map (no end-to-end model required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.flowmap import (
    bytes_per_forward,
    expert_union_per_layer,
    per_sequence_tokens_per_second,
    required_batch,
    required_sparsity,
    tokens_per_second,
)

P = {
    "dense_bytes": 8.0 * 1024**3,
    "n_layers": 40,
    "n_routed": 384,
    "top_k": 6,
    "expert_bytes": 18_800_640,
}
BW = 132.7 * 1024**3
GEO = {
    k: P[k]
    for k in ("n_layers", "n_routed", "top_k", "expert_bytes")
}


def flow(batch=1, sparsity=0.0):
    return bytes_per_forward(
        batch,
        P["dense_bytes"],
        P["n_layers"],
        P["n_routed"],
        P["top_k"],
        P["expert_bytes"],
        sparsity,
    )


def test_single_sequence_bandwidth_model():
    rate = tokens_per_second(
        1, 0.0, flow(), BW, launch_latency_s=0.0
    )
    assert rate == pytest.approx(BW / flow()["total_bytes"], rel=1e-9)
    assert 10.0 < rate < 12.0


def test_union_bounds():
    assert expert_union_per_layer(384, 6, 1) == pytest.approx(6.0)
    big = expert_union_per_layer(384, 6, 100000)
    assert big == pytest.approx(384.0)
    with pytest.raises(ValueError):
        expert_union_per_layer(384, 6, 0)


def test_aggregate_and_per_sequence_are_distinct():
    batch = 8
    f = flow(batch)
    aggregate = tokens_per_second(batch, 0.0, f, BW)
    per_sequence = per_sequence_tokens_per_second(0.0, f, BW)
    assert aggregate == pytest.approx(batch * per_sequence)
    assert aggregate > per_sequence
    assert per_sequence < tokens_per_second(1, 0.0, flow(1), BW)


def test_sparsity_is_explicitly_quality_unverified():
    base = flow()
    sparse = flow(sparsity=0.5)
    assert base["quality_status"] == "lossless-compatible-model"
    assert (
        sparse["quality_status"]
        == "UNVERIFIED_QUALITY_CHANGING_SPARSITY"
    )
    assert (
        tokens_per_second(1, 0.0, sparse, BW)
        > tokens_per_second(1, 0.0, base, BW)
    )
    with pytest.raises(ValueError):
        bytes_per_forward(
            1, 1.0, 40, 384, 6, 1.0, sparsity=1.0
        )


def test_mtp_zero_verify_cost_is_mathematically_optimistic():
    base = tokens_per_second(1, 0.0, flow(), BW)
    mtp_zero_cost = tokens_per_second(1, 2.0, flow(), BW)
    assert mtp_zero_cost == pytest.approx(3 * base)

    with_verify = bytes_per_forward(
        1,
        P["dense_bytes"],
        P["n_layers"],
        P["n_routed"],
        P["top_k"],
        P["expert_bytes"],
        0.0,
        mtp_drafts=2,
        mtp_verify_bytes=1024**3,
    )
    assert (
        tokens_per_second(1, 2.0, with_verify, BW)
        < mtp_zero_cost
    )


def test_batch_sublinear_from_union():
    r1 = tokens_per_second(1, 0.0, flow(1), BW)
    r8 = tokens_per_second(8, 0.0, flow(8), BW)
    assert r8 > r1
    assert r8 < 8 * r1


def test_solvers_are_aggregate_and_sparsity_is_counterfactual():
    kw = dict(
        dense_bytes=P["dense_bytes"],
        bandwidth_bps=BW,
        **GEO,
    )
    # Lossless-compatible model: sparsity=0.  This is aggregate throughput.
    assert required_batch(400, 2.0, 0.0, **kw) == 276

    # Historical B=125 result only exists under unverified 50% neuron skip.
    assert required_batch(400, 2.0, 0.5, **kw) == 125
    assert required_sparsity(400, 32, 2.0, **kw) == pytest.approx(0.79)

    assert required_batch(
        10**9, 0.0, 0.0, ceiling=4, **kw
    ) is None
