"""Tests for the analytic flow-map (no experiments)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.flowmap import (
    bytes_per_forward,
    expert_union_per_layer,
    required_batch,
    required_sparsity,
    tokens_per_second,
)

P = {"dense_bytes": 8.0 * 1024**3, "n_layers": 40, "n_routed": 384, "top_k": 6,
     "expert_bytes": 18800640}
BW = 132.7 * 1024**3
GEO = {k: P[k] for k in ("n_layers", "n_routed", "top_k", "expert_bytes")}


def flow(batch=1, sparsity=0.0):
    return bytes_per_forward(batch, P["dense_bytes"], P["n_layers"], P["n_routed"],
                             P["top_k"], P["expert_bytes"], sparsity)


def test_reproduces_bench():
    # fp4 MoE 4.5GB + dense 8GB at 132.7GiB/s ~= MLX bench scaled to fp4.
    rate = tokens_per_second(1, 0.0, flow(), BW, launch_latency_s=0.0)
    assert rate == pytest.approx(BW / flow()["total_bytes"], rel=1e-9)
    assert 10.0 < rate < 12.0


def test_union_bounds():
    assert expert_union_per_layer(384, 6, 1) == pytest.approx(6.0)
    big = expert_union_per_layer(384, 6, 100000)
    assert big == pytest.approx(384.0)
    with pytest.raises(ValueError):
        expert_union_per_layer(384, 6, 0)


def test_sparsity_and_mtp_monotonic():
    base = tokens_per_second(1, 0.0, flow(), BW)
    sparse = tokens_per_second(1, 0.0, flow(sparsity=0.5), BW)
    mtp = tokens_per_second(1, 2.0, flow(), BW)
    assert sparse > base
    assert mtp == pytest.approx(3 * base)
    with pytest.raises(ValueError):
        bytes_per_forward(1, 1.0, 40, 384, 6, 1.0, sparsity=1.0)


def test_batch_sublinear_from_union():
    r1 = tokens_per_second(1, 0.0, flow(1), BW)
    r8 = tokens_per_second(8, 0.0, flow(8), BW)
    assert r8 > r1
    assert r8 < 8 * r1  # union growth eats linear scaling


def test_solvers():
    kw = dict(dense_bytes=P["dense_bytes"], bandwidth_bps=BW, **GEO)
    assert required_batch(400, 2.0, 0.5, **kw) == 125
    assert required_batch(10**9, 0.0, 0.0, ceiling=4, **kw) is None
    assert required_sparsity(400, 32, 2.0, **kw) == pytest.approx(0.79)
