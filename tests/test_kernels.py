"""Tests for Issue #32 kernel suite (tiny shapes, fast)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.kernels import (
    bench_fp8_dequant,
    bench_gemm,
    bench_launch,
    bench_swiglu_elem,
    relevance,
    time_fn,
)


def test_micros_return_stats():
    for row in (bench_gemm(1, iters=3), bench_swiglu_elem(2, iters=3),
                bench_fp8_dequant(2, iters=3), bench_launch(iters=5)):
        assert row["median_ms"] > 0
        assert row["p95_ms"] >= row["median_ms"]
        assert row["iters"] >= 3


def test_relevance_ranking_math():
    rows = [{"op": "a", "median_ms": 1.0}, {"op": "b", "median_ms": 0.1}]
    ranked = relevance(rows, {"a": 10.0, "b": 1000.0})
    by_op = {r["op"]: r for r in ranked}
    assert by_op["a"]["share_ms"] == 10.0
    assert by_op["b"]["share_ms"] == 100.0
    assert ranked[0]["op"] == "b"
    assert sum(r["share_pct"] for r in ranked) == pytest.approx(
        (10.0 + 100.0) / 109.8 * 100, abs=0.2
    )


def test_time_fn_counts_iters():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    out = time_fn(fn, iters=4, warmup=1)
    assert calls["n"] == 5
    assert out["iters"] == 4
