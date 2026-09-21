"""Tests for Issue #31 grouped execution (small shapes, fast)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.grouped import grouped_swiglu, naive_swiglu, plan_groups


def tiny_weights():
    mx.random.seed(0)
    weights = {}
    for expert in range(4):
        gate = mx.random.normal((16, 8), dtype=mx.float32) * 0.1
        up = mx.random.normal((16, 8), dtype=mx.float32) * 0.1
        down = mx.random.normal((8, 16), dtype=mx.float32) * 0.1
        weights[expert] = (gate, up, down)
    x = mx.random.normal((5, 16), dtype=mx.float32)
    mx.eval(list(weights.values()), x)
    return weights, x


def test_plan_groups_stable_and_complete():
    topk = [[2, 0], [1, 2], [0, 0]]
    groups = plan_groups(topk, 3)
    assert groups == {2: [(0, 0), (1, 1)], 0: [(0, 1), (2, 0), (2, 1)], 1: [(1, 0)]}
    assert plan_groups([[], []], 3) == {}
    try:
        plan_groups([[9]], 3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_grouped_matches_naive():
    weights, x = tiny_weights()
    topk = [[3, 1], [0, 2], [1, 1], [2, 3], [0, 3]]
    router = [[0.6, 0.4]] * 5
    groups = plan_groups(topk, 4)
    ref = naive_swiglu(x, weights, topk, router)
    got = grouped_swiglu(x, weights, groups, router)
    diff = float(mx.max(mx.abs(ref - got)).item())
    assert diff < 1e-4


def test_grouped_without_router():
    weights, x = tiny_weights()
    topk = [[0, 1], [2, 3]]
    groups = plan_groups(topk, 4)
    ref = naive_swiglu(x, weights, topk)
    got = grouped_swiglu(x, weights, groups)
    diff = float(mx.max(mx.abs(ref - got)).item())
    assert diff < 1e-4
