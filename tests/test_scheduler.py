"""Tests for Issue #28 unified scheduler (synthetic geometry)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.scheduler import PRIORITY, UnifiedScheduler

GEOMETRY = {
    "n_layers": 4,
    "window_size": 128,
    "head_dim": 512,
    "index_head_dim": 128,
    "kv_source_layers": [1],
    "compress_ratios": {1: 2},
    "window_bytes_per_elem": 1.0,
    "compress_bytes_per_elem": 0.5,
    "index_bytes_per_elem": 0.5,
}


def sched(mode="joint", total=10 * 1024**2):
    return UnifiedScheduler(total, GEOMETRY, 1024, 4, mode=mode,
                            engram_bytes=1024 * 1024, staging_bytes=1024 * 1024)


def test_reserve_and_slots():
    plan = sched().reserve(1, 1024)
    assert plan["expert_slots_per_layer"] >= 1
    assert plan["expert_pool_bytes"] == plan["expert_slots_per_layer"] * 4 * 1024
    with pytest.raises(RuntimeError):
        sched(total=1024).reserve(1, 1024 * 1024)
    with pytest.raises(ValueError):
        UnifiedScheduler(100, GEOMETRY, 1, 1, mode="chaos")


def test_backpressure_order():
    joint = sched("joint", total=100)
    assert joint.admit_class("demand", 10**9, 0) is True
    assert joint.admit_class("kv_grow", 1, 0) is True  # fits
    assert joint.admit_class("prefetch", 10**9, 0) is True  # sheds engram
    assert joint.metrics()["sheds"]["engram"] == 1
    assert joint.admit_class("engram", 10**9, 0) is False  # bottom sheds nothing
    indep = sched("independent", total=100)
    assert indep.admit_class("prefetch", 10**9, 0) is False
    with pytest.raises(ValueError):
        joint.admit_class("nope", 1, 0)


def test_steal_for_kv():
    joint = sched("joint")
    joint.reserve(1, 64)
    slots_before = joint.metrics()["expert_slots_per_layer"]
    freed = joint.steal_for_kv(1, 0)
    assert freed == {"engram": 0, "expert_slots": 0}  # no shortfall: steals nothing
    freed = joint.steal_for_kv(100 * 1024**2, 10 * 1024**2)  # pressured
    assert freed["engram"] > 0
    assert joint.metrics()["expert_slots_per_layer"] <= slots_before
    indep = sched("independent")
    indep.reserve(1, 64)
    assert indep.steal_for_kv(10**9, 10**9) == {"engram": 0, "expert_slots": 0}


def test_starvation_boost_and_metrics():
    joint = sched("joint")
    joint.record_service("demand", 100, stalled=False)
    assert joint.metrics()["served"]["demand"] == 1
    assert joint.boosts == 0
    for _ in range(200):
        joint.record_service("demand", 100, stalled=False)
    assert joint.boosts >= 1  # prefetch/engram starved -> boosted
    joint.record_service("prefetch", 50, stalled=True)
    assert joint.metrics()["stalls"] == 1
    assert "mean_bytes_per_use" in joint.metrics()
    with pytest.raises(ValueError):
        joint.record_service("nope", 1, False)


def test_priority_order_documented():
    assert PRIORITY[0] == "demand"
    assert PRIORITY[-1] == "engram"
