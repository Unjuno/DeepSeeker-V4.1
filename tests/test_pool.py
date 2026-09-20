"""Tests for Issue #23 bounded pool (synthetic blobs; no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.pool import POLICIES, ResidentPool


def blob(n: int, fill: int) -> np.ndarray:
    return np.full(n, fill, dtype=np.uint8)


def loader_factory(store: dict):
    def loader(layer: int, expert: int) -> np.ndarray:
        return store[(layer, expert)].copy()
    return loader


def test_ceiling_enforced_under_stress():
    pool = ResidentPool(2, 100)
    for expert in range(50):
        pool.admit(0, expert, blob(100, expert % 256))
    assert pool.bytes_resident <= 2 * 100
    assert pool.high_water_bytes <= 2 * 100
    assert pool.evictions == 48


def test_no_stale_after_evict_cycle():
    pool = ResidentPool(1, 10)
    pool.admit(0, 0, blob(10, 1))
    pool.admit(0, 1, blob(10, 2))
    assert pool.lookup(0, 0) is None  # evicted: unreachable
    found = pool.lookup(0, 1)
    assert found is not None and bytes(found) == bytes(blob(10, 2))


def test_pin_safety():
    pool = ResidentPool(1, 10)
    pool.admit(0, 0, blob(10, 1))
    pool.pin(0, 0)
    with pytest.raises(RuntimeError):
        pool.admit(0, 1, blob(10, 2))
    pool.unpin(0, 0)
    assert pool.admit(0, 1, blob(10, 2)) == 0
    with pytest.raises(KeyError):
        pool.pin(0, 99)


def test_demand_fallback_serves_and_counts():
    store = {(0, e): blob(10, e) for e in range(4)}
    pool = ResidentPool(2, 10)
    for _ in range(2):
        for expert in range(4):
            served = pool.demand(0, expert, loader_factory(store))
            assert bytes(served) == bytes(store[(0, expert)])
    metrics = pool.metrics()
    assert metrics["misses"] == 8  # one cold miss per demand call
    assert metrics["hits"] == 8  # post-admit verification lookup


def test_policies_instantiate():
    assert set(POLICIES) == {"lru", "lfu", "fifo"}
    for policy in POLICIES:
        pool = ResidentPool(2, 10, policy)
        pool.admit(0, 0, blob(10, 0))
        assert pool.metrics()["policy"] == policy


def test_blob_over_ceiling_rejected():
    pool = ResidentPool(2, 10)
    with pytest.raises(ValueError):
        pool.admit(0, 0, blob(11, 0))
    with pytest.raises(ValueError):
        ResidentPool(0, 10)
