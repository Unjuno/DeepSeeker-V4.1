"""Tests for Issue #27 KV store (small buffers only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.kvmodel import cache_bytes
from deepseeker.kvstore import BudgetExceeded, KVStore

CONFIG = load_json(
    REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "inference" / "config.json"
)


def store(budget_mib=64):
    return KVStore(CONFIG, budget_mib * 1024**2)


def test_alloc_matches_formula():
    kv = store()
    info = kv.alloc("a", 1, 1024)
    assert info["bytes"] == cache_bytes(kv.geometry, 1, 1024)["total_bytes"]
    assert kv.reconcile("a")["match"] is True
    assert kv.telemetry()["live_requests"] == 1
    kv.free("a")
    assert kv.allocated_bytes == 0


def test_grow_and_reset():
    kv = store()
    kv.alloc("a", 1, 512)
    grown = kv.grow("a", 1024)
    assert grown["bytes"] > 0
    assert kv.reconcile("a")["match"] is True
    with pytest.raises(ValueError):
        kv.grow("a", 256)
    with pytest.raises(KeyError):
        kv.grow("missing", 1024)
    kv.reset()
    assert kv.allocated_bytes == 0


def test_budget_enforced_cleanly():
    kv = KVStore(CONFIG, 1024)
    with pytest.raises(BudgetExceeded):
        kv.alloc("big", 1, 131072)
    assert kv.allocated_bytes == 0
    assert kv.telemetry()["live_requests"] == 0
    with pytest.raises(ValueError):
        KVStore(CONFIG, 0)


def test_pattern_roundtrip():
    kv = store()
    kv.alloc("a", 1, 256)
    kv.write_pattern("a", "window", 9)
    entry = kv._requests["a"]
    assert kv.read_sum("a", "window") == 9 * entry["buffers"]["window"].size
    with pytest.raises(KeyError):
        kv.read_sum("missing", "window")
