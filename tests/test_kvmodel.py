"""Tests for Issue #15 KV model (committed artifacts only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.kvmodel import cache_bytes, cache_geometry, unified_table

SNAPSHOT = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
CONFIG = load_json(SNAPSHOT / "inference" / "config.json")


def test_geometry_matches_runtime():
    geometry = cache_geometry(CONFIG)
    assert geometry["n_layers"] == 40
    assert geometry["window_size"] == 128
    assert geometry["head_dim"] == 512
    assert geometry["index_head_dim"] == 128
    assert geometry["kv_source_layers"] == [2, 8, 14, 20]
    assert geometry["compress_ratios"] == {2: 2, 8: 2, 14: 2, 20: 1}


def test_window_cache_is_context_independent():
    geometry = cache_geometry(CONFIG)
    small = cache_bytes(geometry, 1, 64)
    big = cache_bytes(geometry, 1, 131072)
    assert small["window_bytes"] == big["window_bytes"] == 40 * 128 * 512
    assert big["compress_bytes"] > small["compress_bytes"]
    assert big["index_bytes"] > small["index_bytes"]


def test_hand_computed_totals():
    geometry = cache_geometry(CONFIG)
    kv = cache_bytes(geometry, 1, 4096)
    # window: 40 layers x 128 x 512 fp8
    assert kv["window_bytes"] == 40 * 128 * 512
    # compress fp4: (2048+2048+2048+4096) x 512 x 0.5
    assert kv["compress_bytes"] == int((2048 * 3 + 4096) * 512 * 0.5)
    # index fp4: (2048+2048+2048+4096) x 128 x 0.5
    assert kv["index_bytes"] == int((2048 * 3 + 4096) * 512 * 0.5 / 4)
    assert kv["total_bytes"] == kv["window_bytes"] + kv["compress_bytes"] + kv["index_bytes"]


def test_batch_scales_linearly():
    geometry = cache_geometry(CONFIG)
    one = cache_bytes(geometry, 1, 4096)
    four = cache_bytes(geometry, 4, 4096)
    assert four["total_bytes"] == 4 * one["total_bytes"]


def test_bad_shapes_rejected():
    geometry = cache_geometry(CONFIG)
    with pytest.raises(ValueError):
        cache_bytes(geometry, 0, 4096)
    with pytest.raises(ValueError):
        cache_bytes(geometry, 1, 0)


def test_unified_table_accounting():
    kv = {"total_bytes": 100}
    table = unified_table(
        common_model_bytes=10,
        dense_engram_bytes=20,
        expert_slots_per_layer=2,
        n_layers=4,
        expert_bytes=5,
        kv=kv,
        operating_bytes=1000,
    )
    assert table["components"]["expert-pool"] == 2 * 4 * 5
    assert table["total_bytes"] == 10 + 20 + 40 + 100 + 0 + 1024**3
    assert table["over_budget"] is True
    assert table["headroom_bytes"] == 1000 - table["total_bytes"]
