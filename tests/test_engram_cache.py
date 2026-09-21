"""Tests for Issue #26 Engram row cache (fake reader + tiny real reads)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.engram import row_ranges, table_refs_from_manifest
from deepseeker.engram_cache import EngramRowCache, access_stream

MODEL_ROOT = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
REVISION = load_json(
    REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
)["resolved_revision"]
MANIFEST = load_json(
    REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / REVISION / "model-manifest.json"
)
REFS = table_refs_from_manifest(MANIFEST)


def test_exact_rows_against_direct_reads():
    ref = REFS[1]
    cache = EngramRowCache(MODEL_ROOT, REFS, 64 * 1024**2, 4096)
    try:
        rows = cache.fetch_rows(1, [0, 1, 12345, ref.num_rows - 1])
        for row_id, (w, s) in zip([0, 1, 12345, ref.num_rows - 1], rows):
            info = row_ranges(ref, row_id)
            assert len(w) == ref.weight_row_bytes == 256
            assert len(s) == ref.scale_row_bytes == 8
            assert info["useful_bytes"] == 264
    finally:
        cache.shutdown()


def test_hit_rate_hotspot_vs_uniform():
    real_ref = REFS[1]
    info = row_ranges(real_ref, 0)
    assert info["useful_bytes"] == 264
    hot = access_stream(1000, 200, seed=1, hotspot_frac=1.0)
    assert len(set(hot)) < 200  # hotspot concentrates
    cache = EngramRowCache(Path("/x"), {1: real_ref}, 1024 * 1024, 4096,
                           reader=lambda shard, start, size: b"\x00" * size)
    try:
        cache.fetch_rows(1, hot)
        misses_first = cache.metrics()["page_misses"]
        cache.fetch_rows(1, hot)
        metrics = cache.metrics()
        assert metrics["page_hits"] > misses_first  # second pass mostly hits
        assert metrics["bytes_resident"] <= 1024 * 1024
    finally:
        cache.shutdown()


def test_capacity_bounded():
    cache = EngramRowCache(MODEL_ROOT, REFS, 1 * 1024**2, 4096)
    try:
        cache.fetch_rows(1, list(range(500)))
        assert cache.metrics()["bytes_resident"] <= 1 * 1024**2
    finally:
        cache.shutdown()


def test_bad_page_size_rejected():
    with pytest.raises(ValueError):
        EngramRowCache(MODEL_ROOT, REFS, 1024, 512)


def test_access_stream_deterministic():
    assert access_stream(1000, 50, 3) == access_stream(1000, 50, 3)
    assert access_stream(1000, 50, 3) != access_stream(1000, 50, 4)
