"""Tests for Issue #19 checkpoint IO helpers (synthetic manifest/file)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.checkpoint_io import (
    engram_ranges,
    expert_ranges,
    flatten_ranges,
    merge_group,
    set_nocache,
    summarize_latency,
)

FAKE_MANIFEST = {
    "tensors": [
        {"name": "a", "subsystem": "routed-expert", "layer": 5, "expert": 7,
         "shard": "s1", "file_range": [100, 200]},
        {"name": "b", "subsystem": "routed-expert", "layer": 5, "expert": 7,
         "shard": "s1", "file_range": [0, 50]},
        {"name": "c", "subsystem": "routed-expert", "layer": 5, "expert": 8,
         "shard": "s1", "file_range": [300, 350]},
        {"name": "d", "subsystem": "routed-expert", "layer": 6, "expert": 7,
         "shard": "s2", "file_range": [0, 60]},
        {"name": "e", "subsystem": "engram", "layer": None, "expert": None,
         "shard": "s1", "file_range": [1000, 2000]},
        {"name": "f", "subsystem": "attention", "layer": 5, "expert": None,
         "shard": "s1", "file_range": [2000, 2100]},
    ]
}


def test_expert_ranges_sorted():
    assert expert_ranges(FAKE_MANIFEST, 5, 7) == {"s1": [(0, 50), (100, 200)]}
    assert expert_ranges(FAKE_MANIFEST, 5, 9) == {}


def test_merge_and_flatten():
    merged = merge_group(FAKE_MANIFEST, [(5, 7), (5, 8), (6, 7)])
    assert merged == {"s1": [(0, 50), (100, 200), (300, 350)], "s2": [(0, 60)]}
    jobs = flatten_ranges(merged)
    assert jobs[0] == ("s1", 0, 50)
    assert jobs[-1] == ("s2", 0, 60)


def test_engram_limit():
    assert engram_ranges(FAKE_MANIFEST, 1) == {"s1": [(1000, 2000)]}
    assert engram_ranges(FAKE_MANIFEST, 0) == {}
    clipped = engram_ranges(FAKE_MANIFEST, 1, max_bytes=100)
    assert clipped == {"s1": [(1000, 1100)]}
    offset = engram_ranges(FAKE_MANIFEST, 1, base_offset=500)
    assert offset == {"s1": [(1500, 2000)]}


def test_summarize_latency():
    stats = summarize_latency([0.3, 0.1, 0.2, 0.4])
    assert stats["p50_s"] == pytest.approx(0.3)
    assert stats["p95_s"] == pytest.approx(0.4)
    assert stats["mean_s"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        summarize_latency([])


def test_set_nocache_and_pread_roundtrip(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(bytes(range(256)) * 4)
    fd = os.open(path, os.O_RDONLY)
    try:
        assert set_nocache(fd) is True
        assert os.pread(fd, 100, 10) == bytes(range(10, 110))
    finally:
        os.close(fd)
