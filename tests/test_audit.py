"""Tests for Issue #18 FreeToken audit artifacts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.moe_align import align_block_size

MAP_PATH = REPO_ROOT / "profiles" / "audit" / "freetoken-map.json"


def test_map_schema():
    table = load_json(MAP_PATH)
    assert table["schema"] == "deepseeker.audit-map/v1"
    assert table["subject"]["license"] == "Apache-2.0"
    assert "FreeToken" in table["subject"]["copyright"]
    assert "mismatch" in table["subject"]["note"].lower()
    assert len(table["verdicts"]) >= 10
    for row in table["verdicts"]:
        assert set(row) == {"subsystem", "source", "verdict", "rationale", "deepseeker_issue"}
        assert row["verdict"] in {"reuse", "adapt", "reject"}
        assert row["source"].startswith(("moe/", "kernel/", "checkpoint/", "kvcache/",
                                          "scheduler/", "engine/", "attention/"))
    verdicts = {r["verdict"] for r in table["verdicts"]}
    assert verdicts == {"reuse", "adapt", "reject"}


def test_align_docstring_example():
    topk = np.array([[1, 2, 3], [0, 1, 3], [0, 2, 3], [0, 1, 2]])
    sorted_ids, expert_ids, n_pad = align_block_size(topk, 4, 4)
    assert n_pad == 16
    assert list(expert_ids) == [0, 1, 2, 3]
    real = sorted(int(i) for i in sorted_ids if i != 12)
    assert real == list(range(12))
    assert all(int(sorted_ids[i]) == 12 for i in (3, 7, 11, 15))


def test_align_properties_on_random():
    rng = np.random.default_rng(0)
    topk = rng.integers(0, 384, size=(256, 6))
    start = time.perf_counter()
    sorted_ids, expert_ids, n_pad = align_block_size(topk, 16, 384)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert n_pad % 16 == 0
    assert len(expert_ids) == n_pad // 16
    flat = topk.reshape(-1)
    for expert in range(384):
        run = [int(i) for i in sorted_ids if i != len(flat) and flat[i] == expert]
        assert len(run) == int((flat == expert).sum())
    print(f"\nmoe_align numpy (256tok x top6, E=384): {elapsed_ms:.1f}ms")
    assert elapsed_ms < 5000


def test_align_rejects_bad_input():
    with pytest.raises(ValueError):
        align_block_size(np.array([1, 2, 3]), 4, 4)
    with pytest.raises(ValueError):
        align_block_size(np.array([[0, 99]]), 4, 4)
