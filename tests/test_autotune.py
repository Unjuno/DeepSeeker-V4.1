"""Tests for Issue #12 autotune recommender (no weights)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.autotune import (
    recommend,
    recommend_horizon,
    recommend_predictor_budget,
    recommend_slots,
    recommend_workers,
)


def test_slots_math():
    slots, _ = recommend_slots(47_244_640_256, 315_043_840, 18_800_640, 6)
    assert slots == (47_244_640_256 - 315_043_840 - 4 * 1024**3) // 40 // 18_800_640
    assert 6 <= slots <= 384


def test_slots_fallback():
    slots, why = recommend_slots(None, None, None, 6)
    assert slots == 6
    assert "fallback" in why


def test_predictor_picks_smallest_hitting_budget():
    rows = [
        {"predictor": "lfu", "budget": 6, "recall_at_k": 0.2, "wasted_bytes": 10.0},
        {"predictor": "ema", "budget": 12, "recall_at_k": 0.75, "wasted_bytes": 50.0},
        {"predictor": "lru", "budget": 12, "recall_at_k": 0.74, "wasted_bytes": 30.0},
        {"predictor": "ema", "budget": 24, "recall_at_k": 0.9, "wasted_bytes": 90.0},
    ]
    name, budget, _ = recommend_predictor_budget(rows, target_recall=0.7)
    assert (name, budget) == ("lru", 12)  # within 5%, cheaper waste


def test_predictor_unreachable_target():
    rows = [{"predictor": "lfu", "budget": 6, "recall_at_k": 0.2, "wasted_bytes": 10.0}]
    name, budget, why = recommend_predictor_budget(rows, target_recall=0.7)
    assert (name, budget) == ("lfu", 6)
    assert "unreachable" in why


def test_predictor_empty():
    assert recommend_predictor_budget([], 0.7)[0] == "persistence"


def test_horizon_rules():
    h0 = [{"mean_recall": 0.7}]
    assert recommend_horizon(h0, [{"mean_recall": 0.68}])[0] == 1
    assert recommend_horizon(h0, [{"mean_recall": 0.5}])[0] == 0
    assert recommend_horizon([], h0)[0] == 0
    assert recommend_horizon(h0, [])[0] == 0


def test_workers():
    assert recommend_workers(8)[0] == 8
    assert recommend_workers(None)[0] == 4


def test_recommend_end_to_end(tmp_path):
    machine = {"cpu_performance_cores": 8}
    plan = {"operating_bytes": 47_244_640_256, "dense_engram_bytes": {"bytes": 315_043_840}}
    manifest = {
        "expert_stats": {"median_bytes": 18_800_640},
        "architecture": {"num_experts_per_tok": 6},
    }
    sim = [
        {"predictor": "persistence", "budget": 6, "recall_at_k": 0.71, "wasted_bytes": 1e8},
        {"predictor": "ema", "budget": 6, "recall_at_k": 0.48, "wasted_bytes": 2e8},
    ]
    shadow0 = [{"predictor": "persistence", "mean_recall": 0.71, "switch_at": 3}]
    shadow1 = [{"predictor": "persistence", "mean_recall": 0.71, "switch_at": 5}]
    result = recommend(machine, plan, manifest, sim, shadow0, shadow1, target_recall=0.7)
    assert result["schema"] == "deepseeker.autotune/v1"
    assert result["config"]["predictor"] == "persistence"
    assert result["config"]["budget"] == 6
    assert result["config"]["horizon"] == 1
    assert result["config"]["prefetch_enabled"] is True
    assert result["config"]["cpu_workers"] == 8
    assert set(result["rationale"]) == set(result["config"])
