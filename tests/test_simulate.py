"""Tests for Issue #9 routing simulator (synthetic traces only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.simulate import PREDICTORS, compare, evaluate, make_predictor
from deepseeker.trace import emit_synthetic, make_header

TINY_CFG = {"n_layers": 2, "n_routed_experts": 8, "top_k": 2}


def tiny_header():
    return make_header("repo", "rev", "synthetic", TINY_CFG, seed=3)


def sticky_records():
    """Same expert set every token: persistence must be perfect."""
    records = []
    for token in range(4):
        for layer in range(2):
            records.append(
                {"request_id": "r", "token_pos": token, "layer": layer, "experts": [1, 2]}
            )
    return records


def test_all_predictors_instantiate():
    assert set(PREDICTORS) == {
        "persistence",
        "lru",
        "lfu",
        "ema",
        "token_transition",
        "layer_transition",
        "bundle",
    }
    for name in PREDICTORS:
        assert make_predictor(name).name == name
    with pytest.raises(ValueError):
        make_predictor("oracle")


def test_persistence_perfect_on_sticky():
    rows = {r["predictor"]: r for r in compare(sticky_records(), [2], 2)}
    assert rows["persistence"]["recall_at_k"] == pytest.approx(0.75)
    # First token has no history: 2 layers miss, remaining 6 hit fully.
    assert rows["persistence"]["exact_set"] == pytest.approx(0.75)


def test_cold_start_has_zero_recall():
    result = evaluate(sticky_records()[:2], make_predictor("lfu"), 2, 2)
    assert result["recall_at_k"] == 0.0
    assert result["records"] == 2


def test_budget_monotonic_on_synthetic():
    header = tiny_header()
    records = emit_synthetic(header, n_requests=4, tokens_per_request=8, seed=5)
    recalls = [
        evaluate(records, make_predictor("lfu"), k, 2)["recall_at_k"] for k in (2, 4, 8)
    ]
    assert recalls[0] <= recalls[1] <= recalls[2]
    assert recalls[2] > 0.8  # budget covers the pool after warmup


def test_false_pos_and_bytes_accounting():
    result = evaluate(sticky_records(), make_predictor("lfu"), 4, 2, expert_bytes=100)
    assert result["false_pos"] >= 0.0
    assert result["wasted_bytes"] == pytest.approx(result["false_pos"] * 100)


def test_evaluate_counts_records():
    header = tiny_header()
    records = emit_synthetic(header, 2, 3, seed=6)
    result = evaluate(records, make_predictor("ema"), 2, 2)
    assert result["records"] == len(records) == 2 * 3 * 2
