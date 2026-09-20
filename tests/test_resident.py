"""Tests for Issue #11 resident-pool simulator (synthetic traces only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.resident import POLICIES, compare, simulate
from deepseeker.trace import emit_synthetic, make_header

TINY_CFG = {"n_layers": 2, "n_routed_experts": 8, "top_k": 2}


def tiny_header():
    return make_header("repo", "rev", "synthetic", TINY_CFG, seed=21)


def sticky_records():
    records = []
    for token in range(5):
        for layer in range(2):
            records.append(
                {"request_id": "r", "token_pos": token, "layer": layer, "experts": [1, 2]}
            )
    return records


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        simulate(sticky_records(), "oracle", 2, 2)


def test_budget_floor():
    with pytest.raises(ValueError):
        simulate(sticky_records(), "demand_lru", 0, 2)


def test_sticky_warms_up():
    # Only cold misses: 2 experts x 2 layers, then everything hits.
    demand = simulate(sticky_records(), "demand_lru", 2, 2)
    assert demand["hit_rate"] == pytest.approx(1 - 4 / 20)
    belady = simulate(sticky_records(), "belady", 2, 2)
    assert belady["hit_rate"] == pytest.approx(1 - 4 / 20)
    assert belady["churn_experts"] == 0


def test_prefetch_beats_demand_on_sticky():
    demand = simulate(sticky_records(), "demand_lru", 2, 2)
    pre = simulate(sticky_records(), "prefetch", 2, 2, predictor_name="persistence")
    assert pre["hit_rate"] >= demand["hit_rate"]


def test_belady_bounds_demand_on_synthetic():
    header = tiny_header()
    records = emit_synthetic(header, 3, 10, seed=22)
    for budget in (2, 4):
        demand = simulate(records, "demand_lru", budget, 2)
        optimal = simulate(records, "belady", budget, 2)
        assert optimal["hit_rate"] >= demand["hit_rate"]


def test_metrics_accounting():
    header = tiny_header()
    records = emit_synthetic(header, 2, 4, seed=23)
    row = simulate(records, "demand_lru", 2, 2, expert_bytes=100)
    assert row["uses"] == len(records) * 2
    assert row["miss_bytes_per_record"] == pytest.approx(row["miss_per_record"] * 100)
    assert row["hit_rate"] + row["miss_per_record"] / 2 == pytest.approx(1.0)


def test_compare_covers_policies_and_budgets():
    header = tiny_header()
    records = emit_synthetic(header, 2, 3, seed=24)
    rows = compare(records, [2, 4], 2)
    assert {(r["policy"], r["budget"]) for r in rows} == {
        (p, b) for p in POLICIES for b in (2, 4)
    }
