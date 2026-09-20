"""Tests for Issue #13 budget allocation (synthetic traces only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.allocate import (
    allocate_equal,
    allocate_greedy,
    allocate_proportional,
    evaluate_allocation,
    hit_curve,
    unique_demand,
)
from deepseeker.trace import make_header

TINY_CFG = {"n_layers": 4, "n_routed_experts": 16, "top_k": 2}


def tiny_header():
    return make_header("repo", "rev", "synthetic", TINY_CFG, seed=31)


def skewed_records():
    """Layers 0-2 use 2 sticky experts; layer 3 cycles 6 repeating experts."""
    records = []
    for token in range(6):
        records.append({"request_id": "r", "token_pos": token, "layer": 0, "experts": [0, 1]})
        records.append({"request_id": "r", "token_pos": token, "layer": 1, "experts": [0, 1]})
        records.append({"request_id": "r", "token_pos": token, "layer": 2, "experts": [2, 3]})
        records.append(
            {
                "request_id": "r",
                "token_pos": token,
                "layer": 3,
                "experts": [token % 3, (token % 3) + 6],
            }
        )
    return records


def test_equal_splits_evenly():
    assert allocate_equal(10, 4, 2) == {0: 3, 1: 3, 2: 2, 3: 2}
    with pytest.raises(ValueError):
        allocate_equal(7, 4, 2)


def test_proportional_follows_demand():
    split = allocate_proportional(20, {0: 2, 1: 2, 2: 2, 3: 12}, 2)
    assert sum(split.values()) == 20
    assert all(v >= 2 for v in split.values())
    assert split[3] > split[0]
    with pytest.raises(ValueError):
        allocate_proportional(3, {0: 2, 1: 2}, 2)


def test_greedy_beats_equal_on_skew():
    records = skewed_records()
    demand = unique_demand(records, 4)
    assert demand[3] > demand[0]
    grid = [2, 4, 8]
    curves = {layer: hit_curve(records, layer, grid, 2) for layer in range(4)}
    for layer in range(4):
        assert list(curves[layer]) == grid
    greedy = allocate_greedy(curves, 16, 2)
    equal = allocate_equal(16, 4, 2)
    assert sum(greedy.values()) == 16
    assert greedy[3] > greedy[0]  # skewed demand must skew the split
    g = evaluate_allocation(records, greedy, 2)["hit_rate"]
    e = evaluate_allocation(records, equal, 2)["hit_rate"]
    assert g >= e


def test_evaluate_allocation_empty_rejected():
    with pytest.raises(ValueError):
        evaluate_allocation(skewed_records(), {}, 2)


def test_per_layer_budgets_honored():
    from deepseeker.resident import simulate

    records = skewed_records()
    row = simulate(records, "demand_lru", 2, 2, budgets_per_layer={3: 12})
    assert row["budgets_per_layer"] == {3: 12}
    uniform = simulate(records, "demand_lru", 2, 2)
    assert row["hit_rate"] >= uniform["hit_rate"]
