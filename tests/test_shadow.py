"""Tests for Issue #10 shadow learner (synthetic traces only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.shadow import run_shadow, switch_on
from deepseeker.simulate import EMA, PREDICTORS, evaluate
from deepseeker.trace import emit_synthetic, make_header

TINY_CFG = {"n_layers": 2, "n_routed_experts": 8, "top_k": 2}


def tiny_header():
    return make_header("repo", "rev", "synthetic", TINY_CFG, seed=9)


def sticky_records():
    records = []
    for token in range(5):
        for layer in range(2):
            records.append(
                {"request_id": "r", "token_pos": token, "layer": layer, "experts": [1, 2]}
            )
    return records


def test_horizon_rejected():
    with pytest.raises(ValueError):
        run_shadow(sticky_records(), "persistence", 2, 2, horizon=5)


def test_horizon0_matches_p2_recall():
    header = tiny_header()
    records = emit_synthetic(header, 3, 6, seed=4)
    shadow = run_shadow(records, "ema", 2, 2, horizon=0)
    offline = evaluate(records, EMA(), 2, 2)
    assert shadow["mean_recall"] == pytest.approx(offline["recall_at_k"])
    assert shadow["tokens"] == 18


def test_shadow_switches_on_sticky():
    run = run_shadow(sticky_records(), "persistence", 2, 2, horizon=1)
    at = switch_on(run["log"], window=2, min_recall=0.9, max_waste_per_token=10**9)
    assert at == 2  # tokens 0-1 cold/warm, token 2 first with full history
    assert run["log"][at]["recall"] == pytest.approx(1.0)


def test_shadow_never_switches_on_strict():
    run = run_shadow(sticky_records(), "persistence", 2, 2, horizon=1)
    assert switch_on(run["log"], window=2, min_recall=1.1, max_waste_per_token=10**9) is None
    assert switch_on(run["log"], window=99, min_recall=0.0, max_waste_per_token=10**9) is None


def test_switch_on_rejects_bad_window():
    with pytest.raises(ValueError):
        switch_on([], window=0, min_recall=0.5, max_waste_per_token=1.0)


def test_layer_transition_starves_at_horizon1():
    header = tiny_header()
    records = emit_synthetic(header, 3, 6, seed=8)
    ahead = run_shadow(records, "layer_transition", 2, 2, horizon=1)
    assert ahead["mean_recall"] == pytest.approx(0.0)


def test_all_predictors_shadow_cleanly():
    header = tiny_header()
    records = emit_synthetic(header, 2, 4, seed=12)
    for name in PREDICTORS:
        for horizon in (0, 1):
            run = run_shadow(records, name, 2, 2, horizon=horizon)
            assert run["tokens"] == 8
            assert 0.0 <= run["mean_recall"] <= 1.0
