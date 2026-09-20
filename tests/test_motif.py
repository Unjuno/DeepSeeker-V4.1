"""Tests for Issue #14 motifs and block analysis (synthetic traces only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.motif import (
    block_summary,
    block_working_sets,
    layer_bigrams,
    motif_coverage,
)
from deepseeker.simulate import make_predictor
from deepseeker.trace import emit_synthetic, make_header

TINY_CFG = {"n_layers": 3, "n_routed_experts": 8, "top_k": 2}


def tiny_header():
    return make_header("repo", "rev", "synthetic", TINY_CFG, seed=41)


def planted_records():
    """Token 0 routes 0->1->2 through expert 5 at every layer."""
    records = []
    for token in range(3):
        for layer in range(3):
            records.append(
                {"request_id": "r", "token_pos": token, "layer": layer, "experts": [5, 6]}
            )
    return records


def test_bigrams_find_planted_motif():
    pairs = layer_bigrams(planted_records())
    assert pairs[((0, 5), (1, 5))] == 3
    assert pairs[((1, 6), (2, 6))] == 3
    top = {pair for pair, _ in pairs.most_common(8)}
    assert motif_coverage(planted_records(), top) == pytest.approx(1.0)


def test_coverage_empty_motifs():
    assert motif_coverage(planted_records(), set()) == pytest.approx(0.0)


def test_block_sets_sublinear_on_sticky():
    curve = {row["block_size"]: row for row in block_summary(planted_records(), [1, 2, 4], 2)}
    assert curve[1]["mean_unique_experts"] == pytest.approx(2.0)
    assert curve[2]["mean_unique_experts"] == pytest.approx(2.0)
    assert curve[4]["reuse_ratio"] == pytest.approx(2 / (4 * 2))


def test_block_size_floor():
    with pytest.raises(ValueError):
        block_working_sets(planted_records(), 0)


def test_bundle_predictor_covers_bursts():
    predictor = make_predictor("bundle", window=3)
    assert predictor.name == "bundle"
    # Alternating pairs: persistence sees only the other pair, bundle recalls both.
    records = []
    for token in range(6):
        experts = [0, 1] if token % 2 == 0 else [2, 3]
        records.append({"request_id": "r", "token_pos": token, "layer": 0, "experts": experts})
    hits = 0
    for record in records:
        guessed = predictor.predict("r", record["token_pos"], 0, 4)
        hits += len(set(record["experts"]) & set(guessed))
        predictor.update("r", record["token_pos"], 0, record["experts"])
    assert hits == 2 * 6 - 4  # first two tokens cold, rest fully covered


def test_bundle_beats_persistence_on_bursts():
    from deepseeker.simulate import evaluate

    records = []
    for token in range(8):
        experts = [0, 1] if token % 2 == 0 else [2, 3]
        records.append({"request_id": "r", "token_pos": token, "layer": 0, "experts": experts})
    bundle = evaluate(records, make_predictor("bundle", window=4), 2, 2)
    persist = evaluate(records, make_predictor("persistence"), 2, 2)
    assert bundle["recall_at_k"] > persist["recall_at_k"]


def test_motif_stats_on_synthetic():
    header = tiny_header()
    records = emit_synthetic(header, 2, 6, seed=42)
    pairs = layer_bigrams(records)
    assert len(pairs) > 0
    top = {pair for pair, _ in pairs.most_common(10)}
    assert 0.0 <= motif_coverage(records, top) <= 1.0
