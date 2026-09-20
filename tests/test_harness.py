"""Tests for Issue #17 golden harness (no model needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.harness import (
    SCHEMA,
    build_result,
    compare_logits,
    compare_tokens,
    equivalence,
    summarize,
)


def test_summarize():
    stats = summarize([3.0, 1.0, 2.0])
    assert stats == {"n": 3, "median": 2.0, "min": 1.0, "max": 3.0, "range": 2.0}
    with pytest.raises(ValueError):
        summarize([])


def test_tokens_exact_and_budget():
    assert compare_tokens([1, 2, 3], [1, 2, 3])["verdict"] == "PASS"
    assert compare_tokens([1, 2, 3], [1, 9, 3])["verdict"] == "FAIL"
    budgeted = compare_tokens([1, 2, 3], [1, 9, 3], max_mismatch=1)
    assert budgeted["verdict"] == "PASS"
    assert compare_tokens([1, 2], [1, 2, 3])["mismatches"] == 1


def test_logits_bands():
    gold = [[0.1, 0.2], [0.3, 0.4]]
    same = [[0.1, 0.2], [0.3, 0.4]]
    assert compare_logits(gold, same)["verdict"] == "PASS"
    drift = [[0.105, 0.2], [0.3, 0.4]]
    assert compare_logits(drift, drift)["verdict"] == "PASS"
    assert compare_logits(gold, [[0.1 + 0.005, 0.2], [0.3, 0.4]])["verdict"] == "UNCERTAIN"
    assert compare_logits(gold, [[0.5, 0.2], [0.3, 0.4]])["verdict"] == "FAIL"
    assert compare_logits(gold, [[0.1]])["verdict"] == "FAIL"


def test_equivalence_matrix():
    gold_t, cand_t = [1, 2], [1, 2]
    gold_l, cand_l = [[0.1]], [[0.1001]]
    assert equivalence(gold_t, cand_t)["verdict"] == "UNCERTAIN"  # no logits cap
    assert equivalence(gold_t, cand_t, gold_l, cand_l)["verdict"] == "PASS"
    assert equivalence([1], [2])["verdict"] == "FAIL"
    assert equivalence(gold_t, cand_t, gold_l, [[0.9]])["verdict"] == "FAIL"


def test_build_result_schema():
    result = build_result(
        machine_id="m", revision="r", backend="b", corpus_id="c",
        settings={}, ttft_s=[0.1, 0.2], decode_tok_s=[9.0, 11.0],
        accepted_tokens=64,
        equivalence_report={"verdict": "PASS"},
    )
    assert result["schema"] == SCHEMA
    assert result["decode_tok_s"]["median"] == 10.0
    assert result["accepted_tokens"] == 64
    with pytest.raises(ValueError):
        build_result("m", "r", "b", "c", {}, [], [1.0], 1, {})


def test_corpus_committed():
    corpus = load_json(REPO_ROOT / "profiles" / "bench" / "corpus.json")
    assert corpus["schema"] == "deepseeker.bench-corpus/v1"
    assert len(corpus["cases"]) >= 3
    assert corpus["sampling"]["temperature"] == 0.0
