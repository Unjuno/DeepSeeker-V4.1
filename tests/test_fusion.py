"""Tests for Issue #34 fusion accounting (no GPU needed)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.fusion import GEOMETRY, intermediate_bytes, launch_counts, verdict


def test_intermediate_math():
    h, i = GEOMETRY["hidden"], GEOMETRY["inter"]
    inter = intermediate_bytes()
    assert inter["gate_bytes"] == i * 2
    assert inter["down_out_bytes"] == h * 2
    assert inter["unfused_bytes"] == (3 * i + h) * 2
    assert inter["fused_bytes"] == h * 2
    assert 0.5 < inter["saved_ratio"] < 1.0


def test_launch_counts():
    launches = launch_counts()
    assert launches["naive_evals"] == 6 * 40 * 4
    assert launches["fused_evals"] == 6 * 40
    assert launches["ideal_evals"] == 40


def test_verdict_rule():
    assert verdict("x", 1.0, 2.0, True, True)["verdict"] == "PASS"
    assert verdict("x", 2.0, 2.0, True, True)["verdict"] == "REVERT"
    assert verdict("x", 1.0, 2.0, False, True)["verdict"] == "REVERT"
    assert verdict("x", 1.0, 2.0, True, False)["verdict"] == "REVERT"
    assert verdict("x", 1.0, 2.0, True, True)["speedup"] == 2.0
