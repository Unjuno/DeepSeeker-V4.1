"""Tests for Issue #35 partition policy (synthetic measurements)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.partition import TASKS, decide


def test_winner_and_fallback():
    policy = decide({"router-matvec": {"cpu": 1.0, "gpu": 0.1}})["policy"]
    assert policy["router-matvec"]["placement"] == "gpu"
    assert policy["router-matvec"]["fallback"] == "cpu"


def test_tie_prefers_cpu():
    policy = decide({"topk-select": {"cpu": 1.0, "gpu": 1.03}})["policy"]
    assert policy["topk-select"]["placement"] == "cpu"


def test_missing_tasks_default_cpu():
    policy = decide({})["policy"]
    for task in TASKS:
        assert policy[task]["placement"] == "cpu"
    assert policy["ane"]["placement"] == "unused"


def test_schema_present():
    assert decide({})["schema"] == "deepseeker.partition/v1"
