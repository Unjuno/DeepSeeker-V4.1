"""Pure cache-budget policy checks; no MLX, PyTorch, or checkpoint needed."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.cache_policy import expert_lru_budget


def test_static_pins_can_reserve_part_of_total_cache():
    assert expert_lru_budget(16, has_static_pins=True) == 16
    assert expert_lru_budget(256, has_static_pins=True) == 256


def test_unpinned_working_set_keeps_240_entry_floor():
    assert expert_lru_budget(16, has_static_pins=False) == 240
    assert expert_lru_budget(256, has_static_pins=False) == 256
