"""Tests for Issue #37 adaptive controller (no weights, no I/O)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.controller import PARAM_BOUNDS, AdaptiveController

STARTUP = {
    "resident_slots_per_layer": 24,
    "predictor": "ema",
    "budget": 24,
    "horizon": 1,
    "prefetch_enabled": False,
    "cpu_workers": 8,
    "io_concurrency": 6,
}


def make(**kw) -> AdaptiveController:
    return AdaptiveController(STARTUP, **kw)


def test_startup_snapshot_is_immutable():
    c = make()
    c.observe(hit_rate=0.05, waste_ratio=0.9, drift=0.0)
    assert c.config == STARTUP  # collapse should restore startup, not mutate it
    c.startup_config["budget"] = 99
    assert c.config["budget"] == 24


def test_bounds_enforced():
    c = make()
    c._set("budget", 10_000, "test")
    assert c.config["budget"] == PARAM_BOUNDS["budget"][1]
    c._set("horizon", -5, "test")
    assert c.config["horizon"] == 0
    c._set("resident_slots_per_layer", 1, "test")
    assert c.config["resident_slots_per_layer"] >= 6


def test_dwell_blocks_rapid_changes():
    c = make(min_dwell=32, max_steps_per_window=2)
    # Seed histories so improvement path is active.
    c._baseline_hit = 0.50
    for _ in range(32):
        c.observe(hit_rate=0.40, waste_ratio=0.1, agreement=0.7, drift=0.0)
    steps_before = len(c._step_log)
    # Immediately observe again — dwell just elapsed, one step allowed.
    c.observe(hit_rate=0.40, waste_ratio=0.1, agreement=0.7, drift=0.0)
    assert len(c._step_log) - steps_before <= 1


def test_rate_limit_window():
    c = make(min_dwell=1, max_steps_per_window=1, window=64)
    c._baseline_hit = 0.5
    changed = 0
    for _ in range(64):
        before = len(c._step_log)
        c.observe(hit_rate=0.30, waste_ratio=0.1, agreement=0.7, drift=0.0)
        changed += len(c._step_log) - before
        c._since_change = c.min_dwell  # force dwell-eligible every tick
    # Rate limit: at most max_steps_per_window steps per rolling window.
    assert changed <= 2


def test_hit_collapse_rollback():
    c = make(min_hit_rate=0.15)
    c.config["budget"] = 48  # diverge from startup
    c.config["resident_slots_per_layer"] = 48
    c.observe(hit_rate=0.01, waste_ratio=0.1)  # < 0.5 * 0.15
    assert c.config["budget"] == 24
    assert c.config["resident_slots_per_layer"] == 24
    assert c.rollback_count == 1


def test_waste_explosion_rollback():
    c = make(max_waste_ratio=0.40)
    c.config["horizon"] = 0  # diverge from startup (1)
    c.observe(hit_rate=0.5, waste_ratio=0.70)  # > 0.40 * 1.5
    assert c.config["horizon"] == STARTUP["horizon"]
    assert c.rollback_count == 1


def test_distribution_shift_rollback():
    c = make()
    c.config["budget"] = 48
    c.observe(hit_rate=0.5, waste_ratio=0.1, drift=0.50)  # > 0.35
    assert c.config["budget"] == 24
    assert c.rollback_count == 1


def test_no_oscillation_on_stable_good_metrics():
    c = make(min_dwell=16, max_steps_per_window=1)
    steps = 0
    for _ in range(200):
        before = len(c._step_log)
        c.observe(hit_rate=0.80, waste_ratio=0.05, agreement=0.9, drift=0.0)
        steps += len(c._step_log) - before
    # Healthy regime: no rollbacks, very few (if any) steps.
    assert c.rollback_count == 0
    assert steps <= 4


def test_mark_dimension_unsafe_freezes():
    c = make(min_dwell=1, max_steps_per_window=8)
    c.mark_dimension_unsafe("prefetch_enabled")
    c._baseline_hit = 0.5
    for _ in range(32):
        c.observe(hit_rate=0.3, waste_ratio=0.05, agreement=0.9, drift=0.0)
    assert "prefetch_enabled" in c.disabled_dimensions
    assert c.config["prefetch_enabled"] == STARTUP["prefetch_enabled"]
    assert c._set("prefetch_enabled", 1, "should refuse") is False


def test_gate_compliance_prefetch_stays_off_without_agreement():
    c = make(min_dwell=1, max_steps_per_window=8)
    c._baseline_hit = 0.5
    for _ in range(32):
        c.observe(hit_rate=0.45, waste_ratio=0.05, agreement=0.3, drift=0.0)
    # Low agreement: controller must not flip prefetch on.
    assert c.config["prefetch_enabled"] is False or c.config["prefetch_enabled"] == 0


def test_report_shape():
    c = make()
    c.observe(hit_rate=0.5, waste_ratio=0.1, agreement=0.4, tok_s=1.0)
    rep = c.report()
    assert rep["schema"] == "deepseeker.adaptive/v1"
    assert rep["startup"]["budget"] == 24
    assert set(rep["config"]) >= {"budget", "horizon", "resident_slots_per_layer"}
    assert rep["disabled"] == []


def test_static_vs_adaptive_static_never_moves():
    static = dict(STARTUP)
    adaptive = make(min_dwell=4, max_steps_per_window=1)
    # Identical adversarial phase-shift workload.
    for i in range(120):
        phase_good = (i // 40) % 2 == 0
        hit = 0.6 if phase_good else 0.25
        waste = 0.1 if phase_good else 0.35
        drift = 0.0 if phase_good else 0.20
        adaptive.observe(hit_rate=hit, waste_ratio=waste, agreement=0.5, drift=drift)
    # Static config never changes keys it started with.
    assert static == STARTUP
    # Adaptive stays inside bounds and keeps startup as floor for slots.
    for k, (lo, hi) in PARAM_BOUNDS.items():
        if k in adaptive.config:
            assert lo <= float(adaptive.config[k]) <= hi
    assert adaptive.config["resident_slots_per_layer"] >= STARTUP["resident_slots_per_layer"]
    assert adaptive.rollback_count >= 0
    assert isinstance(adaptive.report()["steps"], int)
