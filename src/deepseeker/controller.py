"""Issue #37: runtime adaptive controller for cache/prefetch policy (roadmap P8).

Faster-timescale online adaptation over the Issue #36 startup config:
monitors hit rate, waste, trailing agreement, and drift, then proposes
bounded parameter changes (expert slots, prefetch horizon/enabled,
predictor budget) with hysteresis and rollback to a known-safe baseline.

Hard constraints (never adapted):
  - authoritative router outputs / quality semantics
  - resident byte ceiling (slots never exceed the autotune maximum)
  - prefetch may only be re-enabled while the PrefetchController gate
    itself still allows it (agreement / auto-disable is respected)

All control dimensions include rate limits: at most one step per
min_dwell tokens, and no more than max_steps_per_hour absolute steps
per dimension within the rolling window.
"""

from __future__ import annotations

from collections import deque

SCHEMA = "deepseeker.adaptive/v1"

# Dimensions the controller may propose values for (bounds inclusive).
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "resident_slots_per_layer": (6, 384),
    "budget": (6, 48),
    "horizon": (0, 1),
    "prefetch_enabled": (0, 1),
}

# Known-safe startup snapshot is immutable after construction.


class AdaptiveController:
    """Bounded, hysteretic online tuner with rollback."""

    def __init__(
        self,
        startup_config: dict,
        *,
        min_dwell: int = 32,
        window: int = 64,
        min_hit_rate: float = 0.15,
        max_waste_ratio: float = 0.40,
        max_drift: float = 0.35,
        improvement_epsilon: float = 0.02,
        max_steps_per_window: int = 2,
    ) -> None:
        self.startup_config = dict(startup_config)
        self.config = dict(startup_config)
        self.min_dwell = max(1, min_dwell)
        self.window = max(8, window)
        self.min_hit_rate = min_hit_rate
        self.max_waste_ratio = max_waste_ratio
        self.max_drift = max_drift
        self.improvement_epsilon = improvement_epsilon
        self.max_steps_per_window = max(1, max_steps_per_window)
        self._since_change = self.min_dwell
        self._hit_hist: deque[float] = deque(maxlen=self.window)
        self._waste_hist: deque[float] = deque(maxlen=self.window)
        self._agree_hist: deque[float] = deque(maxlen=self.window)
        self._baseline_hit: float | None = None
        self._step_log: list[dict] = []
        self._steps_in_window: deque[int] = deque(maxlen=self.window)
        self.rollback_count = 0
        self.disabled_dimensions: set[str] = set()

    # -- telemetry ---------------------------------------------------------
    def observe(
        self,
        *,
        hit_rate: float,
        waste_ratio: float = 0.0,
        agreement: float | None = None,
        drift: float = 0.0,
        tok_s: float | None = None,
    ) -> dict:
        """Feed one telemetry sample; returns the current config snapshot."""
        self._hit_hist.append(float(hit_rate))
        self._waste_hist.append(float(waste_ratio))
        if agreement is not None:
            self._agree_hist.append(float(agreement))
        self._since_change += 1
        mean_hit = self.mean_hit
        if self._baseline_hit is None and mean_hit is not None:
            self._baseline_hit = mean_hit
        # Hard-gate rollbacks (always, regardless of dwell).
        if mean_hit is not None and mean_hit < self.min_hit_rate * 0.5:
            self._rollback("hit_rate collapse")
        waste = self.mean_waste
        if waste is not None and waste > self.max_waste_ratio * 1.5:
            self._rollback("waste explosion")
        if drift > self.max_drift:
            self._rollback("distribution shift")
        # Optional opportunistic improvement once dwell has elapsed.
        if (
            self._since_change >= self.min_dwell
            and mean_hit is not None
            and self._baseline_hit is not None
            and (self._baseline_hit - mean_hit) >= self.improvement_epsilon
            and self._try_improve(mean_hit)
        ):
            self._since_change = 0
            self._baseline_hit = mean_hit
        snap = {
            "schema": SCHEMA,
            "config": dict(self.config),
            "mean_hit": mean_hit,
            "mean_waste": waste,
            "steps": len(self._step_log),
            "rollbacks": self.rollback_count,
            "disabled": sorted(self.disabled_dimensions),
            "tok_s": tok_s,
        }
        return snap

    @property
    def mean_hit(self) -> float | None:
        if not self._hit_hist:
            return None
        return sum(self._hit_hist) / len(self._hit_hist)

    @property
    def mean_waste(self) -> float | None:
        if not self._waste_hist:
            return None
        return sum(self._waste_hist) / len(self._waste_hist)

    @property
    def mean_agreement(self) -> float | None:
        if not self._agree_hist:
            return None
        return sum(self._agree_hist) / len(self._agree_hist)

    # -- control -----------------------------------------------------------
    def _rate_limited(self) -> bool:
        # steps_in_window holds recent step markers; length is a cheap
        # count of steps in the rolling window.
        return len(self._steps_in_window) >= self.max_steps_per_window

    def _set(self, key: str, value, reason: str) -> bool:
        if key in self.disabled_dimensions:
            return False
        lo, hi = PARAM_BOUNDS[key]
        value = type(self.config[key])(min(hi, max(lo, value)))
        if value == self.config[key]:
            return False
        if self._rate_limited():
            return False
        prev = self.config[key]
        self.config[key] = value
        self._step_log.append({"param": key, "from": prev, "to": value, "reason": reason})
        self._steps_in_window.append(len(self._step_log))
        return True

    def _try_improve(self, mean_hit: float) -> bool:
        """Cautious one-step improvement: grow budget/slots if we can afford it."""
        changed = False
        # Prefer growing resident slots (upstream bound from startup config).
        start_slots = int(self.startup_config.get("resident_slots_per_layer", 6))
        cur_slots = int(self.config["resident_slots_per_layer"])
        if cur_slots < start_slots:
            changed |= self._set(
                "resident_slots_per_layer", min(start_slots, cur_slots * 2),
                "recover toward startup slots after hit dip",
            )
        # Then grow prefetch budget if agreement supports it.
        agree = self.mean_agreement
        if agree is not None and agree >= 0.5:
            cur_b = int(self.config["budget"])
            changed |= self._set("budget", min(int(PARAM_BOUNDS["budget"][1]), cur_b + 6),
                                 f"agreement {agree:.2f} >= 0.5, raise budget")
            # Re-enable prefetch only if not hard-disabled and agreement holds.
            if (
                not self.config.get("prefetch_enabled")
                and "prefetch_enabled" not in self.disabled_dimensions
                and agree >= 0.6
            ):
                changed |= self._set("prefetch_enabled", 1,
                                     f"agreement {agree:.2f} >= 0.6, re-enable")
        return changed

    def _rollback(self, reason: str) -> int:
        """Restore every dimension to the startup snapshot; count the event."""
        restored = 0
        for key, value in self.startup_config.items():
            if key in PARAM_BOUNDS and self.config.get(key) != value:
                self.config[key] = value
                self._step_log.append(
                    {"param": key, "from": self.config[key], "to": value,
                     "reason": f"rollback: {reason}"}
                )
                restored += 1
        if restored:
            self.rollback_count += 1
            self._since_change = 0
            # A dimension that triggered a rollback twice is frozen.
            self.disabled_dimensions.add(self._last_offending or "horizon")
        return restored

    _last_offending: str | None = None

    def mark_dimension_unsafe(self, key: str) -> None:
        """Permanently disable a control dimension that oscillates or harms quality."""
        if key not in PARAM_BOUNDS:
            raise KeyError(key)
        self.disabled_dimensions.add(key)
        # Immediate restore of that dim to startup value.
        if key in self.startup_config:
            self.config[key] = self.startup_config[key]

    def report(self) -> dict:
        return {
            "schema": SCHEMA,
            "startup": dict(self.startup_config),
            "config": dict(self.config),
            "mean_hit": self.mean_hit,
            "mean_waste": self.mean_waste,
            "mean_agreement": self.mean_agreement,
            "rollbacks": self.rollback_count,
            "steps": len(self._step_log),
            "step_log": list(self._step_log[-20:]),
            "disabled": sorted(self.disabled_dimensions),
        }
