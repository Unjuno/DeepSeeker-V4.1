"""Issue #28: unified expert/KV/Engram/staging scheduler (roadmap P5).

One byte budget, four resource classes, explicit priority order:

  demand expert miss  highest (blocks decode; never shed)
  kv_grow             context growth (shed only by failing the request)
  prefetch            gated predictions (first shed under pressure)
  engram_page         background rows (second shed; refetchable)

Joint policy: KV growth steals from expert slots and Engram before
failing; cross-cache eviction follows the same order. Independent
(fixed-partition) mode keeps every class on its reservation for the
A/B comparison. Starvation guard: any class unserved for `starve_ops`
operations gets one boosted service.

Metric optimized: stall events (expert + engram misses served from
storage) and bytes per served use — not isolated hit rates.
"""

from __future__ import annotations

from deepseeker.kvmodel import cache_bytes

SCHEMA = "deepseeker.scheduler/v1"
PRIORITY = ("demand", "kv_grow", "prefetch", "engram")


class UnifiedScheduler:
    """Byte-budget arbiter over live pool/engram/KV components."""

    def __init__(
        self,
        total_budget_bytes: int,
        geometry: dict,
        expert_bytes: int,
        n_layers: int,
        engram_bytes: int = 64 * 1024**2,
        staging_bytes: int = 512 * 1024**2,
        starve_ops: int = 200,
        mode: str = "joint",
    ) -> None:
        if total_budget_bytes < 1:
            raise ValueError("budget must be >= 1")
        if mode not in ("joint", "independent"):
            raise ValueError("mode must be joint or independent")
        self._total = total_budget_bytes
        self._geometry = geometry
        self._expert_bytes = expert_bytes
        self._n_layers = n_layers
        self._engram_reservation = engram_bytes
        self._staging_reservation = staging_bytes
        self._starve_ops = starve_ops
        self._mode = mode
        self._expert_slots = 0
        self._served = dict.fromkeys(PRIORITY, 0)
        self._since_service = dict.fromkeys(PRIORITY, 0)
        self._sheds = dict.fromkeys(PRIORITY, 0)
        self._stalls = 0
        self._bytes_per_use: list[float] = []
        self.boosts = 0

    def reserve(self, batch: int, seq_len: int) -> dict:
        """Partition the budget for a request context; returns the plan."""
        kv = cache_bytes(self._geometry, batch, seq_len)["total_bytes"]
        fixed = kv + self._engram_reservation + self._staging_reservation
        if fixed >= self._total:
            raise RuntimeError(
                f"fixed costs {fixed}B exceed budget {self._total}B (KV growth blocked)"
            )
        self._expert_slots = max(1, int((self._total - fixed) // self._n_layers // self._expert_bytes))
        return {
            "kv_bytes": kv,
            "engram_bytes": self._engram_reservation,
            "staging_bytes": self._staging_reservation,
            "expert_slots_per_layer": self._expert_slots,
            "expert_pool_bytes": self._expert_slots * self._n_layers * self._expert_bytes,
        }

    def admit_class(self, cls: str, want_bytes: int, used_bytes: int) -> bool:
        """Backpressure: demand always passes; others shed below them first.

        Joint mode records sheds against every lower-priority class to
        make room; independent mode refuses under pressure. Engram (the
        bottom) is refused when pressured with nothing left to shed.
        """
        if cls not in PRIORITY:
            raise ValueError(f"unknown class {cls!r}")
        if cls == "demand":
            return True
        if used_bytes + want_bytes <= self._total:
            return True
        if self._mode == "independent":
            return False
        victims = PRIORITY[PRIORITY.index(cls) + 1 :]
        if not victims:
            return False
        for victim in victims:
            self._sheds[victim] += 1
        return True

    def record_service(self, cls: str, bytes_moved: int, stalled: bool) -> None:
        """Account one served use; starvation guard boosts idle classes."""
        if cls not in PRIORITY:
            raise ValueError(f"unknown class {cls!r}")
        self._served[cls] += 1
        self._since_service[cls] = 0
        for other in PRIORITY:
            if other != cls:
                self._since_service[other] += 1
        if stalled:
            self._stalls += 1
        self._bytes_per_use.append(float(bytes_moved))
        for other in PRIORITY:
            if other != cls and self._since_service[other] >= self._starve_ops:
                self.boosts += 1
                self._since_service[other] = 0

    def steal_for_kv(self, need_bytes: int, used_bytes: int) -> dict:
        """Joint-mode KV growth plan: steal only the shortfall past budget.

        Returns freed bytes per class; independent mode frees nothing.
        Pools enforce the new ceilings downstream through their own caps.
        """
        freed = {"engram": 0, "expert_slots": 0}
        if self._mode == "independent":
            return freed
        need = max(0, used_bytes + need_bytes - self._total)
        while need > 0 and self._engram_reservation > 0:
            step = min(need, 16 * 1024**2, self._engram_reservation)
            self._engram_reservation -= step
            need -= step
            freed["engram"] += step
        while need > 0 and self._expert_slots > 1:
            step = min(need, self._n_layers * self._expert_bytes)
            self._expert_slots -= max(1, step // (self._n_layers * self._expert_bytes))
            need -= step
            freed["expert_slots"] += step
        return freed

    @property
    def engram_reservation(self) -> int:
        return self._engram_reservation

    def metrics(self) -> dict:
        uses = len(self._bytes_per_use)
        return {
            "schema": SCHEMA,
            "mode": self._mode,
            "total_budget_bytes": self._total,
            "expert_slots_per_layer": self._expert_slots,
            "served": dict(self._served),
            "sheds": dict(self._sheds),
            "stalls": self._stalls,
            "boosts": self.boosts,
            "mean_bytes_per_use": sum(self._bytes_per_use) / uses if uses else 0.0,
        }
