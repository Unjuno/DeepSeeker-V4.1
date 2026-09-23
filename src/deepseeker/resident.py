"""Issue #11: P4 resident expert-pool simulator (roadmap P4).

Bounded per-layer resident pool fed by an Issue #8 trace. Policies:

  demand_lru - page in on miss, evict least-recently-used. No predictor.
  prefetch   - admit the predictor's guess before serving, then behave
               like demand_lru (zero-latency staging: optimistic bound).
  belady     - offline optimal: evict the resident with farthest next use.

Metrics per (policy, budget): hit_rate, misses/record, miss bytes/record,
evictions, and churn (evicted-then-reused experts). Belady is the upper
bound every realizable policy is measured against.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from functools import partial

from deepseeker.simulate import make_predictor

POLICIES = ("demand_lru", "prefetch", "belady")


def _ordered(records: list[dict]) -> list[dict]:
    from deepseeker.trace import expert_records

    return sorted(expert_records(records),
                  key=lambda r: (r["request_id"], r["token_pos"], r["layer"]))


def _future_uses(records: list[dict]) -> dict[tuple[int, int], list[int]]:
    """(layer, expert) -> sorted record indices where it is used."""
    uses: dict[tuple[int, int], list[int]] = {}
    for i, record in enumerate(records):
        for expert in record["experts"]:
            uses.setdefault((record["layer"], expert), []).append(i)
    return uses


def _next_use(
    uses: dict[tuple[int, int], list[int]], layer: int, index: int, resident: int
) -> float:
    """Next record index after `index` using (layer, resident); inf if none."""
    lst = uses.get((layer, resident), [])
    pos = bisect_right(lst, index)
    return float(lst[pos]) if pos < len(lst) else float("inf")


class _Pool:
    """Per-layer bounded resident sets with LRU order."""

    def __init__(self, budget: int | dict[int, int], default: int = 1) -> None:
        if isinstance(budget, int):
            if budget < 1:
                raise ValueError(f"budget must be >= 1, got {budget!r}")
            self._caps = {}
            self._default = budget
        else:
            if default < 1 or any(b < 1 for b in budget.values()):
                raise ValueError(f"budgets must be >= 1: {budget!r} default={default!r}")
            self._caps = dict(budget)
            self._default = default
        self._resident: dict[int, OrderedDict[int, None]] = {}

    def cap(self, layer: int) -> int:
        return self._caps.get(layer, self._default)

    def contains(self, layer: int, expert: int) -> bool:
        return expert in self._resident.get(layer, ())

    def touch(self, layer: int, expert: int) -> None:
        bucket = self._resident.setdefault(layer, OrderedDict())
        if expert in bucket:
            bucket.move_to_end(expert)

    def admit(self, layer: int, expert: int) -> int | None:
        """Admit expert; return evicted expert (or None)."""
        bucket = self._resident.setdefault(layer, OrderedDict())
        if expert in bucket:
            bucket.move_to_end(expert)
            return None
        evicted = None
        if len(bucket) >= self.cap(layer):
            evicted, _ = bucket.popitem(last=False)
        bucket[expert] = None
        return evicted

    def evict_specific(self, layer: int, expert: int) -> None:
        bucket = self._resident.get(layer, OrderedDict())
        bucket.pop(expert, None)

    def residents(self, layer: int) -> list[int]:
        return list(self._resident.get(layer, OrderedDict()))


def simulate(
    records: list[dict],
    policy: str,
    budget: int,
    top_k: int,
    predictor_name: str = "persistence",
    expert_bytes: int = 0,
    budgets_per_layer: dict[int, int] | None = None,
) -> dict:
    """Run one policy over a trace in causal order.

    `budgets_per_layer` overrides the uniform `budget` for listed layers
    (Issue #13 global allocation); unlisted layers keep `budget`.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
    ordered = _ordered(records)
    uses = _future_uses(ordered)
    pool = _Pool(budgets_per_layer if budgets_per_layer else budget, default=budget)
    predictor = make_predictor(predictor_name) if policy == "prefetch" else None
    hits = 0
    misses = 0
    evictions = 0
    evicted_at: dict[tuple[int, int], int] = {}
    total = 0

    def note_eviction(layer: int, expert: int, index: int) -> None:
        evicted_at.setdefault((layer, expert), index)

    for i, record in enumerate(ordered):
        layer = record["layer"]
        actual = record["experts"]
        if policy == "prefetch":
            assert predictor is not None
            for expert in predictor.predict(
                record["request_id"], record["token_pos"], layer, pool.cap(layer)
            ):
                evicted = pool.admit(layer, expert)
                if evicted is not None:
                    evictions += 1
                    note_eviction(layer, evicted, i)
        for expert in actual:
            total += 1
            if pool.contains(layer, expert):
                hits += 1
                pool.touch(layer, expert)
            else:
                misses += 1
                if policy == "belady":
                    bucket = pool.residents(layer)
                    if len(bucket) >= pool.cap(layer) and expert not in bucket:
                        farthest = max(bucket, key=partial(_next_use, uses, layer, i))
                        pool.evict_specific(layer, farthest)
                        evictions += 1
                        note_eviction(layer, farthest, i)
                evicted = pool.admit(layer, expert)
                if evicted is not None:
                    evictions += 1
                    note_eviction(layer, evicted, i)
        if policy == "prefetch":
            assert predictor is not None
            predictor.update(record["request_id"], record["token_pos"], layer, actual)
    churn = sum(
        1
        for (layer, expert), at in evicted_at.items()
        if any(idx > at for idx in uses.get((layer, expert), []))
    )
    return {
        "policy": policy,
        "predictor": predictor_name if policy == "prefetch" else None,
        "budget": budget,
        "budgets_per_layer": budgets_per_layer,
        "records": len(ordered),
        "uses": total,
        "hit_rate": hits / total if total else 0.0,
        "miss_per_record": misses / len(ordered) if ordered else 0.0,
        "miss_bytes_per_record": misses * expert_bytes / len(ordered) if ordered else 0.0,
        "evictions": evictions,
        "churn_experts": churn,
    }


def compare(
    records: list[dict],
    budgets: list[int],
    top_k: int,
    predictor_name: str = "persistence",
    expert_bytes: int = 0,
    policies: tuple[str, ...] = POLICIES,
) -> list[dict]:
    """Run every policy at every budget."""
    return [
        simulate(records, policy, budget, top_k, predictor_name, expert_bytes)
        for policy in policies
        for budget in budgets
    ]
