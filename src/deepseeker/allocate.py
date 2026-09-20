"""Issue #13: global cross-layer expert-slot allocation (roadmap P5 input).

Splits a fixed total slot budget across layers, then scores the split
with the Issue #11 pool simulator. Strategies:

  equal        - total // n_layers everywhere (baseline).
  proportional - shares follow per-layer unique-expert demand, floored at
                 top_k per layer, largest-remainder rounding.
  greedy       - water-fill on measured demand_lru hit-rate curves: each
                 extra slot goes to the layer with the largest marginal
                 hit-rate gain.

The equal-to-best gap is the prize cross-layer sharing plays for.
"""

from __future__ import annotations

from deepseeker.resident import simulate
from deepseeker.trace import layer_histogram


def unique_demand(records: list[dict], n_layers: int) -> dict[int, int]:
    """Unique experts used per layer."""
    return {layer: len(bucket) for layer, bucket in layer_histogram(records, n_layers).items()}


def allocate_equal(total: int, n_layers: int, top_k: int) -> dict[int, int]:
    """Even split; raises when the total cannot cover top_k per layer."""
    if total < top_k * n_layers:
        raise ValueError(f"total {total} < top_k*layers {top_k * n_layers}")
    base, rest = divmod(total, n_layers)
    return {layer: base + (1 if layer < rest else 0) for layer in range(n_layers)}


def allocate_proportional(
    total: int, demand: dict[int, int], top_k: int
) -> dict[int, int]:
    """Shares follow unique-expert demand with top_k floor."""
    layers = sorted(demand)
    n = len(layers)
    if total < top_k * n:
        raise ValueError(f"total {total} < top_k*layers {top_k * n}")
    floors = {layer: top_k for layer in layers}
    remaining = total - top_k * n
    weights = {layer: max(demand[layer] - top_k, 0) for layer in layers}
    weight_total = sum(weights.values())
    if weight_total == 0 or remaining == 0:
        extra = {layer: 0 for layer in layers}
    else:
        extra = {layer: remaining * weights[layer] // weight_total for layer in layers}
        leftover = remaining - sum(extra.values())
        remainders = sorted(
            layers, key=lambda l: (remaining * weights[l]) % weight_total, reverse=True
        )
        for layer in remainders[:leftover]:
            extra[layer] += 1
    return {layer: floors[layer] + extra[layer] for layer in layers}


def hit_curve(
    records: list[dict],
    layer: int,
    budgets: list[int],
    top_k: int,
    policy: str = "demand_lru",
) -> dict[int, float]:
    """Measured hit-rate per budget for one layer in isolation."""
    subset = [r for r in records if r["layer"] == layer]
    return {
        budget: simulate(subset, policy, budget, top_k)["hit_rate"] for budget in budgets
    }


def _lerp_curve(curve: dict[int, float], b: int) -> float:
    """Hit-rate at budget b by linear interpolation over measured grid."""
    from itertools import pairwise

    levels = sorted(curve)
    if b <= levels[0]:
        return curve[levels[0]]
    if b >= levels[-1]:
        return curve[levels[-1]]
    for lo, hi in pairwise(levels):
        if lo <= b < hi:
            frac = (b - lo) / (hi - lo)
            return curve[lo] + frac * (curve[hi] - curve[lo])
    return curve[levels[-1]]  # unreachable


def allocate_greedy(
    curves: dict[int, dict[int, float]],
    total: int,
    top_k: int,
    weights: dict[int, float] | None = None,
) -> dict[int, int]:
    """Hill-climb single-slot moves on interpolated total hits.

    Starts from the proportional-feasible point (here: top_k floors, rest
    even) and moves one slot donor -> recipient while total interpolated
    hits improve. Handles non-concave curves where water-filling stalls;
    ties break toward lower layer indices (deterministic).
    """
    layers = sorted(curves)
    n = len(layers)
    if total < top_k * n:
        raise ValueError(f"total {total} < top_k*layers {top_k * n}")
    if weights is None:
        weights = dict.fromkeys(layers, 1.0)
    alloc = dict.fromkeys(layers, top_k)
    remaining = total - top_k * n
    base, rest = divmod(remaining, n)
    for i, layer in enumerate(layers):
        alloc[layer] += base + (1 if i < rest else 0)

    def total_hits(plan: dict[int, int]) -> float:
        return sum(weights[layer] * _lerp_curve(curves[layer], plan[layer]) for layer in layers)

    improved = True
    while improved:
        improved = False
        current = total_hits(alloc)
        best_move: tuple[int, int] | None = None
        best_value = current
        for donor in layers:
            if alloc[donor] <= top_k:
                continue
            for recipient in layers:
                if recipient == donor:
                    continue
                alloc[donor] -= 1
                alloc[recipient] += 1
                value = total_hits(alloc)
                if value > best_value + 1e-12:
                    best_value = value
                    best_move = (donor, recipient)
                alloc[donor] += 1
                alloc[recipient] -= 1
        if best_move is not None:
            donor, recipient = best_move
            alloc[donor] -= 1
            alloc[recipient] += 1
            improved = True
    return alloc


def evaluate_allocation(
    records: list[dict],
    allocation: dict[int, int],
    top_k: int,
    policy: str = "demand_lru",
    expert_bytes: int = 0,
) -> dict:
    """Score a per-layer split with the pool simulator."""
    if not allocation:
        raise ValueError("allocation is empty")
    return simulate(
        records,
        policy,
        top_k,
        top_k,
        expert_bytes=expert_bytes,
        budgets_per_layer=allocation,
    )
