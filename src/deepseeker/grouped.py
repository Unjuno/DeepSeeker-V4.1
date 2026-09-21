"""Issue #31: expert-wise token grouping + grouped GEMM prototype (MLX).

Planner: token->expert top-k assignments become per-expert row groups
via the Issue #18 align contract (stable order within expert).
Execution: one batched SwiGLU per non-empty expert, router-weighted
scatter-add into token outputs. Naive per-(token, expert) GEMV loop is
the correctness oracle; outputs must match bit-nearly (fp32 accumulate
in the test harness; bf16 compute in both paths).

Live speculative-block distributions arrive with #30; the planner takes
any assignment matrix today.
"""

from __future__ import annotations

import mlx.core as mx


def plan_groups(topk_ids: list[list[int]], n_experts: int) -> dict[int, list[tuple[int, int]]]:
    """Map expert -> [(row, topk position)] (stable: ascending row order).

    Duplicate (row, expert) pairs are kept: the executor evaluates each
    occurrence, matching the naive loop exactly.
    """
    groups: dict[int, list[tuple[int, int]]] = {}
    for row, experts in enumerate(topk_ids):
        for pos, expert in enumerate(experts):
            if not 0 <= expert < n_experts:
                raise ValueError(f"expert {expert} out of range [0, {n_experts})")
            groups.setdefault(expert, []).append((row, pos))
    return groups


def grouped_swiglu(
    x: mx.array,
    weights: dict[int, tuple[mx.array, mx.array, mx.array]],
    groups: dict[int, list[tuple[int, int]]],
    router_weights: list[list[float]] | None = None,
) -> mx.array:
    """Batched SwiGLU per expert + weighted scatter-add. Returns [T, H]."""
    tokens = x.shape[0]
    hidden = x.shape[1]
    out = mx.zeros((tokens, hidden), dtype=x.dtype)
    for expert, occurrences in groups.items():
        gate, up, down = weights[expert]
        rows = [row for row, _ in occurrences]
        xr = x[mx.array(rows)]
        g = xr @ gate
        y = (g * mx.sigmoid(g) * (xr @ up)) @ down
        if router_weights is not None:
            w = mx.array([router_weights[row][pos] for row, pos in occurrences],
                         dtype=x.dtype).reshape(-1, 1)
            y = y * w
        out = out.at[mx.array(rows)].add(y)
    mx.eval(out)
    return out


def naive_swiglu(
    x: mx.array,
    weights: dict[int, tuple[mx.array, mx.array, mx.array]],
    topk_ids: list[list[int]],
    router_weights: list[list[float]] | None = None,
) -> mx.array:
    """Per-(token, expert) oracle path. Must match grouped output."""
    tokens = x.shape[0]
    hidden = x.shape[1]
    out = mx.zeros((tokens, hidden), dtype=x.dtype)
    for row, experts in enumerate(topk_ids):
        acc = None
        for i, expert in enumerate(experts):
            gate, up, down = weights[expert]
            xr = x[row : row + 1]
            g = xr @ gate
            y = (g * mx.sigmoid(g) * (xr @ up)) @ down
            if router_weights is not None:
                y = y * router_weights[row][i]
            acc = y if acc is None else acc + y
        out = out.at[row].add(acc.reshape(-1) if acc is not None else mx.zeros((hidden,)))
    mx.eval(out)
    return out
