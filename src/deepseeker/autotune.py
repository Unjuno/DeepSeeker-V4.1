"""Issue #12: P8 startup autotuning recommender (roadmap P8).

Reads committed target-machine profiles plus Issue #9/#10/#11 sim outputs
and emits one deterministic recommended config with per-field rationale.
Every rule declares its fallback for missing inputs; nothing here runs
inference or needs weights.

Rules:
  resident_slots   operating_bytes minus dense/engram reservations, split
                   over 40 layers in expert units, clamped [top_k, 384].
  predictor/budget smallest budget hitting target recall; among predictors
                   within 5% of the best recall at that budget, least waste.
  horizon          token-ahead (1) iff its best recall is within 10% of
                   horizon-0 best; else 0.
  prefetch         enabled iff the shadow JSON shows a switch-on verdict
                   for the chosen predictor/budget.
  cpu_workers      performance-core count (staging/GEMM overlap).
  io_concurrency   measured download sweet spot (6) — network-bound then,
                   kept as start value for staging until P5 measures it.
"""

from __future__ import annotations

SCHEMA = "deepseeker.autotune/v1"
N_LAYERS = 40
N_ROUTED_EXPERTS = 384
OS_RESERVE_BYTES = 4 * 1024**3


def _get(d: dict, *path, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def recommend_slots(
    operating_bytes: int | None,
    dense_engram_bytes: int | None,
    expert_bytes: int | None,
    top_k: int,
) -> tuple[int, str]:
    """Per-layer resident slots from the Issue #5 operating budget."""
    if not operating_bytes or not expert_bytes:
        return top_k, "no profile bytes; fallback to top_k (no prefetch headroom)"
    pool = operating_bytes - (dense_engram_bytes or 0) - OS_RESERVE_BYTES
    slots = max(top_k, min(N_ROUTED_EXPERTS, int(pool // N_LAYERS // expert_bytes)))
    return slots, (
        f"({operating_bytes} - {dense_engram_bytes or 0} dense - "
        f"{OS_RESERVE_BYTES} os) / {N_LAYERS} layers / {expert_bytes} B = {slots}"
    )


def recommend_predictor_budget(
    sim_rows: list[dict],
    target_recall: float = 0.7,
    waste_cap_bytes: float = float("inf"),
) -> tuple[str, int, str]:
    """Smallest budget hitting target recall; cheapest waste near the top."""
    if not sim_rows:
        return "persistence", 6, "no sim data; persistence@6 is the sticky prior"
    budgets = sorted({r["budget"] for r in sim_rows})
    for budget in budgets:
        cand = [r for r in sim_rows if r["budget"] == budget]
        hitting = [r for r in cand if r["recall_at_k"] >= target_recall]
        if not hitting:
            continue
        best = max(r["recall_at_k"] for r in hitting)
        near = [r for r in hitting if r["recall_at_k"] >= best - 0.05]
        affordable = [r for r in near if r["wasted_bytes"] <= waste_cap_bytes] or near
        winner = min(affordable, key=lambda r: (r["wasted_bytes"], r["predictor"]))
        return (
            winner["predictor"],
            budget,
            (
                f"budget {budget} first to reach recall {target_recall}; "
                f"{winner['predictor']} least waste within 5% of best ({best:.3f})"
            ),
        )
    best_row = max(sim_rows, key=lambda r: (r["recall_at_k"], -r["wasted_bytes"]))
    return (
        best_row["predictor"],
        best_row["budget"],
        (
            f"target recall {target_recall} unreachable; best available "
            f"({best_row['recall_at_k']:.3f})"
        ),
    )


def recommend_horizon(shadow0: list[dict], shadow1: list[dict]) -> tuple[int, str]:
    """Token-ahead iff within 10% of horizon-0 best recall."""
    def best(rows: list[dict]) -> float:
        return max((r["mean_recall"] for r in rows), default=0.0)

    if not shadow0 or not shadow1:
        return 0, "shadow data missing for a horizon; stay at horizon 0"
    h0, h1 = best(shadow0), best(shadow1)
    if h0 <= 0:
        return 0, "horizon-0 recall is zero; no basis for lookahead"
    if h1 >= 0.9 * h0:
        return 1, f"token-ahead recall {h1:.3f} within 10% of {h0:.3f}"
    return 0, f"token-ahead recall {h1:.3f} too far below {h0:.3f}"


def recommend_workers(cpu_performance_cores: int | None) -> tuple[int, str]:
    """One staging worker per performance core."""
    if not cpu_performance_cores:
        return 4, "no machine profile; conservative fallback"
    return cpu_performance_cores, "one worker per performance core"


def recommend(
    machine: dict,
    memory_plan: dict,
    manifest: dict,
    sim_rows: list[dict] | None = None,
    shadow0: list[dict] | None = None,
    shadow1: list[dict] | None = None,
    target_recall: float = 0.7,
) -> dict:
    """Build the full recommended config with rationale."""
    expert_bytes = int(_get(manifest, "expert_stats", "median_bytes", default=0) or 0)
    top_k = int(_get(manifest, "architecture", "num_experts_per_tok", default=6) or 6)
    slots, slots_why = recommend_slots(
        _get(memory_plan, "operating_bytes"),
        _get(memory_plan, "dense_engram_bytes", "bytes"),
        expert_bytes or None,
        top_k,
    )
    predictor, budget, pred_why = recommend_predictor_budget(sim_rows or [], target_recall)
    horizon, horizon_why = recommend_horizon(shadow0 or [], shadow1 or [])
    workers, workers_why = recommend_workers(machine.get("cpu_performance_cores"))
    switched = any(
        r.get("predictor") == predictor and r.get("switch_at") is not None
        for r in (shadow1 if horizon == 1 else shadow0) or []
    )
    prefetch = switched and horizon == 1
    config = {
        "resident_slots_per_layer": min(slots, budget),
        "predictor": predictor,
        "budget": budget,
        "horizon": horizon,
        "prefetch_enabled": prefetch,
        "cpu_workers": workers,
        "io_concurrency": 6,
    }
    rationale = {
        "resident_slots_per_layer": f"{slots_why}; capped by sim budget {budget}",
        "predictor": pred_why,
        "budget": pred_why,
        "horizon": horizon_why,
        "prefetch_enabled": (
            "shadow switched on for chosen predictor"
            if prefetch
            else "no shadow switch-on verdict; stay shadow-only"
        ),
        "cpu_workers": workers_why,
        "io_concurrency": "measured Issue #6 download sweet spot; revisit in P5",
    }
    return {
        "schema": SCHEMA,
        "target_recall": target_recall,
        "config": config,
        "rationale": rationale,
    }
