"""Issue #35: CPU/GPU work-partition policy (roadmap P7).

Rule engine over measured task latencies. Placement principle, proved
by measurement below: control stays on CPU (sub-ms decisions, no
transfer), data plane on GPU (bandwidth-bound GEMMs); ANE unused (no
toolchain on this machine, dynamic routing shapes unsuitable — the
issue explicitly allows a measured no-ANE PASS).

Policy output: {task: (placement, fallback)} with per-task rationale.
Fallback is always the other device; quality invariant holds because
placement never changes math, only where it runs.
"""

from __future__ import annotations

SCHEMA = "deepseeker.partition/v1"

TASKS = (
    "router-matvec",
    "topk-select",
    "engram-hash",
    "expert-gemm",
    "cache-decisions",
    "tokenize",
)


def decide(measurements: dict[str, dict[str, float]]) -> dict:
    """Pick per-task winner from {task: {device: ms}} measurements.

    Winner = fastest device; ties (<5% gap) prefer CPU (frees GPU for
    the data plane and avoids sync points). ANE never wins: unevaluated.
    """
    policy = {}
    for task in TASKS:
        times = measurements.get(task, {})
        if not times:
            policy[task] = {"placement": "cpu", "fallback": "gpu",
                            "reason": "no measurement; CPU default (control-plane prior)"}
            continue
        ranked = sorted(times.items(), key=lambda kv: kv[1])
        winner, best = ranked[0]
        if len(ranked) > 1:
            second = ranked[1][1]
            if (second - best) / best < 0.05:
                winner, best = ("cpu", times["cpu"]) if "cpu" in times else (winner, best)
                reason = "within 5%: CPU preferred (GPU left for data plane)"
            else:
                reason = f"measured {best:.3f}ms vs {second:.3f}ms"
        else:
            reason = "only candidate measured"
        fallback = "cpu" if winner == "gpu" else "gpu"
        policy[task] = {"placement": winner, "fallback": fallback,
                        "measured_ms": round(best, 3), "reason": reason}
    policy["ane"] = {"placement": "unused", "fallback": "n/a",
                     "reason": "no CoreML toolchain; dynamic routing shapes unsuitable"}
    return {"schema": SCHEMA, "policy": policy}
