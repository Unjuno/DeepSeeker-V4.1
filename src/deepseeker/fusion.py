"""Issue #34: fusion/layout/overlap accounting + verdicts (roadmap P7).

Analytic baselines (no experiments): per-token intermediate bytes and
launch counts for the expert phase at true geometry, then measured
candidates each get PASS (improves target workload, memory/quality
held) or REVERT. Losing fusions are reverted individually by rule.
"""

from __future__ import annotations

SCHEMA = "deepseeker.fusion/v1"

# Element counts per token for one expert (SwiGLU: gate+up+down GEMMs,
# sigmoid+mul elementwise, weighted add). bf16 = 2B.
GEOMETRY = {"hidden": 5120, "inter": 2304, "top_k": 6, "n_layers": 40}


def intermediate_bytes(tokens: int = 1, dtype_bytes: int = 2) -> dict:
    """Unfused intermediate traffic per token (one expert): gate, up,
    activated, down-input. Fused SwiGLU keeps gate/up/activated on-chip
    and only writes the down output."""
    h = GEOMETRY["hidden"]
    i = GEOMETRY["inter"]
    gate = tokens * i * dtype_bytes
    up = tokens * i * dtype_bytes
    act = tokens * i * dtype_bytes
    down_out = tokens * h * dtype_bytes
    unfused = gate + up + act + down_out
    fused = down_out
    return {
        "gate_bytes": gate,
        "up_bytes": up,
        "act_bytes": act,
        "down_out_bytes": down_out,
        "unfused_bytes": unfused,
        "fused_bytes": fused,
        "saved_bytes": unfused - fused,
        "saved_ratio": (unfused - fused) / unfused if unfused else 0.0,
    }


def launch_counts(tokens: int = 1) -> dict:
    """Eval-dispatch counts per token: naive (per GEMM eval) vs fused
    (one eval per expert) vs ideal (one eval per layer)."""
    per_expert_evals = 4  # gate, up, elem, down
    experts = GEOMETRY["top_k"] * GEOMETRY["n_layers"]
    return {
        "naive_evals": experts * per_expert_evals,
        "fused_evals": experts,
        "ideal_evals": GEOMETRY["n_layers"],
    }


def verdict(name: str, faster_ms: float, baseline_ms: float,
            memory_ok: bool, quality_ok: bool) -> dict:
    """PASS only on measured win with memory + quality held."""
    wins = faster_ms < baseline_ms
    passed = bool(wins and memory_ok and quality_ok)
    return {
        "fusion": name,
        "verdict": "PASS" if passed else "REVERT",
        "baseline_ms": baseline_ms,
        "candidate_ms": faster_ms,
        "speedup": baseline_ms / faster_ms if faster_ms else 0.0,
        "memory_ok": memory_ok,
        "quality_ok": quality_ok,
    }
