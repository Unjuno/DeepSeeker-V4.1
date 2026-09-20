"""Issue #17: golden correctness/equivalence + benchmark harness.

Backend-agnostic: a backend maps (prompt, sampling) -> {tokens, logits?,
seconds}. The harness records machine-readable results (schema
deepseeker.bench/v1) with TTFT/prefill/decode split, repeats as
median/range, and equivalence verdicts:

  PASS      token-exact match, or logits within tolerance
  UNCERTAIN backend numerical differences suspected (logit drift with
            token agreement, or missing logits)
  FAIL      token mismatch beyond tolerance

An intentional perturbation (flipped token / noisy logits) must FAIL:
that is the minimum test proving the oracle bites.
"""

from __future__ import annotations

import datetime
import statistics

SCHEMA = "deepseeker.bench/v1"


def summarize(samples: list[float]) -> dict:
    """Median/range summary; empty input is an error, not zero."""
    if not samples:
        raise ValueError("no samples to summarize")
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "range": ordered[-1] - ordered[0],
    }


def compare_tokens(golden: list[int], candidate: list[int], max_mismatch: int = 0) -> dict:
    """Token-exact comparison with an allowed mismatch budget."""
    mismatches = sum(1 for a, b in zip(golden, candidate) if a != b)
    mismatches += abs(len(golden) - len(candidate))
    ok = mismatches <= max_mismatch
    return {
        "verdict": "PASS" if ok else "FAIL",
        "mismatches": mismatches,
        "allowed": max_mismatch,
        "golden_len": len(golden),
        "candidate_len": len(candidate),
    }


def compare_logits(
    golden: list[list[float]],
    candidate: list[list[float]],
    atol: float = 1e-3,
    drift_warn: float = 1e-2,
) -> dict:
    """Logit comparison with tolerance + UNCERTAIN drift band."""
    if len(golden) != len(candidate):
        return {"verdict": "FAIL", "reason": "length mismatch", "max_abs_diff": None}
    worst = 0.0
    for grow, crow in zip(golden, candidate):
        if len(grow) != len(crow):
            return {"verdict": "FAIL", "reason": "width mismatch", "max_abs_diff": None}
        worst = max(worst, max(abs(a - b) for a, b in zip(grow, crow)))
    if worst <= atol:
        verdict = "PASS"
    elif worst <= drift_warn:
        verdict = "UNCERTAIN"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "max_abs_diff": worst, "atol": atol}


def equivalence(
    golden_tokens: list[int],
    candidate_tokens: list[int],
    golden_logits: list[list[float]] | None = None,
    candidate_logits: list[list[float]] | None = None,
    max_mismatch: int = 0,
) -> dict:
    """Combined verdict: tokens decide, logits refine or hedge."""
    tok = compare_tokens(golden_tokens, candidate_tokens, max_mismatch)
    if golden_logits is None or candidate_logits is None:
        verdict = tok["verdict"] if tok["verdict"] == "FAIL" else "UNCERTAIN"
        return {"verdict": verdict, "tokens": tok, "logits": None,
                "reason": "no logits: token-only verdict capped at UNCERTAIN"}
    log = compare_logits(golden_logits, candidate_logits)
    if tok["verdict"] == "FAIL" or log["verdict"] == "FAIL":
        verdict = "FAIL"
    elif tok["verdict"] == "PASS" and log["verdict"] == "PASS":
        verdict = "PASS"
    else:
        verdict = "UNCERTAIN"
    return {"verdict": verdict, "tokens": tok, "logits": log, "reason": ""}


def build_result(
    machine_id: str,
    revision: str,
    backend: str,
    corpus_id: str,
    settings: dict,
    ttft_s: list[float],
    decode_tok_s: list[float],
    accepted_tokens: int,
    equivalence_report: dict,
    io_counters: dict | None = None,
) -> dict:
    """Assemble a schema-validated benchmark result."""
    if accepted_tokens < 0:
        raise ValueError("accepted_tokens must be >= 0")
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "machine_id": machine_id,
        "revision": revision,
        "backend": backend,
        "corpus_id": corpus_id,
        "settings": settings,
        "ttft_s": summarize(ttft_s),
        "decode_tok_s": summarize(decode_tok_s),
        "accepted_tokens": accepted_tokens,
        "equivalence": equivalence_report,
        "io_counters": io_counters or {},
    }
