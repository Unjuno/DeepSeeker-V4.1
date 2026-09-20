"""Issue #10: P3 shadow online learner (offline protocol).

Runs a predictor *alongside* inference replayed from an Issue #8 trace:
it logs what it would prefetch but never controls memory. Two horizons:

  horizon 0 - record-causal: prediction may use earlier layers of the
              same token (online-legal: layers execute in order).
  horizon 1 - token-ahead: prediction for token t uses only tokens < t.
              Intra-token predictors (layer_transition) legitimately
              score ~0 here: at token-ahead time no prefix is executing.

Switch-on decision (roadmap P3): sustained rolling recall@K with bounded
wasted bytes/token over a window. Until the Issue #6 checkpoint and
runtime deps land, traces are synthetic; the protocol is identical.
"""

from __future__ import annotations

from deepseeker.simulate import make_predictor


def _group_tokens(records: list[dict]) -> list[tuple[str, int, list[dict]]]:
    """Group records into (request_id, token_pos, [records]) in order."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        groups.setdefault((record["request_id"], record["token_pos"]), []).append(record)
    ordered = sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    return [(req, tok, sorted(rs, key=lambda r: r["layer"])) for (req, tok), rs in ordered]


def run_shadow(
    records: list[dict],
    predictor_name: str,
    budget: int,
    top_k: int,
    expert_bytes: int = 0,
    horizon: int = 0,
) -> dict:
    """Stream a trace through a predictor; return per-token log + totals."""
    if horizon not in (0, 1):
        raise ValueError(f"horizon must be 0 or 1, got {horizon!r}")
    predictor = make_predictor(predictor_name)
    tokens = _group_tokens(records)
    log = []
    total_recall = 0.0
    total_fp = 0
    n_layers_seen = 0
    for request_id, token_pos, layer_records in tokens:
        if horizon == 1:
            guesses = {
                r["layer"]: predictor.predict(request_id, token_pos, r["layer"], budget)
                for r in layer_records
            }
        rec_recall = 0.0
        rec_fp = 0
        for record in layer_records:
            actual = record["experts"]
            if horizon == 0:
                guessed = predictor.predict(
                    request_id, token_pos, record["layer"], budget
                )
            else:
                guessed = guesses[record["layer"]]
            rec_recall += len(set(actual) & set(guessed)) / top_k
            rec_fp += len(set(guessed) - set(actual))
            n_layers_seen += 1
            if horizon == 0:
                predictor.update(request_id, token_pos, record["layer"], actual)
        if horizon == 1:
            for record in layer_records:
                predictor.update(request_id, token_pos, record["layer"], record["experts"])
        n_layer = len(layer_records)
        entry = {
            "request_id": request_id,
            "token_pos": token_pos,
            "recall": rec_recall / n_layer if n_layer else 0.0,
            "fp": rec_fp,
            "wasted_bytes": rec_fp * expert_bytes,
        }
        log.append(entry)
        total_recall += rec_recall
        total_fp += rec_fp
    return {
        "predictor": predictor_name,
        "budget": budget,
        "horizon": horizon,
        "tokens": len(log),
        "mean_recall": total_recall / n_layers_seen if n_layers_seen else 0.0,
        "mean_fp_per_token": total_fp / len(log) if log else 0.0,
        "mean_waste_per_token": total_fp * expert_bytes / len(log) if log else 0.0,
        "log": log,
    }


def switch_on(
    log: list[dict],
    window: int,
    min_recall: float,
    max_waste_per_token: float,
) -> int | None:
    """First log index where the trailing window sustains the criteria.

    Returns None when the predictor never qualifies: it must stay a
    shadow and never take over memory control.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window!r}")
    for i in range(len(log)):
        if i + 1 < window:
            continue
        span = log[i + 1 - window : i + 1]
        if all(
            e["recall"] >= min_recall and e["wasted_bytes"] <= max_waste_per_token
            for e in span
        ):
            return i
    return None
