"""Issue #8: P1 ground-truth routing-trace schema and offline tooling.

Records authoritative Top-k routing decisions per (request, token, layer)
as JSONL. Online collection hooks into the reference runtime once the
Issue #6 checkpoint and runtime deps land; everything here runs without
weights so the P2 simulator can be developed against synthetic traces.

Record (v1):
  request_id   str    prompt/request identity
  token_pos    int    >= 0, position in the request
  layer        int    [0, n_layers)
  experts      [int]  authoritative Top-k expert IDs, length == top_k
  scores       [float]|null  router scores aligned with experts (optional)
  timing_ms    float|null    layer routing time (optional, online only)
  bytes_moved  int|null      weight bytes touched (optional, online only)
  extra        object extra subsystem events (kv/engram; optional)

File header (first line, kind == "header"):
  kind/schema  "header" / "deepseeker.trace/v1"
  repo_id, revision, source ("synthetic" | "reference"),
  n_layers, n_routed_experts, top_k, created_at_utc
"""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path

SCHEMA = "deepseeker.trace/v1"
HEADER_KIND = "header"


def arch_config(manifest: dict) -> dict:
    """Routing geometry from the Issue #2 manifest architecture block."""
    arch = manifest.get("architecture", {})
    return {
        "n_layers": arch["num_hidden_layers"],
        "n_routed_experts": arch["n_routed_experts"],
        "top_k": arch["num_experts_per_tok"],
    }


def make_header(
    repo_id: str,
    revision: str,
    source: str,
    cfg: dict,
    seed: int | None = None,
) -> dict:
    """File header dict; validates routing geometry eagerly."""
    for key in ("n_layers", "n_routed_experts", "top_k"):
        if not isinstance(cfg.get(key), int) or cfg[key] <= 0:
            raise ValueError(f"bad routing geometry: {key}={cfg.get(key)!r}")
    header = {
        "kind": HEADER_KIND,
        "schema": SCHEMA,
        "repo_id": repo_id,
        "revision": revision,
        "source": source,
        "n_layers": cfg["n_layers"],
        "n_routed_experts": cfg["n_routed_experts"],
        "top_k": cfg["top_k"],
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if seed is not None:
        header["seed"] = seed
    return header


def validate_record(record: dict, header: dict) -> list[str]:
    """Return problem strings (empty == valid) for one trace record."""
    problems = []
    if record.get("kind", "record") != "record":
        return [f"bad kind: {record.get('kind')!r}"]
    top_k = header["top_k"]
    n_exp = header["n_routed_experts"]
    if not isinstance(record.get("request_id"), str) or not record["request_id"]:
        problems.append("request_id must be a nonempty string")
    if not isinstance(record.get("token_pos"), int) or record["token_pos"] < 0:
        problems.append(f"token_pos out of range: {record.get('token_pos')!r}")
    layer = record.get("layer")
    if not isinstance(layer, int) or not 0 <= layer < header["n_layers"]:
        problems.append(f"layer out of range: {layer!r}")
    experts = record.get("experts")
    if (
        not isinstance(experts, list)
        or len(experts) != top_k
        or any(not isinstance(e, int) or not 0 <= e < n_exp for e in experts)
    ):
        problems.append(f"experts must be {top_k} IDs in [0, {n_exp}): {experts!r}")
    elif len(set(experts)) != len(experts):
        problems.append(f"duplicate expert IDs: {experts!r}")
    scores = record.get("scores")
    if scores is not None and (
        not isinstance(scores, list)
        or len(scores) != top_k
        or any(not isinstance(s, (int, float)) for s in scores)
    ):
        problems.append(f"scores must align with experts: {scores!r}")
    for key in ("timing_ms", "bytes_moved"):
        value = record.get(key)
        if value is not None and not isinstance(value, (int, float)):
            problems.append(f"{key} must be numeric: {value!r}")
    extra = record.get("extra")
    if extra is not None and not isinstance(extra, dict):
        problems.append(f"extra must be an object: {extra!r}")
    return problems


def write_trace(path: Path, header: dict, records: list[dict]) -> int:
    """Write header + records as JSONL; returns record count."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        f.writelines(json.dumps({"kind": "record", **r}) + "\n" for r in records)
    return len(records)


def read_trace(path: Path) -> tuple[dict, list[dict]]:
    """Read JSONL trace; raises ValueError on malformed header/records."""
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        raise ValueError("empty trace file")
    header = json.loads(lines[0])
    if header.get("kind") != HEADER_KIND or header.get("schema") != SCHEMA:
        raise ValueError(f"bad trace header: {lines[0][:120]!r}")
    records = []
    for i, line in enumerate(lines[1:], 2):
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"line {i}: invalid JSON: {exc}") from exc
        problems = validate_record(record, header)
        if problems:
            raise ValueError(f"line {i}: {problems[0]}")
        record.pop("kind", None)
        records.append(record)
    return header, records


def layer_histogram(records: list[dict], n_layers: int) -> dict[int, dict[int, int]]:
    """Per-layer {expert_id: count} over trace records."""
    hist: dict[int, dict[int, int]] = {layer: {} for layer in range(n_layers)}
    for record in records:
        bucket = hist[record["layer"]]
        for expert in record["experts"]:
            bucket[expert] = bucket.get(expert, 0) + 1
    return hist


def trace_stats(header: dict, records: list[dict]) -> dict:
    """Summary stats: coverage, unique experts, adjacent-token stability."""
    n_layers = header["n_layers"]
    hist = layer_histogram(records, n_layers)
    unique = {layer: len(bucket) for layer, bucket in hist.items()}
    by_request: dict[tuple[str, int], list[int]] = {}
    for record in records:
        by_request.setdefault((record["request_id"], record["layer"]), []).append(
            record["token_pos"]
        )
    stable_pairs = 0
    total_pairs = 0
    keyed = {(r["request_id"], r["layer"], r["token_pos"]): r["experts"] for r in records}
    for (request, layer), positions in by_request.items():
        for a, b in zip(sorted(positions), sorted(positions)[1:]):
            if b == a + 1:
                total_pairs += 1
                if set(keyed[(request, layer, a)]) == set(keyed[(request, layer, b)]):
                    stable_pairs += 1
    return {
        "records": len(records),
        "requests": len({r["request_id"] for r in records}),
        "unique_experts_per_layer": unique,
        "mean_unique": (
            sum(unique.values()) / n_layers if n_layers else 0.0
        ),
        "adjacent_same_set_ratio": (stable_pairs / total_pairs if total_pairs else None),
    }


def emit_synthetic(
    header: dict,
    n_requests: int,
    tokens_per_request: int,
    seed: int = 0,
    stickiness: float = 0.7,
    hot_fraction: float = 0.15,
) -> list[dict]:
    """Deterministic synthetic routing trace with realistic structure.

    Each layer has a hot expert subset (hot_fraction of the pool) that
    receives most picks; consecutive tokens keep the same expert set with
    probability `stickiness`. Gives the P2 simulator non-uniform,
    temporally correlated input without weights.
    """
    rng = random.Random(seed)
    n_layers = header["n_layers"]
    n_exp = header["n_routed_experts"]
    top_k = header["top_k"]
    hot = {
        layer: rng.sample(range(n_exp), max(top_k, int(n_exp * hot_fraction)))
        for layer in range(n_layers)
    }
    records = []
    for req in range(n_requests):
        request_id = f"synthetic-{seed}-{req}"
        prev: dict[int, list[int]] = {}
        for token in range(tokens_per_request):
            for layer in range(n_layers):
                if token and rng.random() < stickiness and layer in prev:
                    experts = prev[layer]
                else:
                    pool = hot[layer]
                    experts = rng.sample(pool, top_k)
                    if rng.random() < 0.1:
                        experts = rng.sample(range(n_exp), top_k)
                    experts = sorted(experts)
                prev[layer] = experts
                records.append(
                    {
                        "request_id": request_id,
                        "token_pos": token,
                        "layer": layer,
                        "experts": experts,
                    }
                )
    return records
