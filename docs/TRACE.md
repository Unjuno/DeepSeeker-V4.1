# Routing traces (Issue #8, roadmap P1)

Ground-truth routing traces record the authoritative Top-k expert choice
per (request, token, layer). No predictive optimization reads these yet;
they are the measurement substrate P2 simulates against.

## Format (JSONL, schema `deepseeker.trace/v1`)

First line is a header (`kind: header`) with `repo_id`, `revision`,
`source` (`synthetic` | `reference`), routing geometry (`n_layers`,
`n_routed_experts`, `top_k`) and timestamp. Every following line is a
record: `request_id`, `token_pos`, `layer`, `experts[top_k]`, plus
optional `scores`, `timing_ms`, `bytes_moved`, `extra` (kv/engram events).

Validation rules: layer in range, exactly `top_k` distinct expert IDs in
range, scores aligned when present. `read_trace` rejects corrupt files
with the offending line number.

## Tooling

    python scripts/collect_trace.py emit-synthetic --requests 4 --tokens 32 --out /tmp/t.jsonl
    python scripts/collect_trace.py validate --trace /tmp/t.jsonl
    python scripts/collect_trace.py stats --trace /tmp/t.jsonl
    python scripts/collect_trace.py readiness

`stats` reports per-layer unique-expert counts, mean coverage, and the
adjacent-token same-set ratio — the first predictability signals P2 needs.

## Synthetic traces

Seeded generator with per-layer hot-expert subsets (15% of the pool) and
70% token-to-token set stickiness, so P2 development sees non-uniform,
temporally correlated routing without weights. Deterministic per seed.

## Online collection (blocked, tracked by `readiness`)

Needs: 48/48 verified checkpoint shards + small files (Issue #6), and
runtime modules `torch`, `tilelang`, `PIL`, `transformers`, `safetensors`.
Then: convert to TP8 (`inference/convert.py`), hook `model.py` routing to
append records (authoritative router untouched), and validate output with
`collect_trace.py validate`. Quality invariant holds: recording never
changes routing decisions.
