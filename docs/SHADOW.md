# Shadow online learner (Issue #10, roadmap P3)

Runs each predictor alongside inference replayed from a trace. The shadow
logs what it *would* prefetch; it never controls memory by construction
(no admission/eviction path exists in this module).

## Horizons

- `horizon 0` — record-causal; intra-token prefixes allowed (layers run in
  order, so this is online-legal). Matches P2 recall exactly.
- `horizon 1` — token-ahead; prediction for token t uses only tokens < t.
  `layer_transition` legitimately scores ~0 here: with no prefix executing
  yet, there is nothing to condition on. Its real prefetch window opens
  intra-token at horizon 0.

## Switch-on rule

`switch_on(log, window, min_recall, max_waste_per_token)` returns the first
token index whose trailing `window` tokens *all* clear both bars, else
`None` (stay a shadow). Defaults: window 20, recall ≥ 0.7, waste ≤ 200 MB
per token. Tune per question: the bars, not the code, encode policy.

## Usage

    python scripts/run_shadow.py --trace /tmp/t.jsonl --budget 12 --horizon 1
    python scripts/run_shadow.py --trace /tmp/t.jsonl --budget 12 --window 20 \
        --min-recall 0.7 --max-waste-mb 200 --json-out shadow.json

A `BLOCKED` verdict is data: the predictor stays a shadow. Promotion to
P4 residency control requires a sustained switch-on *on online traces*,
not synthetic ones.

## Online hookup (blocked, same gates as P1)

Checkpoint 48/48 + runtime deps (`readiness` in `collect_trace.py`), then
drive `run_shadow`'s loop from live `model.py` routing callbacks instead
of `read_trace`. The loop, horizons, and criteria are unchanged.
