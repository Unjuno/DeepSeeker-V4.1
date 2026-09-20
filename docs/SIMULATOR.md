# Offline routing + cache simulator (Issue #9, roadmap P2)

Compares next-expert-set predictors on Issue #8 traces under fixed
per-layer memory budgets. Causal: each record is predicted before the
simulator observes it, so cold-start misses are counted honestly.

## Baselines

- `persistence` — previous token's set, same layer/request
- `lru` — per-layer most-recently-seen experts
- `lfu` — per-layer most-frequent experts
- `ema` — per-layer exponentially-decayed frequency (alpha 0.2)
- `token_transition` — first-order P(set_t | set_{t-1}) per layer
- `layer_transition` — P(set_layer | set_{layer-1}) within a token

## Usage

    python scripts/collect_trace.py emit-synthetic --requests 4 --tokens 64 --out /tmp/t.jsonl
    python scripts/run_routing_sim.py --trace /tmp/t.jsonl --budgets 6,12,24

Columns: `recall@k` (mean covered fraction of the true top-k),
`exact` (exact-set rate), `fp/rec` (wasted prefetches per record),
`waste/rec` (fp × median expert bytes, 18.8 MB from the Issue #2 manifest).

## Reading the table

- Budgets are per-layer resident slots out of 384 experts.
- `recall@k` rises monotonically with budget for count-based predictors;
  the gap between predictors at small budgets (6–12) is the signal that
  decides whether P4 prefetching is worth building.
- High `fp/rec` with high recall means the predictor is hedging: good
  coverage, expensive bandwidth. P3 switch-on criteria need both.
- Synthetic traces have 70% token stickiness by construction, so
  `persistence` starts strong; online traces will reorder the ranking.

## Not yet covered

Global (cross-layer) budget allocation, multi-token speculative blocks,
and motif/bundle predictors. Add them here once online traces show the
simple baselines plateauing.
