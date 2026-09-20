# Routing motifs + block working sets (Issue #14, roadmap P2/P6)

Two offline analyses that close the deferred P2 items and open P6.

## Layer bigram motifs

`layer_bigrams` counts ((layer, expert) → (layer+1, expert)) pairs within
tokens. `motif_coverage` reports what fraction of adjacent-layer traffic
a top-M table captures. High coverage at small M means a compact
transition table can steer P6 block verification or SANT-style lookahead;
low coverage means routing is effectively unpredictable across layers.

## Speculative-block working sets

`block_working_sets` groups B consecutive tokens per (request, layer);
`reuse_ratio` = mean unique experts / (B × top_k). 1.0 = every token is
all-new; lower = block-level weight reuse a DSpark-style verifier could
amortize over. Read the curve, not one point: sublinear growth in B is
the P6 precondition.

## Bundle predictor

`bundle` (windowed frequency, default W=4) joins the Issue #9 baselines:
it recalls experts recurring within a short horizon where single-token
persistence alternates away. Compare it with the others via
`run_routing_sim.py` — it needs no code changes, only `--predictors`.

## Usage

    python scripts/analyze_motifs.py --trace /tmp/t.jsonl --blocks 1,2,4,8 --top-motifs 10
    python scripts/run_routing_sim.py --trace /tmp/t.jsonl --budgets 6,12 --predictors persistence,bundle
