# Cross-layer budget allocation (Issue #13, roadmap P5 input)

Fixes a global expert-slot total, splits it across the 40 layers three
ways, and scores each split with the Issue #11 pool simulator.

## Strategies

- `equal` — even split. The baseline cross-layer sharing must beat.
- `proportional` — shares follow per-layer unique-expert demand with a
  top_k floor and largest-remainder rounding.
- `greedy` — hill-climbs single-slot moves on linearly interpolated
  hit-rate curves until no move improves total hits. Handles non-concave
  curves where naive water-filling stalls on flat stretches.

## Usage

    python scripts/allocate_budget.py --trace /tmp/t.jsonl --total 480
    python scripts/allocate_budget.py --trace /tmp/t.jsonl --total 480 \
        --curve-budgets 6,12,24,48 --json-out alloc.json

`--total` must cover top_k per layer. Curve measurement is the slow step
(40 layers × curve grid); reuse `--json-out` across questions.

## Reading the result

The equal-to-greedy hit-rate gap bounds what static cross-layer sharing
is worth. If greedy barely beats equal, layer demand is uniform and P5
should spend complexity on prefetch timing instead of placement. The
allocation that wins here becomes the P4 pool's static layout.
