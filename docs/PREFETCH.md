# Gated predictive prefetch (Issue #25, roadmap P4)

Predictor → staging → pool, with the router authoritative: every serve
resolves through pool lookup with demand fallback, so ON/OFF outputs are
byte-identical by construction (compared in the A/B run).

## Gate

Prefetch submits only while enabled; a trailing window of per-token
set-agreement below `min_agreement` auto-disables to demand-only
(distribution-shift revert). A BLOCKED verdict is a first-class outcome,
not a failure — gates #21/#22 decide production enablement on real data.

## Usage

    python scripts/run_prefetch.py --trace /tmp/t.jsonl --slots 6 --max-tokens 6
    python scripts/run_prefetch.py --trace /tmp/t.jsonl --slots 6 --max-tokens 6 \
        --predictor ema --budget 12 --horizon 1 --min-agreement 0.5 --window 3

Reported: quality identity, OFF/ON stalls + hit rates, submits,
served-prefetched, stale count, trailing agreement, verdict GO/BLOCKED.

## Honest limits

Lookahead submits predictions, not guarantees: late misses fall back to
demand staging (wait) or direct load. Waste = staged-but-unused bytes
(stale counter × expert bytes). Overlap math (did prefetch actually hide
latency?) belongs to #28 scheduling, not here.
