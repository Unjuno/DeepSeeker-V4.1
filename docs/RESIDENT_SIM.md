# Resident expert-pool simulator (Issue #11, roadmap P4)

Simulates a bounded per-layer resident pool over Issue #8 traces. Answers
the P4 design question offline: how much miss traffic can prefetching
close, and what does the optimal policy buy?

## Policies

- `demand_lru` — page in on miss, evict least-recently-used. No predictor;
  the floor every smarter policy must beat.
- `prefetch` — admit the predictor's guess (zero-latency staging, an
  optimistic bound), then demand-page with LRU eviction. Bad guesses can
  evict useful residents: churn is measured, not assumed away.
- `belady` — offline optimal eviction (farthest next use). The ceiling;
  the demand→belady gap is the prize prefetching plays for.

## Usage

    python scripts/run_resident_sim.py --trace /tmp/t.jsonl --budgets 6,12,24
    python scripts/run_resident_sim.py --trace /tmp/t.jsonl --budgets 12 \
        --policies demand_lru,prefetch,belady --predictor ema --json-out res.json

Columns: `hit_rate`, `miss/rec`, `missMB/rec` (misses × 18.8 MB median
expert bytes from the Issue #2 manifest), `evict`, `churn` (experts
evicted and later reused — the anti-thrashing signal).

## Honest limits

- Staging is zero-latency here; real async staging overlaps prefetch with
  compute (P5 scheduling), so live misses will exceed simulated ones.
- Budgets are per-layer slots; cross-layer sharing and KV/Engram
  contention belong to P5.
- Promotion to a real resident scheduler needs these numbers reproduced
  on online traces plus the P3 switch-on verdict.
