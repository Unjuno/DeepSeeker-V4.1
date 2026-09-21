# Unified scheduler (Issue #28, roadmap P5)

One byte budget across expert pool, KV/index, Engram pages, and
staging, with explicit priority (demand > kv_grow > prefetch >
engram), backpressure shed order, cross-cache stealing, and a
starvation guard. Optimizes stalls + bytes/use, never isolated hit
rates.

## Modes

- `independent`: fixed partitions; pressure refuses (KV growth can
  block). The baseline joint must beat.
- `joint`: KV growth steals Engram first, then expert slots; pressure
  sheds below the requester. Demand always passes.

## Usage

    python scripts/run_scheduler.py --trace /tmp/t.jsonl --ctx 32768 --budget-mib 8192
    python scripts/run_scheduler.py --trace /tmp/t.jsonl --ctx 32768 --budget-mib 8192 \
        --prefetch gated --max-tokens 4 --json-out sched.json

Mid-run the KV context grows 4x: joint steals and continues,
independent may block the growth (counted). Prefetch `gated` routes
expert misses through the #25 controller; `demand` pages directly.

## Starvation rule

Any class unserved for 200 operations gets a boost credit. Boosts are
counted in metrics — a rising boost rate means the priority order is
wrong for the workload, not that the guard is working.
