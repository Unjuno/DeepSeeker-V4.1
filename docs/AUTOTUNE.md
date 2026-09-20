# Startup autotuning (Issue #12, roadmap P8)

`scripts/autotune.py` turns committed measurements into one deterministic
startup config. It does not benchmark live; it reconciles artifacts.

## Inputs

- `profiles/m1-max-64gb/machine.json` — 8 performance cores, 64 GB unified.
- `profiles/m1-max-64gb/resident-plan.json` — 47.2 GB operating budget,
  315 MB dense Engram reservation.
- Issue #2 manifest — 18.8 MB median expert bytes, top-6 routing.
- Optional `--sim` / `--shadow0` / `--shadow1` JSONs from the Issue #9/#10
  CLIs. Absent inputs select documented priors, never fail.

## Rules (each emits its rationale)

- `resident_slots_per_layer`: (operating − dense − 4 GB OS reserve) /
  40 layers / expert bytes, clamped to [top_k, 384], then capped by the
  sim-chosen budget.
- `predictor`/`budget`: smallest budget reaching target recall (default
  0.7); within 5% of the best recall there, least wasted bytes wins.
- `horizon`: token-ahead iff its best recall is within 10% of horizon-0.
- `prefetch_enabled`: only on a shadow switch-on verdict for the chosen
  predictor — otherwise the system stays shadow-only (P3 gate respected).
- `cpu_workers`: one per performance core. `io_concurrency`: 6, the
  measured Issue #6 download sweet spot, flagged for P5 re-measurement.

## Runtime adaptation (not this issue)

Faster-timescale policy switching during inference belongs to P5/P8
online work: feed live recall/waste back into these same rules. The rule
functions are pure and importable for that loop.
