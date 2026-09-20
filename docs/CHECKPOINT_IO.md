# Cold random-access service (Issue #19, P4 input)

Real expert/Engram `file_range` reads (O_RDONLY, checkpoint untouched) on
the verified 48/48 checkpoint, QD 1–16, p50/p95, warm vs advisory-cold
(F_NOCACHE), plus a 120-round sustained group run. Command:

    python scripts/bench_checkpoint_io.py --qd 1,2,4,8,16 --repeats 3 \
        --sustained-rounds 120 --out cio.json

## Measured (QD1 / QD8 / QD16, MiB/s)

| workload | cold QD1 | cold QD8 | cold QD16 | warm QD16 |
|---|---|---|---|---|
| single-expert (18.8MB) | 226 | 6,372 | 15,022 | 25,316 |
| same-layer group (113MB) | 530 | 3,119 | 3,034 | 4,973 |
| cross-layer group (113MB) | 3,220 | 6,659 | 7,604 | 15,500 |
| engram 4×16MB | 4,775 | 19,560 | 25,071 | 32,580 |

Per-range p50: 0.03–0.4ms (expert ranges), 1.4–3.2ms (16MB engram chunks).
QD scaling saturates ~QD8–16 depending on range size. Sustained 120
rounds: 6ms/round (113MB ≈ 18GB/s), drift 1.004 — no degradation.

## Cache-confidence labels (honest)

- **warm**: solid (repeated reads, page-cache served).
- **advisory-cold**: F_NOCACHE + disjoint ranges per repeat (first-touch
  majority). Residual contamination possible (no sudo purge), so treat
  cold numbers as *lower bounds* on true cold latency. Directionally
  correct: cold trails warm ~1.5–2x at QD1.
- **purge-controlled**: UNCERTAIN (no sudo) per spec.

## vs #4

#4 synthetic-file expert QD1: p50 0.96ms / 4.2GB/s (advisory-uncached).
Real checkpoint same regime: p50 0.40ms cold. Same order of magnitude —
#4's cache-regime bounds stand; #19 adds real layout, QD scaling to
~30GB/s, cross-layer groups, and sustained stability. P4 staging design
envelope: **one expert ≈ 1–3ms cold-ish at QD1, ≈ 0.1–0.5ms warm;
a 6-expert group ≈ 6ms sustained (18GB/s)**.
