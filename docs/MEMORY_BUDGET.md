# Resident-memory budget (64 GiB M1 Max)

## Safe-capacity probe

```bash
python scripts/probe_memory_capacity.py [--max-compressor-mb 512]
```

Holds 8→48 GiB (half numpy-touched, half MLX-held) for 5 s per level,
stops on any swap onset or >512 MiB compressor growth, releases between
levels, verifies recovery. Never destabilized the machine across 4 runs.

Measured ceilings varied with background load: 24 / 32 / 32 / 48 GiB
(operating = ceiling − 4 GiB reserve). The committed plan uses the
cleanest run (48/44 GiB). Capacity is STATE-dependent: re-run quiescent
before final runtime sizing; the planner accepts any capacity file.

## Planner

```bash
python scripts/plan_resident_memory.py [--engram-cache-mib 64]
    [--staging-experts 8] [--cold-ms-per-expert X]
```

Inputs: `memory-capacity.json` + Issue #2 manifest + Issue #3 Engram
hit rates + Issue #4 service times. Output: `resident-plan.json`/`.md`.

Every component carries provenance
(measured / manifest / arch-estimate / assumed); estimates expose their
formula and multiplier. Engram lookup tables are never resident
(0 B by construction); index-state is an explicit HIGH-uncertainty
estimate that binds long-context feasibility.

## Headline plans (44 GiB operating, 64 MiB Engram cache, 8-expert staging)

- S1@4K: 1889 slots (12.3%); S1@32K: 1679 (10.9%, ~42/layer);
  S1@128K: 957 (6.2%).
- S2 (MTP): −40 slots vs S1 (3 MTP reserve slots + static).
- S3 (multimodal): −52 slots vs S1 (0.9 GiB vision).
- S4 long context: 512K/1M flagged OVER (+33.8/+101.1 GiB shortfall)
  under the high-uncertainty index estimate — surfaced, not hidden.

## Recommendations

- Engram cache: 64 MiB default (prose hit 0.74–0.84; 16 MiB already
  strong on short traces; nothing justifies GiBs).
- Staging: 8 experts default (140 MiB); pool is insensitive to
  staging size (1655–1686 slots for 32→1), so prefer deeper staging.
- Storage modes: optimistic 0.90 ms / advisory 0.96 ms (measured),
  unknown-cold parameterized (default 3x advisory).
- Next unknown to kill: per-token sparse-index state size (binds 512K+),
  then quiescent capacity re-run, then route locality.
