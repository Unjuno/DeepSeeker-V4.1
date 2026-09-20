# Status by issue (2026-09-21 JST)

Machine: M1 Max / 64GB / macOS 26.6.2 / python 3.14.5 / torch 2.14.0.
Upstream pin: `dba1be0a40aa45a94ad051997016db3960a90277` (all issues).

## Weights (Issue #6) — COMPLETE

- 48/48 shards verified (`verify: 48/48 PASS`, 510,296,708,312 bytes) + 4 small files.
- `status`: `complete: true` (sha256-vs-upstream-lfs + manifest reconciliation).
- Footprint: ~475GB under Git-ignored `models/DeepSeek-V4.1-Flash/`.
- Incidents during fetch, all resolved: duplicate downloader processes
  corrupting `.part` files (24 oversized parts deleted), per-connection
  decay fixed by 256MB rotation, concurrency raised 3→6 (~4x to ~13MB/s),
  repeated checksum-mismatches on shards 27/42 self-healed by retry passes.

## Bootstrap / reference (P0)

| Issue | Title | State |
|-------|-------|-------|
| #1 | Machine probe | DONE (`probe_machine.py`, `profiles/m1-max-64gb/machine.json`) |
| #2 | Tensor manifest | DONE (96,085 tensors, manifest reconciled) |
| #3 | Engram analysis | DONE (row-stream+cache policy; tables never fully resident) |
| #4 | Storage benchmarks | DONE (patterns + timings) |
| #5 | Memory budget planner | DONE (47.2GB operating budget, `resident-plan.json`) |
| #6 | Checkpoint manager | DONE (this section) |
| #7 | Reference verification | DONE — blocking checks PASS; found+fixed missing `encoding/` package (fetch patterns extended, 13 files); `verify_reference.py`. Full `generate.py` run still awaits runtime deps (below) |

## Offline analysis (P1–P2, P4–P5 inputs, P8) — all DONE, no weights needed

| Issue | Title | Key result |
|-------|-------|------------|
| #8 | Trace schema + tooling | `deepseeker.trace/v1` JSONL; synthetic generator; `collect_trace.py` |
| #9 | Routing simulator | 7 predictors; persistence wins small budgets (recall 0.71@6), ema at 24 (0.79, costly) |
| #10 | Shadow learner | Switch-on protocol; all BLOCKED under default bars (honest negative); horizon semantics verified |
| #11 | Resident-pool sim | demand/prefetch/Belady; persistence-prefetch ≡ demand (needs lookahead); Belady cuts miss 37%@24 |
| #12 | Autotune recommender | persistence@6, horizon 1, prefetch OFF (P3 gate), slots 6, 8 CPU workers |
| #13 | Budget allocation | greedy 0.738 vs equal 0.734 — near-uniform demand, placement not the lever |
| #14 | Motifs + blocks | bigram top-20 coverage 0.1% (unpredictable); 8-token reuse ratio 0.35 (P6 GO); bundle joins baselines |
| #15 | KV model + unified budget | KV @131k ctx = 102.5MB (≈1/1000 of old estimate); unified 19.1GB vs 45.1GB → 25.9GB headroom; experts are the battle |

## Test/lint health

118 tests, all passing. `ruff check scripts/ src/ tests/` clean.

## Blocked / next

1. **Runtime deps**: `Pillow`, `tilelang==0.1.8` missing (tilelang likely needs source build on ARM).
2. **TP8 conversion + first `generate.py` run** (`convert.py --model-parallel 8`, `run.sh`): needs deps + 48/48 (now satisfied).
3. **Online traces** (P1 second half): hook `model.py` routing → `collect_trace validate`; re-run #9–#13 numbers on real data.
4. **P4 real scheduler / P5 joint scheduling / P6 live blocks / P7 kernels**: gated on (2)+(3).
