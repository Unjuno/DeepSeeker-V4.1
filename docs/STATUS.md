# Status by issue (2026-09-23 JST) — **all GitHub issues CLOSED (0 open)**

Machine: M1 Max / 64GB / macOS 26.6.2 / python 3.14.5 / torch 2.14.0.
Upstream pin: dba1be0a40aa45a94ad051997016db3960a90277 (all issues).

## Weights (Issue #6) — COMPLETE

- 48/48 shards verified (verify: 48/48 PASS, 510,296,708,312 bytes) + 4 small files.
- Local status reported complete: true with sha256-vs-upstream-lfs + manifest reconciliation.
- Footprint: ~475GB under Git-ignored models/DeepSeek-V4.1-Flash/.
- Incidents during fetch were resolved: duplicate downloader processes,
  per-connection decay, and checksum-mismatch retries.

## Bootstrap / reference (P0)

| Issue | Title | State |
|-------|-------|-------|
| #1 | Machine probe | DONE |
| #2 | Tensor manifest | DONE — 96,085 tensors, manifest reconciled |
| #3 | Engram analysis | DONE — row-stream + cache policy |
| #4 | Storage benchmarks | DONE — patterns + timings |
| #5 | Memory budget planner | DONE |
| #6 | Checkpoint manager | DONE — 48/48 verified |
| #7 | Reference verification | DONE — static/blocking checks passed; full generate still blocked on runtime portability |
| #16 | First inference baseline | DONE — MLX path 0.0387 accepted tok/s; reference FAIL (CUDA/tilelang) |

## Offline analysis — implemented

The offline tooling for #8–#15 is implemented and useful for plumbing and
counterfactual analysis, but its routing conclusions must be re-run on
authoritative online traces before controlling the runtime.

Notable prior offline results include:

- persistence was strongest at small candidate budgets in synthetic/offline routing data
- the shadow switch-on gate remained BLOCKED under its default bars
- Belady provided a useful resident-cache upper bound
- bundle/motif analysis showed weak simple bigram coverage
- multi-token blocks showed reuse signal worth measuring on real inference
- the corrected KV/index model is much smaller than the original rough estimate

These are not substitutes for real router traces.

## Issue #16 measurement status — RESOLVED

Reference runtime remains unrunnable on M1 Max (tilelang/NCCL/CUDA).
**MLX path is the measured baseline:** 0.0387 accepted tok/s
(`profiles/runs/mtp/base8.json`); full settings and quality check in the
#16 completion comment.

## Flow-map audit correction

The analytic flow-map and BF16 proxy were audited on 2026-09-21.

Corrections:

1. BF16 is 16 bits/value and raw FP4 is 4 bits/value.  The old statement that
   BF16 was only 2x FP4 traffic was wrong.
2. The actual manifest-selected routed-expert payload is about 4.20 GiB/token
   (including scale tensors) versus 15.82 GiB for the BF16 proxy, a 3.7647x
   traffic ratio.  This does not imply a 3.7647x runtime speedup because FP4
   dequantization/kernel/runtime costs differ.
3. Batch B > 1 in the flow-map means concurrent sequences.  Its tok/s is
   aggregate serving throughput, not the generation rate of one conversation.
4. Non-zero neuron sparsity is quality-unverified and is outside DeepSeeker's
   lossless claim until exact equivalence is proved.  It is excluded from the
   default flow table.
5. MTP rows with zero verification traffic are optimistic upper bounds until
   real candidate/verification cost is measured.
6. The flow-map does not reproduce the 109.8 ms BF16 proxy.  They model
   different traffic and are now documented separately.

Under the corrected manifest traffic model with sparsity=0 and an optimistic
zero-cost a=2 MTP assumption, 400 tok/s appears only as aggregate throughput
at a very large concurrent batch (about B=275).  Per-sequence rate in that
same modeled batch is about 1.46 tok/s.  This is not evidence for 400
single-sequence accepted tok/s.

The project stretch target remains:

    single_sequence_accepted_tok_s = 400

until the project explicitly decides otherwise.  Aggregate batched throughput
must never be substituted for that metric.

See docs/FLOWMAP.md for the corrected model and caveats.

## Correctness rule

DeepSeeker may:

- predict future expert choices to move weights earlier
- cache/reside immutable weights
- group verified tokens to reuse weights
- fuse/dequantize/optimize kernels while preserving semantics

DeepSeeker may not claim lossless acceleration by skipping required expert
neurons or changing authoritative router/verification decisions without an
independent exact-equivalence proof.

## Current next steps

All roadmap issues #1–#41 are closed. Follow-ups (not open issues):

1. Raise accepted tok/s from 0.039 toward the 400 stretch target
   (grouped MoE live path, expert-cap 960, MTP acceptance >5%).
2. Re-run sustained gate with tuned expert pool (post-#38 thrash finding).
3. Keep regression thresholds in the #39 section green on every change.

## Since (issues #16+) — all closed

- #16: reference unrunnable (proven); **MLX baseline 0.0387 accepted tok/s**.
- #17: golden harness closed (deterministic PASS, perturb FAIL).
- #18: FreeToken audit closed (reuse/adapt/reject map; geometry mismatch found).
- #19: cold checkpoint IO closed (expert ~1-3ms, sustained drift 1.004).
- #20: multi-domain authoritative traces closed (en/code/reason/ja schema-valid).
- #23/#24/#25: pool + staging + gated prefetch closed (NO-GO verdict).
- #26/#27/#28: Engram cache, KV manager, unified scheduler closed.
- #29: MTP speculative baseline closed (lossless, 0.32× baseline, not beneficial).
- #31: grouped MoE closed (1.11× @8, 1.62× @32 tok microbench).
- #32–#35: kernels/fusion/partition closed.
- #36/#37: startup autotuner + adaptive controller closed.
- #38: sustained gate closed (stability PASS / throughput CONDITIONAL FAIL).
- #39: e2e report + regression thresholds closed (400 tok/s NOT reached).
- #41: SSD policy closed (enforced + monitored).
- **Open GitHub issues: 0. Open PRs: 0.**


## Final end-to-end (Issue #39)

- Baseline accepted decode: **0.0387 tok/s** (M1 Max, greedy, 8 tokens, `profiles/runs/mtp/base8.json`).
- MTP speculative: lossless but **0.32× baseline** (acceptance 5%); not beneficial yet (`full8.json`).
- Grouped MoE microbench: **1.11×** @8 tokens, **1.62×** @32 (random experts, maxdiff 1.56e-2 gate).
- Startup autotune: `recommended.json` cap=960 slots; controller #37 unit-tested.
- Multi-domain authoritative traces: en/code/reason/ja all schema-validated.
- Sustained gate #38: 124 min monitor, stability PASS / throughput CONDITIONAL FAIL (expert I/O thrash at cap 64).
- **400 accepted tok/s stretch target: NOT REACHED** (measured 0.039).
- Golden quality harness: PASS / perturb FAIL as expected.

Regression thresholds (initial):

| metric | threshold |
|---|---|
| accepted tok/s (Say hello, 8 tok) | ≥ 0.030 (baseline 0.0387) |
| golden equivalence | PASS |
| trace schema validate | PASS all 4 domains |
| sustained free% min | ≥ 30 |
