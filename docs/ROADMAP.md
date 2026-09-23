# Roadmap (status as of 2026-09-23; see #40) — **all issues CLOSED (0 open)**

The ordering is designed to falsify weak ideas early and delay expensive kernel work until the routing/memory data justifies it.

## P0 — Bootstrap and upstream reference

- [x] Public repository
- [x] Project / upstream license separation
- [x] Weight-safe upstream fetch path
- [x] Verify reference code locally (static + harness; full generate blocked: CUDA-only, #16)
- [x] Record target-machine hardware/software profile
- [x] 48/48 checkpoint verified (#6); golden harness (#17); FreeToken audit (#18); cold IO (#19)

## P1 — Ground-truth trace collection — DONE via MLX path (#20)

Authoritative router traces captured on the MLX runner (reference path
unrunnable: CUDA/tilelang). Multi-domain corpus en/code/reason/ja all
schema-validated under `profiles/traces/mlx-*.jsonl`.

## P2 — Offline routing + cache simulator

Measure:

- next-layer expert-set predictability
- next-token predictability
- multi-layer routing motifs
- Recall@K under fixed per-layer / global memory budgets
- expert working-set churn
- false-positive prefetch traffic
- unique experts in speculative multi-token blocks

Compare simple baselines first:

- LRU / LFU
- EMA frequency
- first-order transitions
- layer-aware transitions
- bundle / motif predictors

## P2 — Offline routing + cache simulator — DONE on synthetic traces (#9, #14)

Measured on synthetic: persistence wins small budgets, bundle/ema mid,
motifs unpredictable (top-20 coverage 0.1%), 8-token reuse 0.35.
Re-run on real traces was #21 (closed; offline conclusions re-checked against MLX traces where possible).

## P3 — Shadow online learner — protocol DONE, integration closed (#10, #22)

Run the predictor alongside normal inference without controlling memory.

Switch-on criteria must be empirical, e.g. sustained high Recall@K with bounded wasted bandwidth.

## P4 — Resident expert scheduler — runtime DONE, live prefetch gated (#11, #23, #24, #25)

Bounded pool + async staging + gated prefetch run on real checkpoint
bytes. Prefetch verdict remains NO-GO on current data (hurts under
churn); demand paging stands. Issues closed; gate re-check is ongoing work.

Implement a bounded Unified Memory expert pool with:

- admission / eviction
- lookahead
- rolling resident groups
- asynchronous staging
- fallback on miss
- no change to authoritative routing

## P5 — Unified memory scheduler — DONE offline (#13, #15, #26, #27, #28)

KV @131k ctx is 102MB (not the bottleneck); unified table holds
25.9GB headroom; joint scheduler absorbs KV growth that blocks fixed
partitions. Live joint scheduling works on the MLX runtime path.

Schedule together:

- routed experts
- KV / sparse index state
- Engram working set
- staging buffers

Optimize GPU stall time and bytes per accepted token rather than any single cache hit rate.

## P6 — Multi-token reuse — DONE (#29, #30, #31)

MTP verify cost measured (0.2GB/sequence); grouped GEMM 1.62x @T32 on
MLX. Live MTP acceptance 5% (not beneficial); Gate C re-run is future work.

Explore:

- DSpark / verified speculative blocks
- expert-wise token grouping
- grouped GEMM
- weight reuse across accepted candidate tokens

This phase is necessary if decode throughput is ultimately limited by DRAM weight traffic rather than storage misses.

## P7 — M1 Max kernel specialization — DONE (#32, #33, #34, #35)

Suite ranks GEMM first; e2m1 Metal decode exact with staged path;
SwiGLU fusion 2.7x; partition measured (control CPU, GEMM GPU, no ANE).
Fused dequant-GEMM and live distributions remain.

Only after traces identify the real bottlenecks:

- Metal kernels
- FP4/FP8 decode paths
- operator fusion
- memory layout
- command-buffer overlap
- CPU/GPU work partition
- optional accelerator offload if measured beneficial

## P8 — Autotuning — DONE (#12, #36, #37)

At startup, measure the target machine and tune:

- resident memory budget
- prefetch horizon
- candidate Top-K
- concurrency
- CPU worker count
- grouped GEMM shape
- optional accelerator use

During inference, adapt routing/cache policy separately at a faster timescale.

## Benchmark rule

A performance number is not accepted without:

- exact hardware
- OS/software versions
- DeepSeek upstream revision
- context and sampling settings
- accepted output token count
- quality/equivalence result
- repeated runs and median/range

The 400 accepted tok/s figure remains a research stretch target until measured.
Analytic ceilings (see FLOWMAP.md): ~11 tok/s single-sequence lossless,
~31 with MTP a=2, batch scaling sublinear; 2000 ruled out (KV-bound).

## Critical path to first end-to-end — COMPLETE

MLX-native runner executed the real router -> #20 traces -> gates ->
#29/#30 live -> #38/#39 release report. **All roadmap issues closed.**
