# Roadmap

The ordering is designed to falsify weak ideas early and delay expensive kernel work until the routing/memory data justifies it.

## P0 — Bootstrap and upstream reference

- [x] Public repository
- [x] Project / upstream license separation
- [x] Weight-safe upstream fetch path
- [ ] Verify reference code locally on target M1 Max
- [ ] Record target-machine hardware/software profile

## P1 — Ground-truth trace collection

Instrument the official reference path to record:

- prompt / request identity
- token position
- layer
- authoritative Top-k expert IDs
- router scores/logits where practical
- KV/index events
- Engram accesses
- timing and bytes moved

No predictive optimization yet.

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

## P3 — Shadow online learner

Run the predictor alongside normal inference without controlling memory.

Switch-on criteria must be empirical, e.g. sustained high Recall@K with bounded wasted bandwidth.

## P4 — Resident expert scheduler

Implement a bounded Unified Memory expert pool with:

- admission / eviction
- lookahead
- rolling resident groups
- asynchronous staging
- fallback on miss
- no change to authoritative routing

## P5 — Unified memory scheduler

Schedule together:

- routed experts
- KV / sparse index state
- Engram working set
- staging buffers

Optimize GPU stall time and bytes per accepted token rather than any single cache hit rate.

## P6 — Multi-token reuse

Explore:

- DSpark / verified speculative blocks
- expert-wise token grouping
- grouped GEMM
- weight reuse across accepted candidate tokens

This phase is necessary if decode throughput is ultimately limited by DRAM weight traffic rather than storage misses.

## P7 — M1 Max kernel specialization

Only after traces identify the real bottlenecks:

- Metal kernels
- FP4/FP8 decode paths
- operator fusion
- memory layout
- command-buffer overlap
- CPU/GPU work partition
- optional accelerator offload if measured beneficial

## P8 — Autotuning

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
