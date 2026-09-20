# Analytic decode flow-map — corrected scope

This document is an analytic hypothesis model, not an end-to-end inference
measurement.

## Important corrections (2026-09-21)

Three earlier interpretations were wrong or insufficiently qualified:

1. **BF16 -> FP4 traffic**
   - The MLX proxy uses BF16 weights: 16 bits/value.
   - Raw FP4 is 4 bits/value, so raw BF16 traffic is 4x raw FP4, not 2x.
   - Using the Issue #2 manifest, the 240 selected routed experts contain
     about 4.20 GiB encoded FP4+scale payload, versus 15.82 GiB for the BF16
     proxy: a 3.7647x traffic ratio.
   - Real FP4 runtime cannot be obtained by simply dividing the BF16 timing by
     this ratio because dequantization/kernel/runtime costs differ.

2. **Batch throughput scope**
   - B means concurrent sequences.
   - For B > 1, B*(1+a)/latency is aggregate serving throughput.
   - The generation rate of one sequence is (1+a)/latency.
   - Therefore a modeled 400 aggregate tok/s at large B is NOT 400 tok/s for
     one conversation.

3. **Neuron sparsity and the quality invariant**
   - Non-zero neuron sparsity means skipping expert computation.
   - That is not lossless by default.
   - DeepSeeker must use s=0 for quality-preserving claims unless exact
     equivalence is independently demonstrated.
   - Sparsity scenarios remain useful only as explicitly marked
     quality-unverified counterfactual research.

There was also an incorrect statement that this flow model "reproduces" the
109.8 ms MLX benchmark.  It does not.  The 109.8 ms result is a BF16,
MoE-only proxy measurement.  This flow model uses manifest-derived encoded
expert bytes plus modeled dense weight traffic.  They are separate artifacts.

## Model

Aggregate accepted-token throughput across B synchronized sequences:

    aggregate_tok_s = B * (1 + a) / (bytes_fwd / BW + L)

Per-sequence accepted-token throughput:

    per_sequence_tok_s = (1 + a) / (bytes_fwd / BW + L)

where:

- B = concurrent sequences, not speculative tokens in one conversation
- a = expected extra accepted tokens per sequence/forward
- BW = effective Unified Memory bandwidth assumption
- L = launch/control latency assumption
- s = expert-neuron skip fraction; s must be zero for current lossless claims

Expert union across B independent sequences is approximated as:

    384 * (1 - (1 - 6/384)^B)

The approximation must be replaced by real routing-union measurements once
authoritative online traces are available.

## B=1 manifest traffic model

Current manifest-derived modeled weights per ordinary decode forward:

- routed experts selected by 40 layers x top-6: about 4.20 GiB
- dense/shared/router/head/index modeled fixed traffic: about 7.50 GiB
- total modeled weight traffic: about 11.70 GiB

At the current 132.7 GiB/s bandwidth assumption plus 5 ms launch/control
allowance, the analytic B=1, a=0 result is about 10.7 tok/s.

This is NOT a measured end-to-end result.  Attention compute, dequantization,
runtime scheduling, I/O stalls, cache effects, and other costs can lower it.

## MTP is currently an optimistic upper bound

The table can set a=1 or a=2 to ask "what if one forward produced one or two
additional accepted tokens?"

Until MTP candidate generation and target verification traffic/cost are
measured, a>0 with zero verification bytes is an optimistic upper bound.

In particular, the B=1, a=2 value of about 32.2 tok/s does not establish that
the real MTP path achieves 3x throughput.

## Lossless-compatible aggregate scenarios

All rows below use s=0.  MTP rows still assume zero verification cost and are
therefore optimistic.

| B | a | aggregate tok/s | per-sequence tok/s | GiB/forward | interpretation |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 10.7 | 10.7 | 11.7 | analytic baseline |
| 1 | 2 | 32.2 | 32.2 | 11.7 | optimistic MTP upper bound |
| 8 | 2 | 79.6 | 10.0 | 39.3 | aggregate serving throughput |
| 32 | 2 | 111.1 | 3.47 | 114.0 | aggregate serving throughput |
| 128 | 2 | 211.2 | 1.65 | 240.6 | aggregate serving throughput |
| 275 | 2 | 400.2 | 1.46 | 272.9 | aggregate only; optimistic MTP |

Thus, under this model with s=0 and zero-cost a=2 MTP, 400 aggregate tok/s
requires roughly B=275.  That is not evidence for 400 single-sequence tok/s,
and it is not yet a feasibility result because memory, latency, routing-union,
MTP verification, and compute constraints are not fully represented.

## Historical sparsity scenario — not a lossless claim

The former "400 tok/s needs B=125 with s=0.5" result is mathematically still a
counterfactual of this traffic equation, but it assumes skipping 50% of expert
neurons.  At B=125, a=2, s=0.5 the model gives about 402 aggregate tok/s and
about 3.2 tok/s per sequence.

This scenario MUST NOT be used as evidence that DeepSeeker can reach 400 tok/s
while preserving model quality.

scripts/flow_table.py excludes non-zero sparsity by default.  It is only shown
with the explicit --include-quality-changing option.

## What can still improve one-sequence throughput without degrading quality

The valid research directions are mechanisms that preserve authoritative
semantics while amortizing or hiding traffic/work:

- verified speculative/MTP blocks, with measured candidate + verify cost
- reuse of dense and expert weights across multiple accepted tokens
- expert-wise grouping and grouped GEMM
- correct resident expert reuse
- predictive prefetch that only moves weights early
- kernel/dequant/fusion improvements
- deterministic Engram row caching
- avoiding SSD stalls and swap

Predicting a future expert is quality-safe when it only changes *when* the
weight is loaded.  Skipping required expert neurons is a different operation
and is not quality-safe without an equivalence proof.

## Standing optimization problem

For serving throughput, optimize aggregate tok/s subject to memory and latency
constraints.

For the user's 400 tok/s stretch goal, track a separate metric:

    single_sequence_accepted_tok_s

Do not substitute aggregate batched throughput for that metric.

The next decisive measurements are real MTP verify cost, real multi-token
weight reuse, and authoritative routing traces.
