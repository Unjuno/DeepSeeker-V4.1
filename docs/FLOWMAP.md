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

## MTP module traffic estimate (target verification excluded)

Code analysis of the MTP modules: `dspark_block_size` = 5 drafts, 3 MTP
layers routing top-3 of 128 experts at ~18.75MB each (7.22GB mtp-expert / 3 /
128), plus windowed attention and markov/confidence heads. This estimates
**~0.2GB of MTP-module traffic per sequence-forward** (`MTP_DRAFTS`,
`MTP_VERIFY_BYTES` in `flowmap.py`). It does not include the target model's
multi-token verification forward.

The current `generate_spec()` commits candidates through serial target forwards.
An optional `--spec-batch-probe` separately calls the full backbone on
`[target token] + drafts` to collect diagnostic comparisons; its outputs are
discarded after cache restore and do not determine committed tokens. Its
expert traffic is determined by the union of routed experts across candidate
rows, so it can be several times ordinary single-token traffic. This diagnostic
probe is off by default. The analytic MTP rows below omit target-backbone
verification and are optimistic bounds, not MTP throughput predictions.

### Observed target-verification traffic (2026-10-17)

On M1 Max / MLX 0.32.2, prompt `Say hello.`, top-6, expert cache cap 256,
block size 5, each target verify contained 6 rows. Across four verify passes,
the runner read 16.949–17.947 GiB of routed-expert data per pass (mean
17.536 GiB). Each pass averaged 24.95 unique experts per layer across 40
layers, versus 36 token-expert assignments per layer before grouping. Across
160 layer-pass observations, expert row counts were M=1: 2,951, M=2: 606,
M=3: 255, M=4: 102, M=5: 44, M=6: 34. Thus about 48.8% of routed rows were
grouped into an expert used by multiple verify rows, but most expert groups
still had one row. These are one prompt and four verify passes, not a general
routing distribution or a throughput result. The measured run accepted 0/20
drafts, so it provides no evidence of a useful speculative speedup. A later 24-token, K=2 run after restoring KV RMSNorm accepted 6/20 drafts with two batched full-accept steps, but its output diverged from ordinary decode at output index 22. Replaying and validating all drafts serially restored exact token parity, while MTP ran at 0.238 tok/s versus 0.857 tok/s baseline (0.278x). Thus correctness currently costs more than the batched path saves.

The previous single-sequence threshold derived from 11.9 GiB per MTP forward
must not be used as a target acceptance requirement: it omitted the observed
target-verification traffic. A corrected throughput model needs measured
verify traffic, acceptance length, cache hits, and replay cost together.

### Cache-size and pin probes

Additional 8-token runs on the same prompt compared total expert-cache budgets.
With no static pins, cap 256 measured 0.132 tok/s and 17.54 GiB mean read per
verify; cap 512 measured 0.071 tok/s and 16.62 GiB; cap 1024 measured 0.0448
tok/s and 13.96 GiB. The cap-1024 run reduced observed expert reads, but free
memory fell to 36% during the run and throughput was lower. These are single
runs under changing machine load, so they do not establish a causal curve or an
optimal cap; they do show that increasing residency alone did not help this
MTP workload in these conditions.

At total cap 256, pin-6/layer left 16 dynamic LRU slots and measured 0.120
tok/s with 33.4 GiB mean verify read; pin-4/layer left 96 slots and measured
0.147 tok/s with 18.32 GiB mean verify read. Neither single run establishes a
pin speedup over no-pin cap 256 (0.132 tok/s, 17.54 GiB). Pin-6's LRU capacity
is smaller than the observed 24.95-expert/layer verify union; cache and
prefetch duplication is a likely explanation for its higher reads, but needs
direct cache-event instrumentation and repeat measurements.

### Ordinary decode capacity probe (2026-10-17)

A same-Runner ABBA comparison of ordinary `Runner.generate` used the same
prompt and eight output tokens at cap 256/768/768/256, clearing sequence and
expert-cache state between runs. All four token sequences were identical.
Cap 256 read 31.87 GB and measured 0.782/0.386 tok/s; cap 768 read 25.04 GB
(21.4% less) and measured 0.0959/0.0905 tok/s. The cap-768 runs used 37.6 GiB
MLX active memory and system free memory fell from 92% to 57%, recovering to
91% after the run. Cap 256 used 28.6 GiB active memory. A1/A2 themselves drifted
2x, so the exact speed ratio is not a stable estimate, but both cap-768 runs
were slow despite fewer bytes. Larger residency did not improve this workload
under these conditions; further capacity increases are not justified without a
memory-aware controlled run. See the detailed user report and
`work/cachecap_abba.json` for per-run counters.

## Historical aggregate scenarios — optimistic and unvalidated

These rows use a=2 as a parameter and the earlier ~11.9 GiB MTP cost, which omitted the measured full target verification traffic above. They are retained as historical arithmetic only; none is a measured or currently supported MTP throughput prediction. All rows use s=0.

| B | a | aggregate tok/s | per-sequence tok/s | GiB/forward | interpretation |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 10.7 | 10.7 | 11.7 | analytic baseline |
| 1 | 2 | 31.7 | 31.7 | 11.9 | stale optimistic MTP arithmetic |
| 8 | 2 | 76.6 | 9.57 | 40.9 | stale optimistic aggregate arithmetic |
| 32 | 2 | 105.3 | 3.29 | 120.4 | stale optimistic aggregate arithmetic |
| 128 | 2 | 190.9 | 1.49 | 266.2 | stale optimistic aggregate arithmetic |
| 275 | 2 | ~400 | 1.46 | ~273 | stale optimistic aggregate arithmetic |

The B=275 result is also only historical arithmetic. The current MTP implementation has not demonstrated positive speedup, and its exact serial verification path is slower than ordinary decode. These rows do not support a 40 tok/s single-sequence claim.

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

The next decisive measurements are live MTP acceptance rate, real multi-token
weight reuse, and authoritative routing traces.

### Ordinary decode storage bound (2026-10-17)

An instrumented 8-token `Runner.generate` decode at expert cap 256 measured
31.867 GB routed-expert reads (3.983 GB/output token) at 0.7925 tok/s. An
independent `read_bytes` wrapper matched the runner's decode byte count. The
manifest contains 275.67 GiB of expert payloads, plus 16.7 GiB measured dense
resident weights; all expert payloads plus dense tensors exceed this 64 GiB
machine by a wide margin. At the observed read volume, 40 tok/s would require
about 159 GB/s of useful expert reads if cache reuse stays unchanged. This is
an arithmetic requirement, not an observed SSD bandwidth or a hard upper bound
for all future kernels/cache strategies.

The current `ExpertLoader.io_s` counter wraps both `pread` and worker-side
`_pack_expert`/`mx.array` creation. It overlaps the separately accumulated
`dequant_s` worker sum and must not be interpreted as pure I/O time. See the
research report for the per-worker instrumentation, thermal/memory snapshots,
and scope limitations.

### Nonuniform per-layer cache allocation replay (2026-10-17)

I replayed the existing `profiles/runs/mtp/route_trace_200.json` decode route trace (200 decode positions, 40 layers, 6 experts/layer; 48,000 routed expert references). The first 100 positions were used to greedily allocate a 256-entry layer-local LRU budget by marginal training hit count; the last 100 positions were held out. The resulting allocation concentrated slots in a handful of layers (e.g. layers 4:36, 9:33, 13:37, 17:34, 19:31, 20:24, 30:29), leaving many layers with zero. On the held-out half, simulated hit counts were 6,601/24,000 for global 256-entry LRU, 4,552/24,000 for uniform 6-per-layer, and 4,117/24,000 for the train-optimized nonuniform allocation. Thus this trace does not support replacing the current global LRU with static per-layer budgets; the proposed budget is worse on held-out data. These are route-trace cache simulations only, not runtime measurements. The trace was collected from one prompt/run and does not establish workload generality.

### Batch=1 async MoE pipeline ABBA (2026-10-17)

Ran `scripts/bench_batch.py` four times on the same `Say hello.` prompt, 8 generated tokens, cache cap 256, order ordinary/async/async/ordinary. All four output token lists were identical (`[43,4571,2887,304,2507,1613,1,0]`), with 31.867 GB reported decode expert reads, 1,695 loads, and 1,920 hits each. Ordinary throughput was 0.3824 and 0.3581 tok/s; async was 0.5834 and 0.3103 tok/s. Median ratio async/ordinary is 1.207, but the paired ratios are 1.526 and 0.866, so the speed effect is inconclusive under run drift. The machine was under concurrent load, and baseline throughput (0.31–0.58 tok/s) is materially slower than the earlier 0.79 tok/s ordinary decode run. Async output reports `decode_io_s=0` although the same expert bytes were loaded; the pipeline's worker path does not update the ordinary I/O wall-time counter. Thus the pipeline currently lacks comparable phase accounting. Raw result files are in the task `work/pipeline_abba_{A1,B1,B2,A2}.json`. No default was changed.

### Async worker timing instrumentation (2026-10-17)

The batch=1 async MoE pipeline previously reported `decode_io_s=0` because its direct worker path bypasses `ExpertLoader.load_experts_parallel()` accounting. I added separate `decode_pipeline_read_worker_s` and `decode_pipeline_pack_worker_s` counters, returned in the batch result. They are summed per-worker times and overlap across experts; they must not be added to one another or interpreted as elapsed phase wall time. A post-change 8-token run retained the same output sequence and 31.867 GB decode byte count, with 0.5857 tok/s, 22.869 worker-read seconds, and 8.231 worker-pack seconds. Worker sums exceeding the 13.66s decode wall confirm substantial overlap, while the timing alone does not reveal how much was hidden behind GPU compute. During validation, extending the saved Future tuple initially exposed a stale five-field scatter unpack; that unpack was corrected and the full generation then completed with the same tokens. The default path remains off. The targeted Python files pass byte-compilation and `git diff --check` passes; no test suite was run.

### Route-prefetch research: where the current temporal warm can help (2026-10-17)

The ordinary batch-1 decode issues 6 routed experts in each of 40 backbone layers, a 240-expert working set. With no static pins, `expert_lru_budget` enforces a minimum 240-entry cache and the default is 256. After one complete decode forward, the previous token's 240 backbone experts therefore fit in the LRU; at the start of the next token, `prefetch_last_routes()` only schedules experts absent from that cache and consequently has no demand-read candidates in this mode. This conclusion is limited to single-sequence, backbone-only decode with the stated cache policy; batch and MTP namespaces can exceed that working set. Route-trace replay agrees: same-layer previous-token top-6 overlap is 25.7%, while adjacent-layer same-token overlap is 1.72% (close to the 1.56% random expectation for 6-of-384), so naïve cross-layer reuse is not a useful predictor.

The more relevant lead is a trained same-layer pre-attention router. The existing experiment that applies the true gate weights to stale `xpre`/`xin` features achieved only 1.56% top-1 exact-router agreement and would nearly double reads if it speculatively fetched one wrong expert per route. The ETH pre-attention study instead trains per-layer two-linear predictors against the original router's selected set, reporting 93.03% exact-set accuracy on DeepSeek-V2-Lite and 98.65% when loading 10 candidates for a six-expert route; its study used 10M activation samples and an NVIDIA training host. Those numbers are not transferable to DeepSeeker-V4.1 or M1 Max, but the method is compatible with a quality-preserving prefetch-only experiment: keep the checkpoint router authoritative and discard predictions that are not selected. A small model-specific held-out study is the next useful gate before implementation.

Another open implementation, Edge0, reports a trained cross-token prerouter and +59% decode gains in its own Qwen3.6/Ling model tiers. Its predictions replace the original route and its published model bundles include recovery LoRA, so that path changes model semantics and is not a direct quality-neutral port. Its README reports 14.9–17.7 tok/s for its 35B tier on an M4 Pro, not this model/hardware. Sources: ETH paper https://arxiv.org/abs/2511.10676; Edge0 paper https://arxiv.org/abs/2609.18063; implementation https://github.com/Edge0-AI/Edge0.

### Pre-attention top-1 speculative read (2026-10-17)

Using the pre-attention route capture pilot, I added an opt-in experiment: `DEEPSEEKER_PREATTN_PREFETCH=1`. Before each ordinary batch-1 backbone attention block, the checkpoint gate scores `xpre` and starts reading its top-1 expert. The original post-attention router still chooses all six experts; the candidate is admitted only if it is among those six. Wrong candidates are discarded after accounting for the wasted read. The flag is off by default and disabled for MTP and the MoE pipeline.

On the held-out 35-token pilot, pre-attention top-1 was in the true top-6 for 91.57% of route events; exact top-6 set agreement was only 7.93%. This supports using it as a speculative read candidate, not as a replacement router. The small same-prompt sample does not establish generalization.

An 8-token `Say hello.` A/B/A/B run produced identical tokens in all runs: `[43,4571,2887,304,2507,1613,1,0]`. Ordinary tok/s was 0.0760 then 0.9696; pre-attention tok/s was 0.4026 then 0.6053. The timing spread is too large to claim a speedup; the second pair favored ordinary decode (0.9696 vs 0.6053 tok/s). Each speculative run predicted 320 candidates, of which 290 were in the authoritative top-6, 258 reads were used, and 28 were wasted. Used and wasted reads totaled about 5.38 GB; total decode bytes were 32.39 GB versus 31.87 and 29.68 GB for ordinary. The pipeline overlapped these reads with attention, but this short run does not show a reliable end-to-end benefit. Keep the option experimental and disabled by default.

Raw outputs are `work/pre_attn_{A1,B1,A2,B2}.json`. Validation: Python byte-compilation and `git diff --check` passed. No test suite was run.

### Top-2 pre-attention speculative reads (2026-10-17)

The saved one-prompt route capture was re-scored for candidate rank: rank 1 was in the authoritative top-6 for 91.57% of held-out events; rank 2 was in it for 80.36%. This is only 35 held-out tokens from one prompt. I extended the disabled-by-default experiment to allow `DEEPSEEKER_PREATTN_TOPK=2`; the exact post-attention router still determines execution, and speculative experts are admitted only when that router selects them. The two read workers run concurrently.

A same-Runner A/B/B/A, 8-token run produced the same token sequence in every arm. Ordinary A1/A2 measured 0.4944/0.3213 tok/s; top-2 B1/B2 measured 0.3509/0.3166 tok/s. The run-order drift remains too large for a precise speed effect; the second pair was effectively tied (B2/A2=0.986), and the first pair favored ordinary. Top-2 made 640 predictions, 538 candidates were in the authoritative top-6, 477 reads were used, and 93 were wasted; wasted data was 1.75 GB. Total decode bytes rose from 31.87 GB to 33.62 GB. Thus top-2 did not show a speed win and added I/O; leave it opt-in for further controlled work.

The local OpenCode latest session was resumed twice. Its recommendations repeated already completed work (RoutePred measurement and asynchronous MoE), even after the exact result files and prior work were called out; those recommendations were rejected after source inspection. New external research found an independent MLX SSD-streaming implementation reporting 4.5–5 tok/s on M5 Pro/48GB with a 1664-expert cache, but that is a small, self-reported project on different hardware and a different checkpoint path. It is a comparison lead, not an expected result for this M1 Max checkout. See [upstream repository](https://github.com/ssd-moe/deepseek-v4-flash-mlx). Its measured headline itself attributes the ceiling to streaming experts from SSD; learned prefetch work on other models reports at most 20% improvement over an offload runtime, not a 40 tok/s recipe ([SpecPrefetch paper](https://arxiv.org/abs/2607.24787)).

Raw same-Runner output: task `work/pre_attn_top2_same_runner_abba.json`. Python byte-compilation and `git diff --check` passed; no test suite run.

### Cross-prompt popular-expert pins (2026-10-17)

I captured six prompts x 32 decode tokens (10,920 layer-route rows including prefill). Training on prompts 0–3 and holding prompts 4–5 out, the global LRU-256 simulated 2,559/15,360 route hits (16.66%). Pinning the six most frequent experts in each layer and leaving 16 dynamic LRU slots (same 256 total slots) gave 4,119/15,360 hits (26.81%); the held-out prompt rates were 24.36% and 29.27%. This is a small two-prompt holdout, not a general workload estimate.

A same-Runner A/B/B/A on held-out prompt 4, eight output tokens, used the same popular top-6/layer pins and a 16-entry dynamic LRU. All arms produced `[32111,377,201,1613,1,0,5,223]`. Ordinary tok/s was 0.2847/0.2609; pin tok/s was 0.3394/0.3579. Decode reads fell from 31.85 GB to 27.92 GB (12.3%); same-run pair ratios were 1.19x and 1.37x. But total reads including prefill rose from 66.03 GB to 93.80 GB because the 16-entry dynamic cache thrashes on prompt prefill. Thus this setting improved decode alone while worsening full-request I/O.

The first pin ABBA, before the new temporal-prefetch guard, read 50.40 GB during decode with pins and measured no stable speedup. `prefetch_last_routes()` was filling a 16-entry dynamic cache with a previous-token miss union larger than the cache. After adding a guard to skip that impossible-to-reside union, the 16-entry pin ABBA returned to 27.92 GB and showed the decode result above. A second run kept the default 256 dynamic LRU plus 240 pins (496 total entries): decode bytes fell further to 26.06 GB, but speed was inconclusive (pin 0.2461/0.2616 vs ordinary 0.0920/0.2646 tok/s), with a large A1 outlier under concurrent Ollama memory use. Total reads including prefill were 61.25 GB vs 66.03 GB; no consistent speed gain was proven for the extra cache memory.

This suggests a next experiment: retain a larger temporary LRU during prefill, then shrink to a 16-entry dynamic LRU for decode while keeping the 240 static pins. It could avoid prefill thrashing without changing decode's pin hit set; test full-request bytes and latency before adopting it. Raw route and ABBA data live in task `work/multi_prompt_routes.json`, `work/popular_pins_abba.json`, and `work/popular_pins_abba_cap256.json`. Python byte-compilation and `git diff --check` pass; no test suite run.

### Prefill/decode split cache with cross-prompt pins (2026-10-17)

Added an opt-in Runner setting `expert_prefill_cache_cap` (default 0, preserving
existing behavior). When set above the decode LRU cap, generation uses the
larger dynamic cache during prompt prefill, then shrinks back to the decode cap
before sampling. `shrink_expert_cache()` now measures the cap against
non-pinned residents and never evicts static pins. The split applies to
`generate()` and the per-sequence prefill in `generate_batch()`.

On held-out prompt 4 from the earlier six-prompt capture, same-Runner A/B/B/A
with 8 generated tokens compared ordinary LRU256 against 240 popular pins plus
prefill LRU256/decode LRU16 (256 total decode entries). All token sequences
matched exactly. Ordinary tok/s was 0.2832/0.2680; split-cache pin tok/s was
0.3272/0.3493. Decode expert reads fell 31.85 to 27.92 GB (12.3%), and full
request reads fell 66.03 to 63.11 GB (4.4%). Prefill wall time was similar
(21.26–22.07s pinned, 22.04–23.44s ordinary). This is one prompt and eight
tokens, and timing is noisy; it is a promising cache/byte result, not evidence
of a 40 tok/s path. Artifacts are in task scratch at
`work/popular_pins_prefill256_decode16_abba.json` and
`work/benchmark_popular_pins_prefill_abba.py`. Python byte-compilation and
`git diff --check` passed; no test suite was run.

### More static pins at the same 496-entry decode footprint (2026-10-17)

Replayed the six-prompt route capture with training prompts 0–3 and held-out
prompts 4–5. With a 16-entry dynamic LRU and top-N train-selected static pins
per layer, simulated held-out route hit rate rose from 26.82% at N=6 to 39.14%
at N=12. This replay is limited to two held-out prompts.

Measured a same-Runner A/B/B/A on held-out prompt 4, eight generated tokens:
ordinary LRU256 versus 480 static pins (top-12/layer) plus dynamic LRU16,
using prefill LRU256. All token lists matched. Decode expert reads fell from
31.85 to 24.03 GB (24.6%); whole-request reads fell from 66.03 to 61.16 GB
(7.4%). Throughput was 0.2875/0.2752 tok/s ordinary and 0.2538/0.3254 tok/s
with pins. The paired ratios split (0.88x and 1.18x), so this is not a proven
speedup despite the lower read bytes. The decode cache holds 496 entries, the
same entry count as the earlier 240-pin + LRU256 configuration but with more
space assigned to popular pins. Raw route replay and ABBA results are in task
scratch `work/pin_lru_fixed16_sweep.json` and
`work/popular_pins12_prefill256_decode16_abba.json`. Byte-compilation and
`git diff --check` pass; no tests were run.

### 24 static pins/layer probe stopped for memory pressure (2026-10-17)

The two-prompt route replay predicted 52.29% holdout hits for top-24/layer
pins plus a 16-entry dynamic LRU (960 static pins). On held-out prompt 4, the
first A/B measurements used 8 output tokens and prefill LRU256. A1 ordinary
LRU256 measured 0.2935 tok/s, 27.25s decode, 31.85 GB decode reads. B1 with
960 pins measured 0.0956 tok/s, 83.65s decode, and about 19.08 GB reads; the
8 output token IDs matched. During B1, system-wide free memory fell to 39%.
The run was stopped before B2/A2 because the large pin set cut reads while
tripling decode time under memory pressure. This single pair is not an ABBA
speed conclusion. After shutdown, system-wide free memory returned to 93%.
This probe shows that the simulated route-hit gain does not ensure lower
latency when the hot expert set grows beyond the machine's practical resident
budget. Partial record: task scratch `work/popular_pins24_partial.json`.

### 16 pins/layer repeat with reversed order (2026-10-17)

To separate first-run/order effects from the 16-pin ABBA, I repeated the same
held-out prompt 4 and 8 generated tokens in BAAB order on a fresh Runner. Each
run matched the same token IDs. Across both four-run sequences, ordinary
LRU256 decode reads were 31.85 GB; 640 static pins + LRU16 read 22.03 GB
(30.8% less). Full-request reads fell 66.03 to 60.61 GB (8.2%). But
throughput varied substantially: ABBA ordinary 0.2006/0.3936 tok/s vs pin
0.3760/0.3479; BAAB pin 0.1353/0.1008 vs ordinary 0.2907/0.2600. Combined
medians were 0.2753 ordinary and 0.2416 pin (pin 12.2% lower), with pairwise
direction changing across order. Thus the lower byte count did not yield a
reproducible speed gain. The concurrently resident Ollama model and large
machine-wide memory variation remain measurement caveats; free memory was
54% at one in-run sample and 89% after the runner stopped. Raw data:
`work/popular_pins16_prefill256_decode16_abba.json` and
`work/popular_pins16_prefill256_decode16_baab.json`. `git diff --check` and
Python byte-compilation pass; no tests run.

### MTP target forward: batch/serial logits diverge before routing (2026-10-17)

A teacher-forced comparison on held-out prompt 4 used the same six target
inputs from ordinary decode and restored the exact same prefill cache before
each path. Batch M=2 and serial M=2 kept the same greedy token IDs, but logits
already differed by up to 2.92 (row-1 margin was only 0.10). At M=4, row 2's
argmax changed from token 1613 (serial) to 261 (batch); max logit difference
was 3.69. A plain full-forward timing for M=4 was 2.57s batched vs 6.35s
serial, but this is one prompt/one run and does not establish end-to-end
speculative throughput.

A layer-instrumented repeat found attention outputs already differed at layer
0 by up to 0.00195, while MoE output differences at that layer were at most
4.62e-5. The first routed-expert set difference appeared at layer 3, row 3.
The block hidden-state difference grew with depth. Instrumentation forced an
`mx.eval` at each layer, so its measured batch/serial times are not comparable;
only the activation and route parity observations are used. These observations
point to batch-shape numerical differences in attention as the earliest
measured source, followed by router-set changes. They do not yet identify the
specific attention contraction. A lossless multi-row MTP optimization must
remove or exactly reproduce this difference; the current serial replay guard
remains necessary. Raw diagnostics are in task scratch
`work/batch_serial_logits_diagnostic.json`,
`work/batch_serial_first_layer.json`, and
`work/batch_serial_first_stage.json`.

The follow-up instrumentation compared layer-0 attention inputs, Q tensors,
sparse-attention output, and the final attention output. The layer-0 attention
input was bit-identical for all four rows. Q was identical for rows 0–1, then
differed by up to 0.0078125 and 0.0009765625 for rows 2–3. Sparse-attention
outputs differed by at most 3.05e-5; the final attention output differed by
0.000488–0.001953. This narrows the numerical drift to attention's batch-shaped
projection/output path, but does not identify one exact matmul. Artifact:
`work/attention_batch_inputs.json`.

An opt-in attempt to serialize attention rows while retaining batched MoE did
not restore parity: M=4 still had different argmax on row 1 (270 vs 201), max
logit delta 3.57, and took 9.14s versus 5.20s for ordinary serial forward. It
also exposed a candidate-mask tail mismatch caused by advancing indexer state
row by row; the experimental branch was removed. Keep the ordinary path and
serial replay guard. Batch-shape drift also remains in MoE, so attention-only
serialization is not a viable shortcut.

### 2026-10-17: shared-expert overlap probe and SSD-runtime research

I tested whether the route-independent shared expert could start before the
host copies routed expert IDs from MLX. The opt-in probe called `mx.async_eval`
on the shared branch immediately after routing, before `np.asarray(indices)`.
On one Runner, same prompt, eight generated tokens and ABBA order, all four
token lists matched exactly. Each arm read 31,867,084,800 expert bytes and
recorded 1,695 loads / 1,920 cache hits. Ordinary decode measured 0.782 and
0.658 tok/s (median 0.720); the overlap branch measured 0.616 and 0.591
(median 0.603, 0.838x). This is a short, single-prompt result; it gives no
speedup evidence. I removed the opt-in source change and kept the raw output at
task scratch `work/shared_overlap_abba.json`.

New primary-source leads from the separate `antirez/ds4` runtime are more
substantial but need port-specific validation. Its V4.1 Q4 report on an M5 Max
128 GiB records 10.18 tok/s upstream, 18.66 with one SSD, and 22.25 with two
byte-identical NVMe replicas; output SHA-256 matched across arms. Reported
single-drive techniques include GPU-written route-ID mailboxes with CPU
polling (+6.6%), post-MoE eviction work, small exact fusions, and a revised
read path. On a different V4 Flash model, an M4 Max 36 GiB report measures
3.20 to 4.29 tok/s (+34%) for code changes at identical settings and output.
These reports are author measurements, other code/checkpoints and newer Macs;
they are candidate mechanisms, not projections for this MLX/M1 Max runner.

The next source review is to compare the current `Runner.moe` cache and reader
path with the `ds4` single-drive changes, then select one small, attributable
piece. The naive MLX shared-branch overlap tested above is not the GPU mailbox
design and was slower. MTP's small-M MXFP4 kernel lead remains secondary
because current MTP acceptance is low and end-to-end slower than serial decode.
Current OpenCode session `ses_eb98047b9ffeIjyi0ZFqG1H62F` has an unanswered
request to review layer-0 attention drift; CLI commands currently exit
137/SIGKILL, so I read its saved history without mutating the session.

### 2026-10-17: direct `preadv` into MLX-owned expert buffers

An opt-in path in `src/deepseeker/mlx_weights.py` allocates/evaluates MLX-owned uint8/uint32 buffers on the caller thread, creates writable host views, then fills them from cached read-only shard descriptors using `os.preadv` in the I/O pool. This avoids the normal intermediate NumPy byte buffers and MLX copy. Enable only for experiments with `DEEPSEEKER_DIRECT_PREAD=1`; default remains off.

Same Runner, same `Say hello.` prompt, expert cache cleared per arm, eight generated tokens, with ABBA and then reverse BAAB order: all eight token lists matched; every run reported 31,867,084,800 decode bytes, 1,695 expert loads, and 1,920 cache hits. Ordinary decode median was 0.2760 tok/s (range 0.267–0.291); direct `preadv` median was 0.3391 tok/s (range 0.312–0.379), ratio 1.228 (+22.8%). All four order-matched ratios favored direct reads (1.162, 1.273, 1.358, 1.145). This is a promising but short, one-prompt result, and not a 40 tok/s claim. Keep it opt-in until longer decodes and additional prompts confirm the gain. Raw logs: task scratch `work/direct_pread_abba.json` and `work/direct_pread_baab.json`.

Longer direct-read follow-up: a same-Runner 16-token ABBA comparison also had exact output token parity and identical per-arm counters (61,760,102,400 decode bytes; 3,285 loads; 3,840 hits). Ordinary median was 0.2877 tok/s; direct `preadv` median was 0.3189 (+10.9%), with both pairs favoring direct reads. The smaller uplift than the 8-token run means the effect size is not yet stable. This remains opt-in and is far from the 40 tok/s target. Raw log: task scratch `work/direct_pread_16_abba.json`.

### 2026-10-17: follow-up reader comparisons show strong run-order sensitivity

The direct `preadv` path was checked on a second prompt in three 16-token ABBA/BAAB sequences. Token IDs and read/load/hit counters matched for every arm, but order-matched direct/ordinary ratios ranged 0.758–1.968 (median 1.031). This does not establish a stable gain; keep `DEEPSEEK_DIRECT_PREAD` off by default.

I compared direct-read sequential weight/scale ranges with a temporary parallel-run branch. Eight-token ABBA/BAAB order-matched ratios were 1.323, 1.048, 1.200, and 0.995; 16-token ratios were 0.712 and 1.202. A cache-warmed 16-token ABBA yielded 0.982 and 0.977, so I removed the branch. Same-Runner worker-count experiment (1 vs 6 workers) showed the single-worker path slower, but the configured production pool already defaults to 16 and overlaps distinct experts; do not raise that default based on this result.

All these runs preserved token IDs and per-arm expert read counts. Clearing the in-process LRU did not clear macOS's file cache, while adjacent-run timings varied substantially. Do not treat these microbenchmarks as controlled cold-storage results. Raw task scratch logs: `work/direct_pread_16_prompt2_abba.json`, `work/direct_pread_16_prompt2_baab.json`, `work/direct_pread_16_prompt2_abba_repeat.json`, `work/parallel_runs_abba.json`, `work/parallel_runs_baab.json`, `work/parallel_runs_16_abba.json`, `work/parallel_runs_16_warm_abba.json`, `work/io_threads_1v6_abba.json`.

OpenCode's latest session was continued with `opencode run --session ses_eb98047b9ffeIjyi0ZFqG1H62F`. It returned two source-unsupported attention recommendations (`compute_weights`/`qkv`, then `wk`/`einsum`); neither function/path exists in current `Runner.attention()`, and no code was changed. Live source uses `wq_a` -> RMSNorm -> BF16 `wq_b` projection -> reshape/RoPE. Current diagnostics prove input equality, Q equality for rows 0–1, and Q divergence for rows 2–3, but do not identify which internal projection first differs. Next diagnostic should capture the `wq_a` product, RMSNorm result, pre-RoPE `wq_b` result, and final Q under batch/serial execution.

### 2026-10-17: layer-0 Q divergence at the `wq_b` projection

The new probe `work/diagnose_attention_projection_stages.py` captures the real layer-0 `wq_a` product, `q_norm` output, its BF16 cast, post-RoPE Q, and replayed `wq_b` matmuls for the same four teacher-forced rows used by the prior batch/serial diagnostic. Attention inputs are bit-identical. `wq_a` product error is only 2.2e-8–6.0e-8; `q_norm` output error is <=1.43e-6. The BF16 input to `wq_b` differs only on row 2 by 5.96e-8. The final Q is exact for rows 0–1 and differs by 0.0078125 / 0.0009765625 on rows 2–3. Replaying actual captured inputs through `wq_b` reproduces the deltas. A same-input batched-vs-single-row `wq_b` replay also produces the row-2/3 deltas, showing that batch-shaped GEMM arithmetic is sufficient to cause the Q drift. Different M=4/M=1 Metal matmul paths are a plausible explanation, not directly verified. Full logits still diverge later through MoE; this does not establish that changing Q alone restores MTP parity. Candidate next experiment: row-wise `wq_b` projection under an opt-in switch, with exact Q/logit and latency comparisons.

### 2026-10-17: temporary row-wise Q projection experiment rejected

An opt-in experiment ran multi-row `wq_a`/RMSNorm/`wq_b` one row at a time. All four layer-0 Q rows matched serial bit-for-bit, but sparse-attention outputs still differed by up to 3.05e-5 and attention outputs by up to 0.00195. In three prompts, row-wise Q matched serial greedy argmax for 2/3 prompts; ordinary batch matched 1/3. The third prompt still diverged, and full logits differed by max 3.29–4.76. One-shot timings were noisy and used different expert byte totals; row-wise Q was slower than serial on two prompts. I removed the source branch after it failed the full parity and speed gates. No opt-in flag remains. Raw measurements: task scratch `work/qprojection_serial_multiprompt.json`; baseline and temporary Q-row stage captures are in `work/attention_projection_stages.json` and `work/attention_projection_stages_qrow.json`.
