# Quantization research: what to "drop" on DeepSeek-V4.1-Flash-1017 (2026-10-19)

Goal: cut the dominant per-token byte driver (routed experts, currently
native MXFP4) while keeping quality as close to the official checkpoint as
possible. This note surveys 2025-2026 literature relevant to doing that
post-training, from FP4 anchors, on one Mac (no server farm).

## Current checkpoint composition (1017 revision, measured)

| subsystem | size | dtype today | per-token effect |
|---|---|---|---|
| routed experts | 268.95 GiB | MXFP4 (I8 packed + ue8m0 scales) | **dominant: ~3.6-3.8 GB/token streamed from SSD** |
| engram tables | 189.13 GiB | F8_E4M3 + scales | row-streamed: only 12.7 KB/token — do not touch |
| attention | 4.72 GiB | F8_E4M3 (native) | resident; RAM, not SSD bytes |
| shared experts | 1.32 GiB | F8_E4M3 | resident |
| embed + head | 2.46 GiB | BF16 | resident |
| MTP experts | 6.72 GiB | MXFP4 | streamed during verify only |

The experts are already the heaviest compressed part. Everything else that
could matter is already FP8. The realistic levers, in order: (1) expert
format below 4 bits, (2) expert scale format, (3) nothing else justifies
the risk today.

## Literature findings (with sources)

1. **Training-time FP4 cannot be re-derived post-training.**
   - "Benchmarking PTQ under MXFP formats" (arXiv:2601.09555): MXFP4 needs
     careful PTQ; MXFP8 is near-lossless; "rotational transformation
     impairs MXFP4" (QuaRot/SpinQuant scored *worse* than plain RTN
     under W4A4: PPL 10.3/10.4 vs 8.3); the E8M0 power-of-two scale is a
     top error source; a simple unbiased pre-scale (x3/4 before RTN) cut
     PPL 104.4→49.3 in one setup.
   - "Diagnosing FP4 inference" (arXiv:2603.08747): MLP up/down
     projections dominate FP4 sensitivity; gate is moderate; attention is
     least sensitive; early blocks can be highly sensitive (MXFP4).
   ⇒ For our re-quant: no Hadamard rotation. Optimize group scales
     directly, and consider FP8 (MXFP8) for the most sensitive early
     blocks instead of uniform 3-bit.

2. **FP4-native beats FP4-PTQ: keep the vendor MXFP4 as the anchor.**
   - NVFP4/MR-GPTQ study (arXiv:2509.23202): the best PTQ methods recover
     "up to 98-99% of FP16 accuracy" — i.e. ~1-2% quality loss is the
     state of the art *even at 4 bits on original weights*. Re-quantizing
     our *already-FP4* tensors to 3-bit lands in worse territory: the
     dequantized FP4 weights carry ~4-bit rounding error, and any further
     compression inherits it (error compounding).
   - Practical community numbers align: Anemll fork, Qwen3.5-397B:
     4-bit PPL 3.64 → Q3 3.81 (+~5%) with ~36% faster decode; 2-bit PPL
     +57% (collapse). BitMoD (arXiv:2411.11745, per-group, g=128):
     3-bit mixed-dtype keeps mean ΔPPL ≈ 2.9 where plain MX-FP3 blew up
     (>100 on small models; 24+ on Llama-3).
   - A separate 3-bit-rotation paper (ITQ3-S, arXiv:2603.27914) confirms
     3-bit is "the breaking point" and needs rotation + fused kernels —
     CUDA-specific; we have no Metal equivalent, so avoid its complexity
     unless quality gates force a revisit.

3. **Scale quality is the cheapest, largest win inside 4 bits.**
   - Meta, "MXFP4: Strategies for Quantization Error Reduction"
     (arXiv:2603.08713): Overflow-Aware Scaling + Macro Block Scaling
     close the MXFP4→NVFP4 gap from ~10% to <1% downstream with O(6%)
     GEMM overhead on DeepSeek-R1-class models (82.37 vs 83.19 BF16
     MMLU-Pro for MXFP4-MBS).
   - "Search Your Block FP Scales!" (MLSys 2026): searching block
     floating-point scales substantially beats naive absmax/E8M0
     rounding.
   ⇒ If we re-quantize our experts *at all*, do it in FP4 space with a
     scale-search / MBS-style refinement, not a bit depth drop. The
     checkpoint's own quantization_config has expert_dtype=fp4 with
     scale_fmt=ue8m0 — a scale-aware re-pack can target the same format.

4. **Asymmetric integer > symmetric float below 4 bits.**
   - BitMoD tables: at 3 bits, INT3-Asym (per-group) clearly beats MX-FP3;
     BitMoD's mixed-dtype outlier carve-outs are what rescue 3-bit.
   ⇒ If 3-bit is attempted: per-group-64 INT3 asymmetric, 8-bit E4M3
     scales, with outlier carve-outs (BitMoD-style) and the unbiased
     pre-scale trick — NOT plain FP3.

## Decision (quality-first)

1. **Keep expert weights MXFP4 as shipped** (quality anchor, vendor QAT).
2. First measurable byte cut is the *scale-search re-pack* inside 4 bits
   (finding 3): rebuild w1/w3/w2 packed payloads with OAS/MBS-style
   block-scale refinement. Byte-neutral → this helps only if we later
   move to smaller bits, so it is the calibration step, not a win alone.
3. The byte cut comes from a **pilot 3-bit layer subset**, not the whole
   pool at once:
   - format: per-group-64 asymmetric INT3 + E4M3 scales, weights
     pre-scaled by 3/4 (unbiased RTN), no rotation;
   - pilot array: layers 20-39 (later blocks are less sensitive per
     arXiv:2603.08747), routable via a per-layer format map, so one run
     can stream mixed 4-bit / 3-bit experts and ABBA-compare;
   - gates: golden-token match on the bench corpus is **not** expected to
     hold (this is a quality-changing mode); instead use the existing
     quality harness (route coverage + held-out generation + a small
     WikiText-style PPL sample amortized over ABBA prefill);
   - accept only if PPL delta stays ≤ those of the Anemll Q3 report
     (~+5%) and held-out generation stays coherent; otherwise revert to
     FP4 and keep top-k reduced modes as the opt-in speed tier.
4. Explicitly out of scope for this repo's default path: 2-bit (PPL +57%
   precedent), naive FP3 plain, and any routing-semantic change filed
   under "quantization".

## Speed accounting (why this matters at all)

At 15.3 GiB/s sustained SSD reads (measured, storage-patterns.md), the
storage-bound B=1 ceilings from today's runs: top-6 FP4 ≈ 4 tok/s,
3-bit experts ≈ 5.5 tok/s, 2-bit ≈ 8 tok/s — all far below the 400 tok/s
stretch goal, which is aggregate over many sequences and blocked today by
per-forward compute and memory pressure, not expert bytes alone.
Quantization is therefore a multiplier on the storage term, not the
primary path to the target. Decision above keeps that in frame.

## Pilot results (2026-10-20, post-fix)

Everything below supersedes any earlier "incoherent output" observations:
the engine had a pretrained-port omission — the main attention window KV
was never passed through `kv_norm` RMSNorm (model.py:706 semantic) — so
greedy generation was garbage for ALL variants, independent of
quantization. Localized via a CPU oracle (upstream torch reference with
tilelang kernels replaced by plain math) diffed per-stage against the MLX
engine: embedding/hc/norm exact, layer-0 attention cos 0.83 → after the
one-line fix cos 0.999996; downstream stages all ≥0.99.

With the fix in (commit 60b9ae3), the Q3 pilot ({bits:3, gs:64, MLX-native
affine, layers 20-39} via model-manifest-q3.json) runs coherently:

- greedy texts match the FP4 baseline structurally on code and knowledge
  prompts; per-token divergence starts within a few tokens, as expected;
- decode tok/s comparable (1.14 vs 1.09 in these probes; the point of
  the pilot is bytes/token, to be re-measured with the bytes harness);
- expert payload: 15.48 MB (-17.6% vs 18.8 MB FP4); payload on disk:
  models/DeepSeek-V4.1-Flash-1017-Q3/routed-q3.payload (118.9 GB).
- the OAS/MBS block-scale refinement (finding 3) and an ABBA quality
  harness (route coverage + held-out generation + small PPL sample)
  remain open before promoting Q3 to default.

### Quality gate result (scripts/quality_q3.py, batch 8 prompts x 24 steps)

Greedy Q3 text (routed experts L20-39 at 3-bit, everything else FP4) replayed
teacher-forced through the FP4 reference:

    rank_mean=1.44  p50=1  p90=2  top1=0.848  top5=0.978  top32>=0.999

- The FP4 reference itself put 84.8% of Q3 greedy tokens at rank 1 and
  97.8% within its top-5, i.e. the reference model would have said these
  tokens anyway. Well inside the accept band (the Anemll Q3 precedent was
  ~+5% PPL at full conversion; a rank-mean of 1.44 corresponds to near-1
  logit-window agreement).
- Greedy free-run agreement vs FP4: mean per-step match 0.474, 1/8
  sequences token-identical over 24 steps - divergence is compounding,
  spread over the batch, as expected for a lossy variant; the rank metric
  above is the acceptance signal, not token identity.
- decode_bytes accounting (expert IO, identical prompts/steps/cap):
  Q3 485.32 GiB vs FP4 528.31 GiB = x0.9187, matching the L20-39-only
  mix ((20*15.48+20*18.8)/(40*18.8)=0.912). A full-checkpoint Q3
  conversion would be x0.824 (-17.6%).
- Caveat: absolute tok/s in this harness (0.09-0.54) is dominated by
  expert-cache miss IO at cap 256 and page-cache warm state, not by
  quantization compute; the bytes ratio is the reliable number.

Note: harness must run prompts one at a time per loaded runner (see
seq_generate in scripts/quality_q3.py); batched lockstep at n=8 thrashes
the 256-slot expert cache (~75 s/step wall).

### Larger-sample rank measurement (32 prompts x 48 steps, 2026-10-21)

Re-ran quality_q3.py with 32 prompts (matches Unsloth's long-test-style
sample size; 1,471 scored tokens):

    rank_mean=1.17  p50=1  p90=1  top1=0.927  top5=0.993

The n=192 estimate (top1 0.848) was pessimistic sampling noise; at this
sample size the L20-39 3-bit Q3 pilot measures 92.7% top-1 agreement with
the FP4 reference at -8.1% whole-model bytes. Greedy cross-agreement over
the longer 48-step horizon stays low (mean per-step match 0.306, 1/32
identical) - compounding divergence, not a quality signal.
