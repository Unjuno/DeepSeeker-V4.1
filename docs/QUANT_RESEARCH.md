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
