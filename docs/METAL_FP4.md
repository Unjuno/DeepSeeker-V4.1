# e2m1 FP4 decode on Metal (Issue #33, roadmap P7)

Raw Metal kernels compile and run via MLX at runtime (no Xcode needed —
proven by `metal_kernel` smoke test; only `xcrun` CLI needs a license).

## Layout (from `inference/convert.py`)

Two e2m1fn nibbles per byte (low first) through
`[0,.5,1,1.5,2,3,4,6, 0,-.5,-1,-1.5,-2,-3,-4,-6]`, one ue8m0
power-of-two scale per 32 elements. Host decodes scales
(`decode_scales_u8m0`); the device kernel is LUT + multiply only.

## Measured (gate matrix 5120x2304, batch 1)

- Dequant equivalence: maxdiff **0.00** vs two independent references,
  incl. real checkpoint bytes (finite >99%, within e4m3 range).
- Staged GEMM == bf16 GEMM latency (~0.6ms); streaming traffic halved
  (11.2MB → 5.6MB per matrix). The win is bandwidth on cold loads,
  not resident latency.
- One-shot dequant: 0.9ms per 11.8M elems, amortized over reuses.

## Status vs #33 acceptance

Packed decode, gate/up/down + SwiGLU-compatible staged paths, single +
grouped shapes (via #31), numerical tests, bf16 fallback: done. Fused
dequant-GEMM (no materialization) is the open P7 item; staged is the
production fallback that already halves traffic.
