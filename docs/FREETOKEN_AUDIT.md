# FreeToken audit (Issue #18, cross-cutting)

Source: `https://github.com/FlashML-org/FreeToken` v0.1.3, Apache-2.0
(Copyright 2026 FreeToken Authors / FlashML Team), cloned shallow to
`/tmp/freetoken` for review only. No FreeToken code is vendored; the one
ported contract (`moe_align.py`) carries provenance + license notice.
MIT (DeepSeeker) × Apache-2.0 is compatible with attribution.

## Blocking finding

FreeToken's `deepseek_v4` target is **dim 4096 / 43 layers / 256 experts** —
not our pinned 5120 / 40 / 384. Nothing model-specific transfers without
re-derivation. Audit verdicts below are mechanism-level only.

## UMA delta (the one paragraph)

FreeToken assumes discrete VRAM: PCIe H2D staging, pinned banks, CUDA
streams/graphs, per-GPU autotune tables, and a PCIe-vs-CPU hybrid split.
On M1 Max unified memory there is no PCIe leg, no pinning need, no VRAM
pool — only total-pressure budgeting and Metal/MLX ordering. Everything
that balances or copies *across* host/device boundaries is reject/adapt;
everything that decides *what* to keep or *in what order* is reuse/adapt.

## Reuse/adapt/reject (machine-readable: `profiles/audit/freetoken-map.json`)

Reuse: piece→pack→bank loading contract; FTW layout spec (JSON index +
raw bytes + per-layer grouping); `moe_align_block_size` contract
(ported, numpy, timed below); KV radix/prefix caches incl. SWA variant;
bench JSON schema + closed-form split formula.

Adapt: LRU/hit-split/fetch-cap policy (new transport); six-range async
I/O pattern (new syscalls); elastic split arithmetic (unified baseline);
prefill-first scheduler (MLX ordering); grouped-GEMM batching math
(sort/pad/scales → Metal); hybrid split + bench tiers (rebench Apple);
DSV4 indexer selection semantics (later).

Reject: all Triton/CUDA kernels + H100 tables; PCIe hybrid tuner;
O_DIRECT/pin plumbing; DSV4/DSA attention stack (until dense path works).

## Minimum test: align-contract benchmark

numpy port on M1 CPU, 256 tokens × top-6, E=384, block 16: 64.2ms.
That is NOT negligible next to the ~110ms/token MoE floor — the naive
per-expert scan is O(E·N) and must become an O(N) histogram build in P6
(#31). The benchmark paid for itself by catching this early.

## Recommendations linked to later issues

- #23 pool: adopt piece→bank contract + FTW-style per-layer grouping.
- #24 staging: PinPipeline pattern with pread/mmap + shared buffers.
- #27 KV: SWA radix variant fits our sliding-window arch directly.
- #28 scheduler: elastic split arithmetic on unified baseline.
- #31 grouped GEMM: `moe_align.py` is the front end; Metal body is P7.
- #32 bench: two-tier methodology + JSON schema, Apple rigs.
- #33 kernels: reject all Triton; fp8-roundtrip numerics only reference.
- #35 partition: closed-form split idea, Apple numbers from scratch.
