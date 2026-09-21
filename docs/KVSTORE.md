# KV/sparse-index runtime manager (Issue #27, roadmap P5)

Context-aware allocator over the #15 geometry with real buffers, hard
budgets, and model-vs-actual reconciliation (exact match required —
NumPy backing has no padding here, so drift means a geometry bug).

## Buffers

- `window`: (layers, batch, 128, 512) uint8 — fp8 bytes, exact.
- `compress:{layer}` / `index:{layer}` on the 4 kv-source layers with
  S//ratio columns; last dim halved (fp4 packs 2/byte) so allocated
  bytes equal the formula bit-for-bit.
- Values are opaque until P7 kernels interpret them; the manager moves
  memory, never math (attention semantics untouched).

## Usage

    python scripts/run_kvstore.py --seqs 4096,32768,131072,524288
    python scripts/run_kvstore.py --seqs 4096 --stress

Covers alloc/free/grow/reset lifecycle, pattern roundtrip, reset/reuse
across requests, a higher-context attempt, and a clean over-budget
failure with no partial state. Joint scheduling with experts/Engram is
#28, not here.
