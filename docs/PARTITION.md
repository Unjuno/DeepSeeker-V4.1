# CPU/GPU work partition (Issue #35, roadmap P7)

Measured placement per task; ties go to CPU (GPU left for the data
plane, fewer sync points). ANE: unused — no CoreML toolchain on this
machine and dynamic routing shapes are unsuitable (explicitly allowed
as a PASS outcome by the issue).

## Usage

    python scripts/bench_partition.py --iters 20 --json-out part.json

## Measured (M1 Max, iters=20)

| task | CPU ms | GPU ms | placement |
|---|---|---|---|
| router-matvec | 0.063 | 0.505 | cpu |
| topk-select | 0.002 | 0.224 | cpu |
| engram-hash | 0.031 | — | cpu |
| expert-gemm | 1.230 | 0.641 | gpu |

Control-plane ops belong on CPU (GPU launch overhead dominates tiny
work by 8–100x); only expert GEMMs earn the GPU. Contention round
showed a faster contended GEMM (0.365 vs 0.641) — clock-ramp artifact
of measurement order, not a finding; UMA contention proper needs a
longer soak (#38).
