# Kernel microbenchmark suite (Issue #32, roadmap P7 input)

MLX micros at exact DeepSeek decode shapes + end-to-end relevance
ranking against the measured 109.8ms/token MoE floor.

## Usage

    python scripts/bench_kernels.py --iters 20 --json-out kernels.json

Micros: gate/up GEMM (B=1, 8), SwiGLU elementwise, fp8 block-dequant
sim (u8 + bf16 scales at 128-wide blocking), launch overhead.
Relevance multiplies standalone medians by executions/token: these are
UPPER bounds that do not sum to 100, because lazy fusion shares
launches and scheduling across the real 109.8ms floor (which is the
truth). The ranking still orders P7 targets: GEMM dominates either way.

## Gaps (explicit)

- No custom Metal: no Xcode toolchain on this machine. MLX baselines
  only; Metal scaffold is a future-machine task.
- fp8-dequant is simulated arithmetic (no real fp8 tensors in MLX);
  #33 owns the real kernel.
- Batch/group sizes use analytic distributions until #20 traces land.
