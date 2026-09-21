# Fusion / layout / overlap (Issue #34, roadmap P7)

Measured candidates only; losers reverted by rule. ## Measured (batch-1 expert, M1 Max GPU)

- SwiGLU chain fusion: 2.035ms -> 0.743ms (**PASS**, 2.7x). Unfused
  intermediates 24KB/expert -> fused 10KB (saves 57%).
- Two-stream overlap on twin 2048 GEMMs: 4.755ms -> 4.605ms (**PASS**
  by rule, 3% — marginal, likely noise; overlap is not the lever).
- Launch ladder per token: 960 naive / 240 fused / 40 ideal evals.

## Usage

    python scripts/bench_fusion.py --iters 10 --json-out fusion.json
