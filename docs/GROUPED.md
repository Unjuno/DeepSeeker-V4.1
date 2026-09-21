# Grouped expert execution prototype (Issue #31, roadmap P6)

Planner (`plan_groups`, stable row order) + MLX grouped SwiGLU
(one batched GEMM set per non-empty expert, router-weighted
scatter-add) + naive per-(token, expert) oracle. Equivalence gate:
max abs diff vs naive, FAIL otherwise.

## Usage

    python scripts/bench_grouped.py --tokens 8,32 --experts 64 --iters 10

Random bf16 weights: grouping math is dtype-independent; shapes
(5120/2304, top-6) are real, expert count is scaled (64 of 384) to fit
bench memory. Live speculative-block distributions plug into
`plan_groups` unchanged when #30 unblocks.
