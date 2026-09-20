#!/usr/bin/env python3
"""Issue #16: MLX decode-math floor for DeepSeek-V4.1-Flash MoE geometry.

Measures GPU time for one token's routed-expert math (40 layers x top-6
SwiGLU experts, hidden 5120 / inter 2304) on the M1 Max GPU. Random bf16
weights (MLX has no fp4 GEMM; 2x the bytes of fp4, so pessimistic).

Two modes: naive (6 separate expert GEMMs/layer) and grouped (batched-6).
Attention/KV/router-topk excluded (router matvec included: negligible).
Result is a tok/s CEILING for MoE math alone: end-to-end can only be
slower (attention + weight traffic + scheduling on top).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

N_LAYERS = 40
TOP_K = 6
HIDDEN = 5120
INTER = 2304


def build() -> tuple[list, mx.array]:
    experts = []
    for _ in range(N_LAYERS * TOP_K):
        gate = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16) * 0.02
        up = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16) * 0.02
        down = mx.random.normal((INTER, HIDDEN), dtype=mx.bfloat16) * 0.02
        experts.append((gate, up, down))
    x = mx.random.normal((HIDDEN,), dtype=mx.bfloat16)
    mx.eval(experts, x)
    return experts, x


def swiglu(x: mx.array, gate: mx.array, up: mx.array, down: mx.array) -> mx.array:
    return ((x @ gate) * mx.sigmoid(x @ gate) * (x @ up)) @ down


def bench_naive(experts: list, x: mx.array, iters: int) -> float:
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        acc = None
        for gate, up, down in experts:
            y = swiglu(x, gate, up, down)
            acc = y if acc is None else acc + y
        mx.eval(acc)
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def bench_grouped(experts: list, x: mx.array, iters: int) -> float:
    grouped = []
    for i in range(0, len(experts), TOP_K):
        bundle = experts[i : i + TOP_K]
        gate = mx.stack([e[0] for e in bundle])
        up = mx.stack([e[1] for e in bundle])
        down = mx.stack([e[2] for e in bundle])
        grouped.append((gate, up, down))
    xx = mx.broadcast_to(x, (TOP_K, HIDDEN))
    mx.eval(grouped, xx)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        acc = None
        for gate, up, down in grouped:
            g = xx @ gate
            y = (g * mx.sigmoid(g) * (xx @ up)) @ down
            out = y.sum(axis=0)
            acc = out if acc is None else acc + out
        mx.eval(acc)
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE decode-math floor.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    print(f"device: {mx.default_device()}", flush=True)
    experts, x = build()
    weight_gib = N_LAYERS * TOP_K * 3 * HIDDEN * INTER * 2 / 1024**3
    print(f"bench weights: {weight_gib:.1f} GiB bf16 (fp4 real: half)", flush=True)
    for _ in range(args.warmup):
        y = swiglu(x, *experts[0])
        mx.eval(y)
    naive = bench_naive(experts, x, args.iters)
    grouped = bench_grouped(experts, x, args.iters)
    result = {
        "device": str(mx.default_device()),
        "geometry": {"layers": N_LAYERS, "top_k": TOP_K, "hidden": HIDDEN, "inter": INTER},
        "weight_gib_bf16": round(weight_gib, 2),
        "ms_per_token_naive": round(naive * 1000, 1),
        "ms_per_token_grouped": round(grouped * 1000, 1),
        "ceiling_tok_s_naive": round(1 / naive, 2),
        "ceiling_tok_s_grouped": round(1 / grouped, 2),
        "bytes_per_token_bf16_gib": round(weight_gib, 2),
        "note": "MoE math only; attention/KV traffic/scheduling excluded; bf16 2x fp4 bytes",
    }
    print(json.dumps(result, indent=1))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
