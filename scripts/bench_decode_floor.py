#!/usr/bin/env python3
"""Issue #16: MLX decode-math proxy for DeepSeek-V4.1-Flash MoE geometry.

Measures GPU time for one token's routed-expert math using 40 layers x top-6
SwiGLU experts, hidden 5120 / intermediate 2304, on the M1 Max GPU.

The proxy uses BF16 random weights because MLX does not expose the real packed
FP4 expert path used by the checkpoint.  BF16 is 16 bits/value while raw FP4
is 4 bits/value, so raw BF16 traffic is 4x raw FP4 traffic, not 2x.

For the actual checkpoint, Issue #2 measured one encoded routed expert,
including its scale tensors, at 18,800,640 bytes.  Therefore the 240 selected
experts for one token contain about 4.20 GiB encoded payload versus about
15.82 GiB in this BF16 proxy: a 3.7647x traffic ratio.

This benchmark must not be converted into an end-to-end FP4 tok/s claim by a
simple 2x factor.  Real FP4 dequantization, dense layers, attention, scheduling,
and verification costs require separate measurement/modeling.
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
ENCODED_EXPERT_BYTES = 18_800_640


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
    parser = argparse.ArgumentParser(description="MoE decode-math BF16 proxy.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    print(f"device: {mx.default_device()}", flush=True)
    experts, x = build()
    bf16_bytes = N_LAYERS * TOP_K * 3 * HIDDEN * INTER * 2
    bf16_gib = bf16_bytes / 1024**3
    raw_fp4_gib = bf16_gib / 4.0
    encoded_fp4_bytes = N_LAYERS * TOP_K * ENCODED_EXPERT_BYTES
    encoded_fp4_gib = encoded_fp4_bytes / 1024**3
    proxy_to_encoded_ratio = bf16_bytes / encoded_fp4_bytes

    print(
        f"bench weights: {bf16_gib:.2f} GiB BF16; "
        f"raw FP4 equivalent: {raw_fp4_gib:.2f} GiB; "
        f"actual encoded selected experts: {encoded_fp4_gib:.2f} GiB "
        f"(BF16 proxy / encoded = {proxy_to_encoded_ratio:.4f}x)",
        flush=True,
    )

    for _ in range(args.warmup):
        y = swiglu(x, *experts[0])
        mx.eval(y)

    naive = bench_naive(experts, x, args.iters)
    grouped = bench_grouped(experts, x, args.iters)
    result = {
        "device": str(mx.default_device()),
        "geometry": {
            "layers": N_LAYERS,
            "top_k": TOP_K,
            "hidden": HIDDEN,
            "inter": INTER,
        },
        "weight_gib_bf16_proxy": round(bf16_gib, 4),
        "raw_fp4_gib_same_parameter_count": round(raw_fp4_gib, 4),
        "encoded_fp4_selected_expert_gib_manifest": round(encoded_fp4_gib, 4),
        "bf16_proxy_to_encoded_fp4_ratio": round(proxy_to_encoded_ratio, 6),
        "ms_per_token_naive": round(naive * 1000, 1),
        "ms_per_token_grouped": round(grouped * 1000, 1),
        "proxy_ceiling_tok_s_naive": round(1 / naive, 2),
        "proxy_ceiling_tok_s_grouped": round(1 / grouped, 2),
        "note": (
            "MoE BF16 proxy only. Do not scale by 2x to infer FP4 runtime. "
            "The encoded checkpoint expert payload is 3.7647x smaller than "
            "the BF16 proxy, but real FP4 dequantization and all non-MoE "
            "runtime costs must be measured separately."
        ),
    }
    print(json.dumps(result, indent=1))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=1) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
