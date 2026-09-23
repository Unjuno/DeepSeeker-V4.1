#!/usr/bin/env python3
"""Issue #31: GEMV-per-route vs grouped GEMM at DeepSeek shapes (MLX).

    python scripts/bench_grouped.py --tokens 8,32 --experts 64 --iters 10

Random bf16 weights (grouping math is dtype-independent); shapes are
real (5120/2304, top-6). Reports naive/grouped latency, grouping
overhead, weight bytes/token, and max abs diff (equivalence gate).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlx.core as mx

from deepseeker.grouped import grouped_swiglu, naive_swiglu, plan_groups

HIDDEN = 5120
INTER = 2304
TOP_K = 6


def build(experts: int, seed: int):
    rng = random.Random(seed)
    mx.random.seed(seed)
    weights = {}
    for expert in range(experts):
        gate = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16) * 0.02
        up = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16) * 0.02
        down = mx.random.normal((INTER, HIDDEN), dtype=mx.bfloat16) * 0.02
        weights[expert] = (gate, up, down)
    mx.eval(list(weights.values()))
    return weights, rng


def bench(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Grouped GEMM prototype bench.")
    parser.add_argument("--tokens", type=str, default="8,32")
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trace", type=Path, default=None,
                        help="use real top-k assignments (grouped in blocks of --tokens)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    weights, rng = build(args.experts, 11)
    print(f"experts={args.experts} top_k={TOP_K} shapes={HIDDEN}/{INTER}")
    print("tokens  naive_ms  grouped_ms  align_ms  maxdiff   speedup")
    rows = []
    blocks: list[tuple[int, list[list[int]]]] = []
    if args.trace:
        from deepseeker.trace import expert_records, read_trace

        _header, _records = read_trace(args.trace)
        by_token: dict[int, list[int]] = {}
        for record in expert_records(_records):
            if record["layer"] == 0:
                by_token.setdefault(record["token_pos"], []).extend(record["experts"])
        ordered = [sorted(set(by_token[pos]))[:TOP_K] for pos in sorted(by_token)]
        for tokens in [int(t) for t in args.tokens.split(",") if t.strip()]:
            for start in range(0, len(ordered), tokens):
                chunk = ordered[start:start + tokens]
                if len(chunk) == tokens:
                    blocks.append((tokens, chunk))
    else:
        for tokens in [int(t) for t in args.tokens.split(",") if t.strip()]:
            topk = [sorted(rng.sample(range(args.experts), TOP_K)) for _ in range(tokens)]
            blocks.append((tokens, topk))
    for tokens, topk in blocks:
        router = [[1.0 / TOP_K] * TOP_K for _ in range(tokens)]
        need = sorted({e for row in topk for e in row})
        if args.trace:
            for expert in need:
                if expert not in weights:
                    mx.random.seed(1000 + expert)
                    gate = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16) * 0.02
                    up = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16) * 0.02
                    down = mx.random.normal((INTER, HIDDEN), dtype=mx.bfloat16) * 0.02
                    weights[expert] = (gate, up, down)
            mx.eval(list(weights.values()))
        n_exp = max(need) + 1 if need else args.experts
        x = mx.random.normal((tokens, HIDDEN), dtype=mx.bfloat16)
        mx.eval(x)
        start = time.perf_counter()
        groups = plan_groups(topk, n_exp)
        align_ms = (time.perf_counter() - start) * 1000
        naive_s = bench(lambda x=x, weights=weights, topk=topk, router=router: naive_swiglu(x, weights, topk, router), args.iters)
        grouped_s = bench(lambda x=x, weights=weights, groups=groups, router=router: grouped_swiglu(x, weights, groups, router), args.iters)
        ref = naive_swiglu(x, weights, topk, router)
        got = grouped_swiglu(x, weights, groups, router)
        diff = float(mx.max(mx.abs(ref - got)).item())
        mx.eval(diff)
        distinct = len(groups)
        weight_gib = distinct * 3 * HIDDEN * INTER * 2 / 1024**3
        print(f"{tokens:<6}  {naive_s * 1000:>7.1f}  {grouped_s * 1000:>9.1f}  "
              f"{align_ms:>7.2f}  {diff:.2e}  {naive_s / grouped_s:>6.2f}x")
        rows.append({"tokens": tokens, "experts_used": distinct,
                     "naive_ms": round(naive_s * 1000, 1),
                     "grouped_ms": round(grouped_s * 1000, 1),
                     "align_ms": round(align_ms, 2),
                     "max_abs_diff": diff,
                     "speedup": round(naive_s / grouped_s, 2),
                     "weight_gib": round(weight_gib, 2)})
        if diff > 0.5:
            print("FAIL: equivalence breached")
            return 1
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
