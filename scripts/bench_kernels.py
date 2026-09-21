#!/usr/bin/env python3
"""Issue #32: run the kernel microbench suite + relevance ranking.

    python scripts/bench_kernels.py --iters 20 --json-out kernels.json

Counts per token: gate/up GEMM 480x, SwiGLU-elem 240x, dequant 240x,
launches ~1500x. Ranking shows where P7 specialization can matter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlx.core as mx

from deepseeker.kernels import (
    bench_fp8_dequant,
    bench_gemm,
    bench_launch,
    bench_swiglu_elem,
    relevance,
)

PER_TOKEN = {"gemm-gate-b1": 480.0, "gemm-gate-b8": 60.0, "swiglu-elem": 240.0,
             "fp8-dequant-sim": 240.0, "launch-overhead": 120.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernel microbench suite.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    print(f"device: {mx.default_device()}", flush=True)
    rows = [
        bench_gemm(1, args.iters),
        bench_gemm(8, args.iters),
        bench_swiglu_elem(6, args.iters),
        bench_fp8_dequant(6, args.iters),
        bench_launch(),
    ]
    print("op               median_ms  p95_ms")
    for row in rows:
        print(f"{row['op']:<16} {row['median_ms']:>9.3f}  {row['p95_ms']:>6.3f}")
    print("relevance vs 109.8ms token (upper bounds; fused floor is truth):")
    ranked = relevance(rows, PER_TOKEN)
    for row in ranked:
        print(f"  {row['op']:<16} {row['share_ms']:>7.1f}ms  ({row['share_pct']:>5.1f}%)")
    if args.json_out:
        args.json_out.write_text(json.dumps(ranked, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
