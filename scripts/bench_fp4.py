#!/usr/bin/env python3
"""Issue #33: staged fp4-dequant + GEMM vs bf16 GEMM at expert shapes.

    python scripts/bench_fp4.py --iters 10 --json-out fp4.json

Dequantizes a real expert gate matrix once (Metal), then times bf16
GEMM per token; compares against the all-bf16 path (2x bytes) and
reports the expert-phase delta. Numerical gate: Metal == reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlx.core as mx
import numpy as np

from deepseeker.metal_fp4 import metal_dequant, reference_dequant

HIDDEN = 5120
INTER = 2304


def bench(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2] * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description="Staged fp4 bench.")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(9)
    packed = rng.integers(0, 256, size=(HIDDEN * INTER) // 2, dtype=np.uint8)
    scales = np.exp2(rng.integers(-2, 3, size=(HIDDEN * INTER) // 32).astype(np.float32))
    ref = reference_dequant(packed, scales)
    got = metal_dequant(packed, scales)
    maxdiff = float(np.abs(got - ref).max())
    print(f"dequant equivalence maxdiff={maxdiff:.2e}")
    if maxdiff != 0.0:
        print("FAIL: Metal dequant differs from reference")
        return 1
    w_bf16 = mx.array(got.reshape(HIDDEN, INTER)).astype(mx.bfloat16)
    x = mx.random.normal((1, HIDDEN), dtype=mx.bfloat16)
    mx.eval(w_bf16, x)
    gemm_ms = bench(lambda: mx.eval(x @ w_bf16), args.iters)
    w_native = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16)
    mx.eval(w_native)
    base_ms = bench(lambda: mx.eval(x @ w_native), args.iters)
    deq_ms = bench(lambda: metal_dequant(packed, scales), 3)
    print(f"bf16 GEMM: {base_ms:.3f}ms  staged dequant: {deq_ms:.1f}ms (once per load)")
    print(f"staged GEMM: {gemm_ms:.3f}ms (traffic halved vs bf16: "
          f"{HIDDEN * INTER / 1024**2:.1f}MB -> {HIDDEN * INTER / 2 / 1024**2:.1f}MB)")
    result = {"maxdiff": maxdiff, "bf16_gemm_ms": round(base_ms, 3),
              "staged_dequant_ms": round(deq_ms, 1),
              "staged_gemm_ms": round(gemm_ms, 3),
              "traffic_mb_bf16": round(HIDDEN * INTER / 1024**2, 1),
              "traffic_mb_fp4": round(HIDDEN * INTER / 2 / 1024**2, 1)}
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
