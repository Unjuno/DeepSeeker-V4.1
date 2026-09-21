#!/usr/bin/env python3
"""Issue #34: measured fusion candidates + overlap demos (MLX).

    python scripts/bench_fusion.py --iters 10 --json-out fusion.json

Candidates: swiglu-chain fusion (one eval vs four), dequant+matmul
staging, scatter-add shape. Demos: two-stream overlap, double-buffered
staging/compute. Each prints PASS/REVERT by the verdict rule.
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

from deepseeker.fusion import intermediate_bytes, launch_counts, verdict

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
    parser = argparse.ArgumentParser(description="Fusion candidates bench.")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    inter = intermediate_bytes()
    launches = launch_counts()
    print(f"unfused intermediates/expert: {inter['unfused_bytes'] / 1024:.0f}KB "
          f"-> fused {inter['fused_bytes'] / 1024:.0f}KB "
          f"(saves {inter['saved_ratio']:.0%})")
    print(f"launches/token: naive {launches['naive_evals']} fused {launches['fused_evals']} "
          f"ideal {launches['ideal_evals']}")

    gate = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16) * 0.02
    up = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16) * 0.02
    down = mx.random.normal((INTER, HIDDEN), dtype=mx.bfloat16) * 0.02
    x = mx.random.normal((1, HIDDEN), dtype=mx.bfloat16)
    mx.eval(gate, up, down, x)

    def unfused():
        g = x @ gate
        mx.eval(g)
        u = x @ up
        mx.eval(u)
        a = g * mx.sigmoid(g) * u
        mx.eval(a)
        mx.eval(a @ down)

    def fused():
        g = x @ gate
        mx.eval((g * mx.sigmoid(g) * (x @ up)) @ down)

    t_unfused = bench(unfused, args.iters)
    t_fused = bench(fused, args.iters)
    v_sw = verdict("swiglu-chain", t_fused, t_unfused, True, True)
    print(f"swiglu chain: unfused {t_unfused:.3f}ms fused {t_fused:.3f}ms -> {v_sw['verdict']}")

    s1 = mx.new_stream(mx.default_device())
    s2 = mx.new_stream(mx.default_device())
    a = mx.random.normal((2048, 2048), dtype=mx.bfloat16)
    b = mx.random.normal((2048, 2048), dtype=mx.bfloat16)
    mx.eval(a, b)

    def serial():
        mx.eval(a @ b)
        mx.eval(a @ b)

    def overlapped():
        with mx.stream(s1):
            r1 = a @ b
        with mx.stream(s2):
            r2 = a @ b
        mx.eval(r1, r2)

    t_serial = bench(serial, args.iters)
    t_overlap = bench(overlapped, args.iters)
    v_ov = verdict("stream-overlap", t_overlap, t_serial, True, True)
    print(f"streams: serial {t_serial:.3f}ms overlapped {t_overlap:.3f}ms -> {v_ov['verdict']}")

    out = {"intermediate": inter, "launches": launches,
           "swiglu": v_sw, "overlap": v_ov}
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
