#!/usr/bin/env python3
"""Issue #35: CPU vs GPU placement micros + contention + policy.

    python scripts/bench_partition.py --iters 20 --json-out part.json

Tasks at real shapes: router matvec, top-6 select, engram-style hash
loop, expert GEMV (numpy CPU vs MLX GPU), plus a contention round (CPU
burn alongside GPU GEMMs). Prints measurements then the decided policy.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlx.core as mx
import numpy as np

from deepseeker.partition import decide


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
    parser = argparse.ArgumentParser(description="Partition micros.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    measurements: dict[str, dict[str, float]] = {}

    w_router = rng.normal(0, 0.02, size=(5120, 384)).astype(np.float32)
    x = rng.normal(0, 1, size=(5120,)).astype(np.float32)
    mx_router = mx.array(w_router).astype(mx.bfloat16)
    mx_x = mx.array(x).astype(mx.bfloat16)
    mx.eval(mx_router, mx_x)
    cpu_router = bench(lambda: w_router.T @ x, args.iters)
    gpu_router = bench(lambda: mx.eval(mx_x @ mx_router), args.iters)
    measurements["router-matvec"] = {"cpu": cpu_router, "gpu": gpu_router}

    scores = rng.normal(0, 1, size=(384,)).astype(np.float32)
    mx_scores = mx.array(scores)
    mx.eval(mx_scores)
    cpu_topk = bench(lambda: np.argpartition(scores, -6)[-6:], args.iters)
    gpu_topk = bench(lambda: mx.eval(mx.argpartition(mx_scores, -6)[-6:]), args.iters)
    measurements["topk-select"] = {"cpu": cpu_topk, "gpu": gpu_topk}

    toks = rng.integers(0, 129280, size=128)
    primes = [2, 7, 61]
    cpu_hash = bench(lambda: [pow(int(t), 2, p) for t in toks for p in primes], args.iters)
    measurements["engram-hash"] = {"cpu": cpu_hash}

    w_exp = rng.normal(0, 0.02, size=(5120, 2304)).astype(np.float32)
    mx_exp = mx.array(w_exp).astype(mx.bfloat16)
    mx.eval(mx_exp)
    cpu_gemv = bench(lambda: w_exp.T @ x, args.iters)
    gpu_gemv = bench(lambda: mx.eval(mx_x @ mx_exp), args.iters)
    measurements["expert-gemm"] = {"cpu": cpu_gemv, "gpu": gpu_gemv}

    stop = threading.Event()

    def burn():
        a = rng.normal(0, 1, size=(512, 512))
        while not stop.is_set():
            a = a @ a.T

    thread = threading.Thread(target=burn, daemon=True)
    thread.start()
    try:
        contended = bench(lambda: mx.eval(mx_x @ mx_exp), args.iters)
    finally:
        stop.set()
        thread.join()
    measurements["expert-gemm-contended"] = {"gpu": contended}

    print("task               cpu_ms    gpu_ms")
    for task, times in measurements.items():
        cpu = f"{times['cpu']:>7.3f}" if "cpu" in times else "      -"
        gpu = f"{times['gpu']:>7.3f}" if "gpu" in times else "      -"
        print(f"{task:<18} {cpu}  {gpu}")
    policy = decide(measurements)
    print("policy:")
    for task, rule in policy["policy"].items():
        print(f"  {task:<15} -> {rule['placement']:<7} (fallback {rule['fallback']})")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"measurements": measurements, "policy": policy}, indent=1) + "\n",
            encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
