"""Issue #32: kernel microbenchmark suite at exact DeepSeek shapes (MLX).

Each micro mirrors one hot operator of the decode path at true geometry
(hidden 5120 / inter 2304 / heads 64 / top-6) and reports median/p95.
`relevance()` maps micro timings onto the measured 109.8ms/token MoE
floor to rank specialization targets by end-to-end share.

No custom Metal (no Xcode toolchain on this machine): MLX baselines
only, Metal scaffold explicitly unavailable.
"""

from __future__ import annotations

import time

import mlx.core as mx

HIDDEN = 5120
INTER = 2304
TOP_K = 6
TOKEN_MS_FLOOR = 109.8  # measured MoE floor, bench_decode_floor.py


def _median(times: list[float]) -> float:
    ordered = sorted(times)
    return ordered[len(ordered) // 2]


def _p95(times: list[float]) -> float:
    ordered = sorted(times)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def time_fn(fn, iters: int, warmup: int = 3) -> dict:
    """Time a zero-arg closure that evaluates its own graph."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return {"median_ms": _median(times) * 1000, "p95_ms": _p95(times) * 1000,
            "iters": iters}


def bench_gemm(batch: int, iters: int = 20) -> dict:
    """gate/up projection GEMM [B,5120]@[5120,2304] (bf16)."""
    w = mx.random.normal((HIDDEN, INTER), dtype=mx.bfloat16) * 0.02
    x = mx.random.normal((batch, HIDDEN), dtype=mx.bfloat16)
    mx.eval(w, x)

    def run():
        mx.eval(x @ w)

    out = time_fn(run, iters)
    out.update({"op": f"gemm-gate-b{batch}", "batch": batch,
                "gib": batch * HIDDEN * INTER * 2 / 1024**3})
    return out


def bench_swiglu_elem(tokens: int, iters: int = 20) -> dict:
    """SwiGLU elementwise part on [T,2304] (sigmoid+mul, no GEMM)."""
    g = mx.random.normal((tokens, INTER), dtype=mx.bfloat16)
    u = mx.random.normal((tokens, INTER), dtype=mx.bfloat16)
    mx.eval(g, u)

    def run():
        mx.eval(g * mx.sigmoid(g) * u)

    out = time_fn(run, iters)
    out.update({"op": "swiglu-elem", "tokens": tokens})
    return out


def bench_fp8_dequant(tokens: int, iters: int = 20) -> dict:
    """Simulated fp8 block-dequant: u8 [T,2304] x bf16 scales [T,72].

    Stands in for e4m3+128-wide block scales until P7 owns real fp8
    tensors; measures the dequant arithmetic + bandwidth shape.
    """
    q = mx.random.randint(0, 256, (tokens, INTER)).astype(mx.uint8)
    scales = mx.random.normal((tokens, INTER // 32), dtype=mx.bfloat16) * 0.1
    mx.eval(q, scales)

    def run():
        f = q.astype(mx.bfloat16)
        rep = mx.repeat(scales, 32, axis=1)
        mx.eval(f * rep)

    out = time_fn(run, iters)
    out.update({"op": "fp8-dequant-sim", "tokens": tokens})
    return out


def bench_launch(iters: int = 200) -> dict:
    """Empty-eval overhead: floor of dispatch + sync cost."""
    x = mx.array([1.0], dtype=mx.bfloat16)
    mx.eval(x)

    def run():
        mx.eval(x + 0.0)

    out = time_fn(run, iters)
    out.update({"op": "launch-overhead"})
    return out


def relevance(rows: list[dict], per_token_count: dict[str, float]) -> list[dict]:
    """Share of the 109.8ms token each op class owns at given counts.

    Standalone medians include their own eval+sync, so shares are UPPER
    bounds that do not sum to 100 under lazy fusion (the fused floor is
    the truth; this ranks targets, it does not decompose the floor).
    Launch count assumes per-layer-ish evals, not per-kernel.
    """
    ranked = []
    for row in rows:
        count = per_token_count.get(row["op"], 0.0)
        share_ms = row["median_ms"] * count
        ranked.append({**row, "per_token": count,
                       "share_ms": round(share_ms, 1),
                       "share_pct": round(100 * share_ms / TOKEN_MS_FLOOR, 1)})
    return sorted(ranked, key=lambda r: -r["share_ms"])
