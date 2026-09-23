#!/usr/bin/env python3
"""Finer decode profile: MoE fetch vs GEMM; head; residual breakdown."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream" / "DeepSeek-V4.1-Flash"))

import mlx.core as mx
from encoding.encoding import encode_messages
from transformers import AutoTokenizer

from deepseeker import mlx_forward as F
from deepseeker.baseline import load_json
from deepseeker.mlx_runner import Runner

GOLDEN = [43, 4571, 2887, 304, 2507, 1613, 539, 43]


def _reset(stats: dict) -> None:
    for k, v in stats.items():
        stats[k] = 0.0 if isinstance(v, float) else 0


def main() -> int:
    snap = REPO / "upstream" / "DeepSeek-V4.1-Flash"
    rev = load_json(snap / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(REPO / "profiles" / "deepseek-v4.1-flash" / rev / "model-manifest.json")
    config = load_json(snap / "inference" / "config.json")
    root = REPO / "models" / "DeepSeek-V4.1-Flash"
    tok = AutoTokenizer.from_pretrained(str(root))
    ids = tok.encode(encode_messages([{"role": "user", "content": "Say hello."}], "chat"))

    r = Runner(root, manifest, config, expert_cache_cap=256)
    orig_moe_layer = F.moe_layer
    try:
        base = r.generate(ids, 8, temperature=0.0)
        assert base["tokens"] == GOLDEN, base["tokens"]
        print(f"BASE ok tok/s={base['timings']['tok_s']:.4f}", flush=True)

        stats: dict = {
            "route": 0.0, "fetch": 0.0, "gemm": 0.0, "shared": 0.0,
            "n_fetch": 0, "n_miss": 0, "n_moe": 0,
        }

        def moe_layer_clean(x, gate_w, gate_b, expert_fn, shared_fn, top_k=6,
                            limit=10.0, trace_hook=None, layer_id=0, prefetch_fn=None):
            stats["n_moe"] += 1
            t0 = time.perf_counter()
            scores = F.sqrtsoftplus_scores(x, gate_w)
            weights, indices = F.route_topk(scores, gate_b, top_k)
            mx.eval(scores, weights, indices)
            stats["route"] += time.perf_counter() - t0

            out_arr = mx.zeros_like(x, dtype=mx.float32)
            idx_np = np.asarray(indices, dtype=np.int64)
            w_np = np.asarray(weights, dtype=np.float32)
            unique = np.unique(idx_np)
            if prefetch_fn is not None:
                prefetch_fn(layer_id, [int(e) for e in unique])

            for expert in unique:
                mask = idx_np == expert
                rows = np.where(mask.any(axis=1))[0]
                cols = [int(np.where(mask[r])[0][0]) for r in rows]
                tf = time.perf_counter()
                w1, w3, w2down = expert_fn(int(expert))
                mx.eval(w1, w3, w2down)
                stats["fetch"] += time.perf_counter() - tf
                stats["n_fetch"] += 1
                tg = time.perf_counter()
                xr = x[mx.array(rows)]
                g = xr.astype(mx.float32) @ w1.astype(mx.float32).T
                u = xr.astype(mx.float32) @ w3.astype(mx.float32).T
                g = mx.minimum(g, limit)
                u = mx.clip(u, -limit, limit)
                h = (g / (1.0 + mx.exp(-g))) * u
                rw = mx.array(w_np[rows, cols], dtype=mx.float32).reshape(-1, 1)
                contrib = (h * rw) @ w2down.astype(mx.float32).T
                out_arr = out_arr.at[mx.array(rows)].add(contrib)
                mx.eval(contrib)
                stats["gemm"] += time.perf_counter() - tg
                del xr, g, u, h, rw, contrib
            if trace_hook is not None:
                trace_hook(layer_id, idx_np.tolist(), w_np.tolist())
            t2 = time.perf_counter()
            shared = shared_fn(x)
            mx.eval(shared)
            stats["shared"] += time.perf_counter() - t2
            out_arr = out_arr + shared.astype(mx.float32)
            return out_arr.astype(x.dtype)

        F.moe_layer = moe_layer_clean

        att_stats = {"rest": 0.0, "n": 0}
        orig_att = r.attention

        def att_timed(layer, x, start_pos):
            t0 = time.perf_counter()
            out = orig_att(layer, x, start_pos)
            mx.eval(out)
            att_stats["rest"] += time.perf_counter() - t0
            att_stats["n"] += 1
            return out

        r.attention = att_timed

        head_stats = {"t": 0.0, "n": 0}
        orig_head = r.head_logits

        def head_timed(h):
            t0 = time.perf_counter()
            out = orig_head(h)
            mx.eval(out)
            head_stats["t"] += time.perf_counter() - t0
            head_stats["n"] += 1
            return out

        r.head_logits = head_timed

        r._window.clear()
        r._compress.clear()
        r._last_route.clear()
        _reset(stats)
        att_stats.update(rest=0.0, n=0)
        head_stats.update(t=0.0, n=0)

        t0 = time.perf_counter()
        logits = r.forward(ids, 0)
        mx.eval(logits)
        prefill_s = time.perf_counter() - t0
        _reset(stats)
        att_stats.update(rest=0.0, n=0)
        head_stats.update(t=0.0, n=0)

        t0 = time.perf_counter()
        toks = []
        for i in range(8):
            nxt = int(np.argmax(np.asarray(logits[-1], dtype=np.float32)))
            toks.append(nxt)
            logits = r.forward([nxt], len(ids) + i)
            mx.eval(logits)
        decode_s = time.perf_counter() - t0
        print(f"DECODE toks={toks} match={toks==GOLDEN} "
              f"prefill={prefill_s:.1f}s decode={decode_s:.1f}s "
              f"tok/s={8/decode_s:.4f}", flush=True)
        print(f"MOE route={stats['route']*1e3:.0f}ms fetch={stats['fetch']*1e3:.0f}ms "
              f"gemm={stats['gemm']*1e3:.0f}ms shared={stats['shared']*1e3:.0f}ms "
              f"n_moe={stats['n_moe']} n_fetch={stats['n_fetch']}", flush=True)
        print(f"ATT total={att_stats['rest']*1e3:.0f}ms n={att_stats['n']} "
              f"HEAD={head_stats['t']*1e3:.0f}ms n={head_stats['n']}", flush=True)
        accounted = (stats["route"] + stats["fetch"] + stats["gemm"] + stats["shared"]
                     + att_stats["rest"] + head_stats["t"])
        print(f"RESIDUAL={(decode_s - accounted)*1e3:.0f}ms "
              f"accounted={accounted*1e3:.0f}ms decode={decode_s*1e3:.0f}ms", flush=True)
        out = {
            "decode_s": decode_s, "prefill_s": prefill_s, "tokens": toks,
            "match": toks == GOLDEN,
            "moe_ms": {k: stats[k] * 1e3 for k in stats},
            "att_ms": att_stats["rest"] * 1e3,
            "head_ms": head_stats["t"] * 1e3,
            "residual_ms": (decode_s - accounted) * 1e3,
        }
        p = REPO / "profiles" / "runs" / "mtp" / "fine_profile_gpu.json"
        p.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {p}", flush=True)
    finally:
        F.moe_layer = orig_moe_layer
        r.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
