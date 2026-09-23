#!/usr/bin/env python3
"""One-shot decode phase profile + bf16 MoE A/B (lossless check)."""
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


def phase_wrap(r: Runner, times: dict) -> None:
    orig_att = r.attention
    orig_moe = r.moe
    orig_hc = F.hc_mixes
    orig_head = r.head_logits

    def att(*a, **k):
        t0 = time.perf_counter()
        out = orig_att(*a, **k)
        mx.eval(out)
        times["att"] += time.perf_counter() - t0
        times["n_att"] += 1
        return out

    def moe(*a, **k):
        t0 = time.perf_counter()
        out = orig_moe(*a, **k)
        mx.eval(out)
        times["moe"] += time.perf_counter() - t0
        times["n_moe"] += 1
        return out

    def hc(*a, **k):
        t0 = time.perf_counter()
        out = orig_hc(*a, **k)
        mx.eval(out)
        times["hc"] += time.perf_counter() - t0
        times["n_hc"] += 1
        return out

    def head(*a, **k):
        t0 = time.perf_counter()
        out = orig_head(*a, **k)
        mx.eval(out)
        times["head"] += time.perf_counter() - t0
        times["n_head"] += 1
        return out

    r.attention = att
    r.moe = moe
    r.head_logits = head
    F.hc_mixes = hc
    # restore later
    r._orig = (orig_att, orig_moe, orig_hc, orig_head)


def restore(r: Runner) -> None:
    o = getattr(r, "_orig", None)
    if not o:
        return
    orig_att, orig_moe, orig_hc, orig_head = o
    r.attention = orig_att
    r.moe = orig_moe
    r.head_logits = orig_head
    F.hc_mixes = orig_hc


def bf16_moe() -> None:
    """Patch moe_layer GEMMs to bf16; keep clamps/adds in f32."""
    import deepseeker.mlx_forward as mf

    orig = mf.moe_layer

    def moe_bf16(x, gate_w, gate_b, expert_fn, shared_fn, top_k=6, limit=10.0,
                 trace_hook=None, layer_id=0, prefetch_fn=None):
        scores = mf.sqrtsoftplus_scores(x, gate_w)
        weights, indices = mf.route_topk(scores, gate_b, top_k)
        out = mx.zeros_like(x, dtype=mx.float32)
        idx_np = np.asarray(indices, dtype=np.int64)
        w_np = np.asarray(weights, dtype=np.float32)
        unique = np.unique(idx_np)
        if prefetch_fn is not None:
            prefetch_fn(layer_id, [int(e) for e in unique])
        xb = x.astype(mx.bfloat16)
        for expert in unique:
            mask = idx_np == expert
            rows = np.where(mask.any(axis=1))[0]
            cols = [int(np.where(mask[r])[0][0]) for r in rows]
            w1, w3, w2down = expert_fn(int(expert))
            xr = xb[mx.array(rows)]
            g = (xr @ w1.T).astype(mx.float32)
            u = (xr @ w3.T).astype(mx.float32)
            g = mx.minimum(g, limit)
            u = mx.clip(u, -limit, limit)
            h = (g / (1.0 + mx.exp(-g))) * u
            rw = mx.array(w_np[rows, cols], dtype=mx.float32).reshape(-1, 1)
            hbf = h.astype(mx.bfloat16)
            out = out.at[mx.array(rows)].add((hbf * rw.astype(mx.bfloat16)) @ w2down.T)
            del xr, g, u, h, rw, hbf
        if trace_hook is not None:
            trace_hook(layer_id, idx_np.tolist(), w_np.tolist())
        out = out + shared_fn(x).astype(mx.float32)
        return out.astype(x.dtype)

    mf.moe_layer = moe_bf16
    # Runner.moe may call F.moe_layer - check binding
    return orig


def main() -> int:
    snap = REPO / "upstream" / "DeepSeek-V4.1-Flash"
    rev = load_json(snap / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(REPO / "profiles" / "deepseek-v4.1-flash" / rev / "model-manifest.json")
    config = load_json(snap / "inference" / "config.json")
    root = REPO / "models" / "DeepSeek-V4.1-Flash"
    tok = AutoTokenizer.from_pretrained(str(root))
    ids = tok.encode(encode_messages([{"role": "user", "content": "Say hello."}], "chat"))
    print(f"prompt tokens={len(ids)}", flush=True)

    r = Runner(root, manifest, config, expert_cache_cap=256)
    result: dict = {}
    try:
        # baseline golden (f32 path) — generate 8 tokens
        t0 = time.perf_counter()
        base = r.generate(ids, 8, temperature=0.0)
        wall = time.perf_counter() - t0
        toks = base["tokens"]
        print(f"BASE tokens={toks} wall={wall:.1f}s "
              f"prefill={base['timings']['prefill_s']:.1f} "
              f"decode={base['timings']['decode_s']:.1f} "
              f"tok/s={base['timings']['tok_s']:.4f}", flush=True)
        result["base"] = {
            "tokens": toks,
            "match_golden": toks == GOLDEN,
            "timings": base["timings"],
        }
        mx.eval(mx.array([0]))

        # phase profile: prefill once fresh is expensive; reuse runner
        # Reset by reconstructing is expensive. Profile decode-only after warm.
        # New sequence start_pos=0 does full prefill again — still do it for clean phases.
        times = {"att": 0.0, "moe": 0.0, "hc": 0.0, "head": 0.0,
                 "n_att": 0, "n_moe": 0, "n_hc": 0, "n_head": 0}
        phase_wrap(r, times)
        # clear caches states
        r._window.clear()
        r._compress.clear()
        r._last_route.clear()
        t0 = time.perf_counter()
        logits = r.forward(ids, 0)
        mx.eval(logits)
        prefill_dt = time.perf_counter() - t0
        prefill_phases = dict(times)
        for k, v in times.items():
            times[k] = 0.0 if isinstance(v, float) else 0

        # decode 8 steps with phases
        decode_phases = {"att": 0.0, "moe": 0.0, "hc": 0.0, "head": 0.0}
        t0 = time.perf_counter()
        out_toks = []
        for i in range(8):
            nxt = int(np.argmax(np.asarray(logits[-1], dtype=np.float32)))
            out_toks.append(nxt)
            logits = r.forward([nxt], len(ids) + i)
            mx.eval(logits)
        decode_dt = time.perf_counter() - t0
        decode_phases = {k: times[k] for k in ("att", "moe", "hc", "head")}
        restore(r)
        print(f"PHASE prefill={prefill_dt:.1f}s phases={prefill_phases}", flush=True)
        print(f"PHASE decode={decode_dt:.3f}s toks={out_toks} "
              f"phases_ms={{{', '.join(f'{k}:{v*1e3:.1f}' for k,v in decode_phases.items())}}} "
              f"per_tok_ms={decode_dt/8*1e3:.1f}", flush=True)
        residual = decode_dt - sum(decode_phases.values())
        print(f"PHASE residual_ms={residual*1e3:.1f} "
              f"(n_att={times['n_att']} n_moe={times['n_moe']} n_hc={times['n_hc']} "
              f"n_head={times['n_head']})", flush=True)
        result["phase"] = {
            "prefill_s": prefill_dt,
            "decode_s": decode_dt,
            "decode_phases_ms": {k: v * 1e3 for k, v in decode_phases.items()},
            "residual_ms": residual * 1e3,
            "tokens": out_toks,
            "match_golden": out_toks == GOLDEN,
        }

        # bf16 MoE A/B: generate 8 tokens with patched moe
        # Find how Runner.moe calls F.moe_layer
        import inspect
        src = inspect.getsource(r.moe)
        print("MOE_SRC_HAS_F", "F.moe_layer" in src or "mlx_forward" in src, flush=True)

        # Check binding: Runner.moe may import F.moe_layer at module level
        # Patch both deepseeker.mlx_forward.moe_layer and whatever Runner uses
        orig_moe_layer = F.moe_layer
        def moe_bf16(x, gate_w, gate_b, expert_fn, shared_fn, top_k=6, limit=10.0,
                     trace_hook=None, layer_id=0, prefetch_fn=None):
            scores = F.sqrtsoftplus_scores(x, gate_w)
            weights, indices = F.route_topk(scores, gate_b, top_k)
            out = mx.zeros_like(x, dtype=mx.float32)
            idx_np = np.asarray(indices, dtype=np.int64)
            w_np = np.asarray(weights, dtype=np.float32)
            unique = np.unique(idx_np)
            if prefetch_fn is not None:
                prefetch_fn(layer_id, [int(e) for e in unique])
            xb = x.astype(mx.bfloat16)
            for expert in unique:
                mask = idx_np == expert
                rows = np.where(mask.any(axis=1))[0]
                cols = [int(np.where(mask[r])[0][0]) for r in rows]
                w1, w3, w2down = expert_fn(int(expert))
                xr = xb[mx.array(rows)]
                g = (xr @ w1.T).astype(mx.float32)
                u = (xr @ w3.T).astype(mx.float32)
                g = mx.minimum(g, limit)
                u = mx.clip(u, -limit, limit)
                h = (g / (1.0 + mx.exp(-g))) * u
                rw = mx.array(w_np[rows, cols], dtype=mx.float32).reshape(-1, 1)
                # accumulate in f32 but GEMM in bf16
                contrib = ((h.astype(mx.bfloat16) * rw.astype(mx.bfloat16))
                           @ w2down.T).astype(mx.float32)
                out = out.at[mx.array(rows)].add(contrib)
                del xr, g, u, h, rw, contrib
            if trace_hook is not None:
                trace_hook(layer_id, idx_np.tolist(), w_np.tolist())
            # shared expert also bf16 GEMM for fairness? keep shared as-is
            out = out + shared_fn(x).astype(mx.float32)
            return out.astype(x.dtype)

        # Runner.moe likely does F.moe_layer(...) — patch module attr
        F.moe_layer = moe_bf16

        # Also check if mlx_runner imported moe_layer by name
        import deepseeker.mlx_runner as R
        if getattr(R, "moe_layer", None) is not None:
            R.moe_layer = moe_bf16
            print("PATCHED R.moe_layer", flush=True)

        r._window.clear()
        r._compress.clear()
        r._last_route.clear()
        t0 = time.perf_counter()
        bf = r.generate(ids, 8, temperature=0.0)
        wall_bf = time.perf_counter() - t0
        bf_toks = bf["tokens"]
        print(f"BF16 tokens={bf_toks} match={bf_toks==GOLDEN} wall={wall_bf:.1f}s "
              f"prefill={bf['timings']['prefill_s']:.1f} "
              f"decode={bf['timings']['decode_s']:.1f} "
              f"tok/s={bf['timings']['tok_s']:.4f}", flush=True)
        result["bf16"] = {
            "tokens": bf_toks,
            "match_golden": bf_toks == GOLDEN,
            "timings": bf["timings"],
        }
        F.moe_layer = orig_moe_layer
        if getattr(R, "moe_layer", None) is moe_bf16:
            delattr(R, "moe_layer")

        outp = REPO / "profiles" / "runs" / "mtp" / "phase_profile.json"
        outp.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {outp}", flush=True)
    finally:
        r.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
