#!/usr/bin/env python3
"""TPS maximisation sweep: every cheap knob in one process.

top_k (activated experts), expert cache cap, replacement policy and batch size
are the knobs that do not change the maths of a layer. Each combination gets a
cold expert cache so hit-rate numbers are comparable, and every run reports
tok_s, bytes, loads and hits. Load/state is shared across the whole matrix.

    python scripts/sweep_tps.py --batch 8 --steps 16 \
        --top-ks 1,2 --caps 256,512,1024 --policies lru,lfu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mlx.core as mx
from bench_batch import DEFAULT_PROMPTS
from encoding.encoding import encode_messages
from transformers import AutoTokenizer

from deepseeker.baseline import load_json
from deepseeker.cache_policy import expert_lru_budget
from deepseeker.mlx_runner import Runner


def main() -> int:
    ap = argparse.ArgumentParser(description="TPS knob sweep.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=str, default="",
                    help="batch sizes to sweep in one process, e.g. 4,8,16 "
                         "(default: just --batch)")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeats each config, interleaved (A,B,A,B) so "
                         "system-load drift cancels between configs")
    ap.add_argument("--top-ks", type=str, default="1,2")
    ap.add_argument("--caps", type=str, default="256,512,1024")
    ap.add_argument("--policies", type=str, default="lru")
    ap.add_argument("--shared", type=str, default="on",
                    help="on / off / on,off -- whether the shared expert runs")
    ap.add_argument("--hc-iters", type=str, default="20",
                    help="Sinkhorn iteration counts to sweep, e.g. 20,8,4")
    ap.add_argument("--windows", type=str, default="128",
                    help="attention window sizes to sweep, e.g. 128,64,32")
    ap.add_argument("--attn", type=str, default="fp32",
                    help="fp32 / bf16 / fp32,bf16 -- sparse attention core dtype")
    ap.add_argument("--skip-every", type=str, default="0",
                    help="MoE bypass periods to sweep, e.g. 0,2,3,4")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    ks = [int(v) for v in args.top_ks.split(",") if v.strip()]
    caps = [int(v) for v in args.caps.split(",") if v.strip()]
    pols = [v.strip() for v in args.policies.split(",") if v.strip()]
    shared = [v.strip() for v in args.shared.split(",") if v.strip()]
    hcs = [int(v) for v in args.hc_iters.split(",") if v.strip()]
    wins = [int(v) for v in args.windows.split(",") if v.strip()]
    atns = [v.strip() for v in args.attn.split(",") if v.strip()]
    skips = [int(v) for v in args.skip_every.split(",") if v.strip()]
    batches = ([int(v) for v in args.batches.split(",") if v.strip()]
               if args.batches else [args.batch])
    n = max(batches)

    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json")
    config = load_json(snapshot / "inference" / "config.json")
    model_root = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    tokenizer = AutoTokenizer.from_pretrained(str(model_root))

    texts = (DEFAULT_PROMPTS * ((n // len(DEFAULT_PROMPTS)) + 1))[:n]
    toks = [tokenizer.encode(encode_messages([{"role": "user", "content": t}], "chat"))
            for t in texts]
    common = min(len(t) for t in toks)
    prompts = [t[:common] for t in toks]

    configs = [(b, k, c, p, sh, hc, w, a, sk) for b in batches
               for k in ks for c in caps for p in pols
               for sh in shared for hc in hcs for w in wins for a in atns
               for sk in skips]
    # repeat outside, configs inside: A,B,C,A,B,C -- load drift hits all alike.
    matrix = [cfg for _ in range(max(1, args.repeat)) for cfg in configs]
    print(f"sweep batches={batches} steps={args.steps} runs={len(matrix)} "
          f"ks={ks} caps={caps} policies={pols} shared={shared} "
          f"hc={hcs} windows={wins}", flush=True)

    runner = Runner(model_root, manifest, config, expert_cache_cap=caps[0],
                    max_batch=max(n, 1), top_k=ks[0],
                    hc_iters=hcs[0], window=wins[0])
    rows = []
    for rep, (b, k, cap, pol, sh, hc, w, a, sk) in enumerate(matrix):
        effective_cap = expert_lru_budget(cap, has_static_pins=False)
        runner._top_k = k
        runner._expert_cache_cap = effective_cap
        # generate_batch() switches between the prefill and decode cache caps.
        # Update all three fields or _use_decode_cache() silently restores the
        # constructor's original cap, making cap sweeps no-ops.
        runner._expert_decode_cache_cap = effective_cap
        runner._expert_prefill_cache_cap = effective_cap
        runner._expert_policy = pol
        runner._no_shared = (sh == "off")
        runner._hc_iters = int(hc)
        runner._window_size = int(w)
        runner._attn_bf16 = (a == "bf16")
        runner._skip_every = int(sk)
        runner._expert_cache.clear()
        runner._expert_used_now.clear()
        runner._expert_protected.clear()
        runner._expert_freq.clear()
        # A big previous cap bloats the Metal free list, which starves the file
        # page cache and poisons every later config -- drop it before each run.
        mx.clear_cache()
        ev0 = runner.expert_evictions
        t0 = time.perf_counter()
        res = runner.generate_batch(prompts[:b], args.steps)
        wall = time.perf_counter() - t0
        signature = (b, k, pol, sh, hc, w, a, sk)
        prior = next((old for old in rows
                      if tuple(old["math_signature"]) == signature), None)
        rec = {"batch": b, "top_k": k, "cap": cap,
               "effective_decode_cap": runner._expert_decode_cache_cap,
               "effective_prefill_cap": runner._expert_prefill_cache_cap,
               "effective_cache_cap": runner._expert_cache_cap,
               "policy": pol,
               "shared": sh, "hc_iters": hc, "window": w, "attn": a,
               "skip_every": sk, "rep": rep,
               "tok_s": res["tok_s"], "decode_s": res["decode_s"],
               "prefill_s": res["prefill_s"], "wall_s": wall,
               "bytes": res["decode_bytes"],
               "loads": res["decode_load_calls"],
               "expert_loads": res["expert_loads"],
               "hits": res["expert_cache_hits"],
               "io_s": res["decode_io_s"],
               "outputs": res["outputs"],
               "math_signature": list(signature),
               "matches_prior_same_math_outputs": (
                   None if prior is None else res["outputs"] == prior["outputs"]),
               "evictions": runner.expert_evictions - ev0}
        rows.append(rec)
        print(f"B={b:2d} k={k} cap={cap:5d} {pol:3s} sh={sh:3s} "
              f"hc={hc:3d} w={w:3d} a={a:4s} sk={sk} rep={rep} "
              f"tok_s={rec['tok_s']:6.3f} decode={rec['decode_s']:6.2f}s "
              f"bytes={rec['bytes'] / 2**30:7.2f}GiB io={rec['io_s']:5.2f}s "
              f"loads={rec['loads']:5d} hits={rec['hits']:5d} "
              f"evict={rec['evictions']:5d}", flush=True)
    runner.shutdown()

    best = max(rows, key=lambda r: r["tok_s"])
    print(f"BEST B={best['batch']} k={best['top_k']} cap={best['cap']} "
          f"{best['policy']} shared={best['shared']} hc={best['hc_iters']} "
          f"window={best['window']} attn={best['attn']} "
          f"skip={best['skip_every']} tok_s={best['tok_s']:.3f}", flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
