#!/usr/bin/env python3
"""Rank budgeted text under the exact model.

Free-runs each union budget, then replays those tokens through an exact run
(teacher forcing, budget 0) and reports where the exact model itself ranks each
token it is being shown. That is the honest quality question: not "did the
greedy path diverge" (it always does) but "would the reference model have said
this anyway".

    python scripts/quality_union.py --batch 8 --steps 16 --budgets 24,16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench_batch import DEFAULT_PROMPTS
from encoding.encoding import encode_messages
from transformers import AutoTokenizer

from deepseeker.baseline import load_json
from deepseeker.mlx_runner import Runner


def ranks_of(tokens: list[list[int]],
             tops: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Exact-model rank of each replayed token (1 = would have been chosen).

    tops[s] is the distribution *after* consuming token s, so it scores
    tokens[s + 1]; the final prediction has no successor to score.
    """
    out = []
    for step in range(len(tops) - 1):
        part, _ = tops[step]
        for row, seq in enumerate(tokens):
            want = seq[step + 1]
            hit = np.where(part[row] == want)[0]
            out.append(int(hit[0]) + 1 if hit.size else part.shape[1] + 1)
    return np.asarray(out, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Union-budget quality: rank budgeted text under the exact model.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--expert-cap", type=int, default=256)
    ap.add_argument("--budgets", type=str, default="24,16,12,8")
    ap.add_argument("--variants", type=str, default="",
                    help="speed knobs to score against the exact model, e.g. "
                         "\"k=1,hc=4,win=64,k=1;hc=4\" (knobs inside a "
                         "variant are ';' separated, variants by ',')")
    ap.add_argument("--strict", action="store_true",
                    help="hard-cap the union (mirrors --union-strict)")
    ap.add_argument("--record-top", type=int, default=64)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    budgets = [int(v) for v in args.budgets.split(",") if v.strip()] if not args.variants else []
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    n = args.batch
    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json")
    config = load_json(snapshot / "inference" / "config.json")
    model_root = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash-1017"
    tokenizer = AutoTokenizer.from_pretrained(str(model_root))

    texts = (DEFAULT_PROMPTS * ((n // len(DEFAULT_PROMPTS)) + 1))[:n]
    toks = [tokenizer.encode(encode_messages([{"role": "user", "content": t}], "chat"))
            for t in texts]
    common = min(len(t) for t in toks)
    prompts = [t[:common] for t in toks]
    print(f"quality batch={n} steps={args.steps} budgets={budgets} "
          f"variants={variants}", flush=True)

    runner = Runner(model_root, manifest, config,
                    expert_cache_cap=args.expert_cap, max_batch=max(n, 1))

    def apply(d: dict) -> None:
        """Pin the exact knobs, then apply whatever a variant overrides."""
        runner._union_budget = int(d.get("budget", 0))
        runner._union_strict = bool(args.strict) and runner._union_budget > 0
        runner._top_k = int(d.get("k", 6))
        runner._hc_iters = int(d.get("hc", 0))
        runner._window_size = int(d.get("win", 0))
        runner._no_shared = d.get("shared") == "off"

    def knobs(d: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in d.items()) or "exact"

    jobs: list[tuple[str, dict]] = (
        [("budget", {"budget": b}) for b in budgets]
        + [("variant", dict(kv.split("=", 1) for kv in v.split(";") if "=" in kv))
           for v in variants])

    # 1. exact free run: the reference text (and its own ranks, all 1)
    apply({})
    exact = runner.generate_batch(prompts, args.steps, record_top=args.record_top)
    print(f"exact coverage={exact['route_coverage']:.4f} "
          f"bytes={exact['decode_bytes'] / 2**30:.2f}GiB "
          f"tok_s={exact['tok_s']:.4f}", flush=True)
    # self-check: the reference replaying its own text must rank it first
    r0 = ranks_of(exact["outputs"], exact["top_logits"])
    print(f"self-check rank max={r0.max()} top1={float((r0 == 1).mean()):.3f}",
          flush=True)
    if r0.max() != 1:
        raise SystemExit("rank plumbing is off; refusing to report")

    rows = []
    for kind, d in jobs:
        apply(d)
        t0 = time.perf_counter()
        free = runner.generate_batch(prompts, args.steps)
        # 2. replay that text through the exact model
        apply({})
        scored = runner.generate_batch(prompts, args.steps,
                                       force_tokens=free["outputs"],
                                       record_top=args.record_top)
        r = ranks_of(free["outputs"], scored["top_logits"])
        rec = {"kind": kind, "knobs": knobs(d),
               "budget": d.get("budget"),
               "coverage": free["route_coverage"],
               "bytes": free["decode_bytes"], "tok_s": free["tok_s"],
               "rank_mean": float(r.mean()), "rank_p50": int(np.median(r)),
               "top1": float((r == 1).mean()), "top5": float((r <= 5).mean()),
               "top32": float((r <= 32).mean()),
               "wall_s": time.perf_counter() - t0}
        rows.append(rec)
        print(f"{rec['knobs'][:34]:34s} coverage={rec['coverage']:.4f} "
              f"bytes={rec['bytes'] / 2**30:7.2f}GiB tok_s={rec['tok_s']:.4f} "
              f"rank_mean={rec['rank_mean']:6.1f} p50={rec['rank_p50']:3d} "
              f"top1={rec['top1']:.3f} top5={rec['top5']:.3f} "
              f"top32={rec['top32']:.3f}", flush=True)
    runner.shutdown()

    base_bytes = exact["decode_bytes"]
    for rec in rows:
        rec["bytes_speedup"] = base_bytes / rec["bytes"] if rec["bytes"] else 0.0
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
