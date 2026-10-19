#!/usr/bin/env python3
"""Activated-expert sweep: what does a narrower route (top_k) buy in speed
and bytes, and what does it cost in output distribution?

top_k controls how many experts each token activates (config default 6).
It cuts route compute and expert bytes roughly linearly, so it is the one
exact-decoding knob that touches both sides of the IO/compute split.

One process, one Runner; every config reuses the loaded weights and the warm
expert cache. The first config is the reference (exact top_k=6 unless told
otherwise) and is re-run at the end as a control, so run-to-run noise in
tok_s is a measured number rather than an assumption.

    python scripts/sweep_topk.py --batch 8 --steps 16 --top-ks 6,4,3,2
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

from bench_batch import DEFAULT_PROMPTS
from encoding.encoding import encode_messages
from sweep_union import compare
from transformers import AutoTokenizer

from deepseeker.baseline import load_json
from deepseeker.mlx_runner import Runner


def main() -> int:
    ap = argparse.ArgumentParser(description="Activated-expert (top_k) sweep.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--expert-cap", type=int, default=256)
    ap.add_argument("--top-ks", type=str, default="6,4,3,2")
    ap.add_argument("--union-budget", type=int, default=0,
                    help="also cap the batch union (0 = exact)")
    ap.add_argument("--quality", action="store_true",
                    help="teacher-force each run on the reference so KL/flip "
                         "are measured on identical inputs (2x the runs)")
    ap.add_argument("--quality-only", action="store_true",
                    help="teacher-force every pass and skip its free run: "
                         "all passes then see identical inputs, so bytes and "
                         "tok_s are directly comparable too")
    ap.add_argument("--repeat-control", action="store_true",
                    help="re-run the first top_k at the end to measure noise")
    ap.add_argument("--record-top", type=int, default=32)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    ks = [int(v) for v in args.top_ks.split(",") if v.strip()]
    if not ks:
        ap.error("--top-ks must not be empty")
    seq = list(ks)
    if args.repeat_control:
        seq.append(ks[0])
    n = args.batch

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
    print(f"sweep batch={n} steps={args.steps} cap={args.expert_cap} "
          f"top_ks={seq} union={args.union_budget} quality={args.quality}",
          flush=True)

    def _print(rec: dict) -> None:
        kl_s = "-" if rec["kl"] is None else f"{rec['kl']:.4f}"
        fl_s = "-" if rec["flip"] is None else f"{rec['flip']:.4f}"
        print(f"top_k={rec['top_k']} pass={rec['pass']} "
              f"coverage={rec['coverage']:.4f} "
              f"bytes={rec['bytes'] / 2**30:7.2f}GiB "
              f"tok_s={rec['tok_s']:.4f} loads={rec['loads']} "
              f"kl={kl_s} flip={fl_s} forced={rec['forced']}", flush=True)

    runner = Runner(model_root, manifest, config,
                    expert_cache_cap=args.expert_cap, max_batch=max(n, 1),
                    union_budget=args.union_budget, top_k=seq[0])
    ref: dict | None = None
    ref_k = seq[0]
    rows = []
    for i, k in enumerate(seq):
        runner._top_k = k
        runner._union_budget = args.union_budget
        t0 = time.perf_counter()
        # record_top must be on for every run: the reference's top_logits are
        # what the teacher-forced runs are compared against, and compare()
        # returns (0, 0, 0) on an empty trace.
        want_forced = ref is not None and (args.quality or args.quality_only)
        free = None if (want_forced and args.quality_only) else             runner.generate_batch(prompts, args.steps, record_top=args.record_top)
        if free is None:
            forced = runner.generate_batch(
                prompts, args.steps, force_tokens=ref["outputs"],
                record_top=args.record_top)
            if not ref["top_logits"]:
                raise SystemExit("reference has no top_logits")
            kl, flip, margin = compare(ref["top_logits"], forced["top_logits"])
            rec = {"top_k": k, "pass": i, "coverage": forced["route_coverage"],
                   "tok_s": forced["tok_s"], "decode_s": forced["decode_s"],
                   "bytes": forced["decode_bytes"],
                   "wall_s": time.perf_counter() - t0,
                   "loads": forced["decode_load_calls"],
                   "kl": kl, "flip": flip, "margin": margin, "forced": True}
            rows.append(rec)
            _print(rec)
            continue
        wall = time.perf_counter() - t0
        rec = {"top_k": k, "pass": i, "coverage": free["route_coverage"],
               "tok_s": free["tok_s"], "decode_s": free["decode_s"],
               "bytes": free["decode_bytes"], "wall_s": wall,
               "loads": free["decode_load_calls"]}
        if ref is None:
            ref = free
            rec.update(flip=0.0, kl=0.0, margin=0.0, forced=False)
        elif want_forced:
            if not ref["top_logits"]:
                raise SystemExit("reference has no top_logits")
            runner._top_k = k
            forced = runner.generate_batch(
                prompts, args.steps, force_tokens=ref["outputs"],
                record_top=args.record_top)
            kl, flip, margin = compare(ref["top_logits"], forced["top_logits"])
            rec.update(kl=kl, flip=flip, margin=margin, forced=True)
        else:
            rec.update(kl=None, flip=None, margin=None, forced=False)
        rows.append(rec)
        _print(rec)
        if i == 0:
            ref_k = k
    runner.shutdown()

    base = next(r for r in rows if r["pass"] == 0)
    for rec in rows:
        rec["bytes_speedup"] = base["bytes"] / rec["bytes"] if rec["bytes"] else 0.0
        rec["tok_speedup"] = rec["tok_s"] / base["tok_s"]
        rec["reference_top_k"] = ref_k
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
