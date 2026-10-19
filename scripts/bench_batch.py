#!/usr/bin/env python3
"""Lockstep batched decode throughput on M1 Max.

Prefills N equal-length sequences one at a time, then decodes them together:
every per-layer sync, expert prefetch and attention build is amortised across
the whole batch. Reports aggregate tok/s (the number that matters when more
than one request is in flight) alongside per-sequence tok/s.

    python scripts/bench_batch.py --batch 8 --max-tokens 24
    python scripts/bench_batch.py --batch 1 --max-tokens 24   # golden sanity
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

from encoding.encoding import encode_messages
from transformers import AutoTokenizer

from deepseeker import mlx_forward as F
from deepseeker.baseline import load_json
from deepseeker.mlx_runner import Runner

DEFAULT_PROMPTS = [
    "Say hello.",
    "Summarise the water cycle.",
    "Write a haiku about rain.",
    "List three sorting algorithms.",
    "Explain what a mutex is.",
    "Translate 'good morning' to French.",
    "What is the capital of Peru?",
    "Give me a Python one-liner to reverse a string.",
    "Describe photosynthesis in one sentence.",
    "Name two HTTP status codes for errors.",
    "How many bytes in a kibibyte?",
    "Write a SQL query that counts rows.",
    "Tell me a short joke.",
    "What year did Apollo 11 land?",
    "Define entropy without equations.",
    "Suggest a commit message for a typo fix.",
    "Explain the difference between TCP and UDP.",
    "Give a synonym for 'rapid'.",
    "What does the shell command `pwd` do?",
    "Outline a two-day trip to Kyoto.",
    "Convert 100C to Fahrenheit.",
    "Name a primary colour.",
    "What is a GPU good at?",
    "Write an ANSI escape to reset colours.",
    "How do I list files in Linux?",
    "Explain overfitting in one line.",
    "What port does HTTPS use?",
    "Give an antonym for 'scarce'.",
    "Draft a two-word commit subject.",
    "What is 17 times 19?",
    "Suggest a name for a paint app.",
    "Explain a hash table in one sentence.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Lockstep batched decode bench.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--prompt", type=str, default=None,
                        help="use N copies of one prompt (identical sequences)")
    parser.add_argument("--expert-cap", type=int, default=256)
    parser.add_argument("--expert-policy", choices=("lru", "lfu"), default="lru")
    parser.add_argument("--expert-pin", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="extra decode steps excluded from the timing")
    parser.add_argument("--union-strict", action="store_true",
                        help="hard-cap the batch union at --union-budget "
                             "experts (trades coverage for SSD bytes)")
    parser.add_argument("--union-budget", type=int, default=0,
                        help="cap the batch's expert union to N experts per "
                             "layer (0 = exact routing); reports gate-weight "
                             "coverage so quality is a measured number")
    parser.add_argument("--top-k", type=int, default=0,
                        help="activated experts per token (0 = config's "
                             "n_activated_experts=6); cuts route compute and "
                             "expert bytes roughly linearly")
    parser.add_argument("--no-shared", action="store_true",
                        help="skip the always-on shared expert (speed mode; "
                             "degrades output quality)")
    parser.add_argument("--hc-iters", type=int, default=0,
                        help="Sinkhorn iterations for hc mixes (0 = checkpoint's "
                             "hc_sinkhorn_iters=20, exact)")
    parser.add_argument("--window", type=int, default=0,
                        help="attention window size (0 = checkpoint's "
                             "window_size=128, exact)")
    parser.add_argument("--attn-bf16", action="store_true",
                        help="run the sparse attention core in bf16 (speed "
                             "mode; degrades numerics)")
    parser.add_argument("--phase", action="store_true",
                        help="sync after each block subphase and print where "
                             "decode time goes (forces GPU syncs; slow)")
    parser.add_argument("--skip-every", type=int, default=0,
                        help="bypass the MoE sublayer on layers where "
                             "layer %% N == 0 (0 = exact); cuts expert bytes "
                             "and MoE compute, degrades quality")
    parser.add_argument("--skip-block", type=int, default=0,
                        help="bypass the whole Block (hc mix + attention + "
                             "MoE) on layers where layer %% N == 0 (0 = "
                             "exact); skips all weight reads for those "
                             "layers, degrades quality")
    parser.add_argument("--keep-layers", type=int, default=-1,
                        help="keep only the first N index-source layers "
                             "under --skip-block (default -1 = keep all; 0 = "
                             "skip the whole stack)")
    parser.add_argument("--dedup", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="run each byte-identical prompt once and fan the "
                             "result out to its copies (exact; the default "
                             "prompt pool has 32 entries, so batch 512 is 16 "
                             "copies apiece). Use --no-dedup to time the "
                             "duplicated rows too")
    parser.add_argument("--proj-bf16", action="store_true",
                        help="run the wq_a/wkv contractions in bf16 instead "
                             "of f32 (Apple AMX only accelerates bf16 matmul); "
                             "costs numerics")
    parser.add_argument("--head-vocab", type=int, default=0,
                        help="restrict generation to the N lowest token ids "
                             "(0 = full 129280): head cost is linear in V, so "
                             "16384 takes f_head 8.20ms -> 1.37ms. Costs the "
                             "rest of the vocabulary")
    parser.add_argument("--head-compose", action=argparse.BooleanOptionalAction,
                        default=True,
                         help="fold embed+norm+head into one lookup table when "
                              "keep=0 and --head-vocab bounds the head")
    parser.add_argument("--step-cpu", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="run the composed decode step on the host "
                             "(~2.5us) instead of Metal (~13us)")
    parser.add_argument("--scale-inv-head", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="skip rmsnorm's row reduction in decode (argmax-"
                             "exact; only bf16/8-bit heads can differ)")
    parser.add_argument("--head-quant", action="store_true",
                        help="store the LM head as 8-bit affine groups "
                             "(gs=128): 683MB instead of 1324MB to read per "
                             "step, 8.0ms vs 13.8ms at batch 32; wins because "
                             "the head is bandwidth-bound, not FLOP-bound. "
                             "Costs numerics")
    parser.add_argument("--head-bf16", action="store_true",
                        help="run the LM head GEMM in bf16 instead of the "
                             "materialised f32 weight (halves the 2.66GiB "
                             "per-step read, uses tensor cores; costs numerics)")
    parser.add_argument("--no-engram", action="store_true",
                        help="drop the Engram memory path (ngram hash, row "
                             "fetch + fp8 dequant, wkv, gate); degrades "
                             "quality and removes the dominant decode phase")
    parser.add_argument("--batch-prefetch", action="store_true",
                        help="also warm the previous step's batch-wide expert "
                             "union (off by default: it thrashes the cache)")
    parser.add_argument("--indexer-gpu-topk", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="opt-in GPU indexer Top-K (experimental): keep scores "
                             "on device, verified bit-identical (10/10 tie microbench). "
                             "Default off. Unmeasured.")
    parser.add_argument("--moe-pipeline", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="opt-in async MoE pipeline (experimental): overlap "
                             "SSD expert loads with GPU qmms within one layer "
                             "via per-expert Futures + mx.async_eval. Routing, "
                             "weighting, and cache semantics unchanged; outputs "
                             "bit-identical (sorted scatter-add). Default off.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    n = args.batch
    if n < 1:
        parser.error("--batch must be >= 1")

    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json")
    config = load_json(snapshot / "inference" / "config.json")
    model_root = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    tokenizer = AutoTokenizer.from_pretrained(str(model_root))

    texts = [args.prompt] * n if args.prompt else DEFAULT_PROMPTS[:n]
    if args.prompt is None and n > len(DEFAULT_PROMPTS):
        texts = (DEFAULT_PROMPTS * ((n // len(DEFAULT_PROMPTS)) + 1))[:n]
    token_lists = [tokenizer.encode(encode_messages(
        [{"role": "user", "content": t}], "chat")) for t in texts]
    common = min(len(t) for t in token_lists)
    if common < 2:
        parser.error("prompts too short after truncation")
    prompts = [t[:common] for t in token_lists]

    pin = None
    if args.expert_pin:
        hot = load_json(REPO_ROOT / "profiles" / "deepseek-v4.1-flash"
                        / revision / "hot_experts.json")
        per = hot.get("per_layer", hot) if isinstance(hot, dict) else hot
        if isinstance(per, list):
            pin = {ly: [int(e) for e in es[:args.expert_pin]]
                   for ly, es in enumerate(per)}
        else:
            pin = {int(ly): [int(e) for e in es[:args.expert_pin]]
                   for ly, es in per.items() if str(ly).isdigit()}
        print(f"PIN top-{args.expert_pin}/layer = "
              f"{sum(len(v) for v in pin.values())} experts", flush=True)

    print(f"batch={n} prompt_tokens={common} max_tokens={args.max_tokens} "
          f"cap={args.expert_cap} policy={args.expert_policy} "
          f"prefetch={args.batch_prefetch} union={args.union_budget} "
          f"top_k={args.top_k or 'config'} shared={not args.no_shared}",
          flush=True)

    runner = Runner(model_root, manifest, config,
                    expert_cache_cap=args.expert_cap,
                    expert_policy=args.expert_policy,
                    expert_pin=pin,
                    max_batch=max(n, 1),
                    batch_prefetch=args.batch_prefetch,
                    union_budget=args.union_budget,
                    union_strict=args.union_strict,
                    top_k=args.top_k,
                    no_shared=args.no_shared,
                    hc_iters=args.hc_iters, window=args.window,
                    attn_bf16=args.attn_bf16, skip_every=args.skip_every,
                    skip_block=args.skip_block, no_engram=args.no_engram,
                    head_bf16=args.head_bf16, proj_bf16=args.proj_bf16,
                    keep_layers=args.keep_layers,
                    head_quant=args.head_quant,
                    scale_inv_head=args.scale_inv_head,
                    head_compose=args.head_compose,
                    step_cpu=args.step_cpu,
                    moe_pipeline=args.moe_pipeline,
                    indexer_gpu_topk=args.indexer_gpu_topk,
                    head_vocab=args.head_vocab)
    t0 = time.perf_counter()
    if args.phase:
        runner._phase = {}
        runner._att_phase_prof = True
    res = runner.generate_batch(prompts, args.max_tokens, dedup=args.dedup)
    if args.phase:
        ph = runner._phase
        total = res["decode_s"]
        parts = " ".join(f"{k}={v:.2f}" for k, v in sorted(ph.items()))
        print(f"PHASE decode_s={total:.2f} {parts} "
              f"other={total - sum(ph.values()):.2f}", flush=True)
        at = runner._att_stats or {}
        print("ATTN " + " ".join(f"{k}={v:.2f}" for k, v in sorted(at.items())),
              flush=True)
    wall = time.perf_counter() - t0
    ec = getattr(runner, "_engram_cache", None)
    if ec is not None:
        print(f"ENGRAM pages hits={ec.page_hits} misses={ec.page_misses} "
              f"read={ec.bytes_read / 2**30:.3f}GiB rows={ec.rows_served}",
              flush=True)
    runner.shutdown()

    print(f"BATCH tok_s={res['tok_s']:.4f} (aggregate) "
          f"per_seq={res['per_seq_tok_s']:.4f} "
          f"decode={res['decode_s']:.2f}s prefill={res['prefill_s']:.2f}s "
          f"wall={wall:.2f}s", flush=True)
    print(f"BATCH io={res['decode_io_s']:.2f}s "
          f"bytes={res['decode_bytes'] / 2**30:.2f}GiB "
          f"load_calls={res['decode_load_calls']} "
          f"hits={res['expert_cache_hits']} loads={res['expert_loads']}",
          flush=True)
    print(f"BATCH coverage={res['route_coverage']:.4f}", flush=True)
    if F.SYNC["n"]:
        print(f"BATCH moe_sync gpu_wait={F.SYNC['gpu_wait']:.2f}s "
              f"cpu={F.SYNC['cpu']:.2f}s n={int(F.SYNC['n'])}", flush=True)
    for i, toks in enumerate(res["outputs"]):
        print(f"BATCH[{i}] {toks[:8]}", flush=True)

    if args.json_out:
        payload = {"batch": n, "prompt_tokens": common,
                   "max_tokens": args.max_tokens,
                   "config": {"expert_cap": args.expert_cap,
                              "expert_policy": args.expert_policy,
                              "expert_pin": args.expert_pin,
                              "batch_prefetch": args.batch_prefetch,
                              "union_budget": args.union_budget,
                              "top_k": args.top_k,
                              "no_shared": args.no_shared,
                              "hc_iters": args.hc_iters,
                              "window": args.window,
                              "attn_bf16": args.attn_bf16,
                              "skip_every": args.skip_every,
                              "skip_block": args.skip_block,
                              "no_engram": args.no_engram,
                              "head_bf16": args.head_bf16,
                              "proj_bf16": args.proj_bf16,
                              "keep_layers": args.keep_layers,
                              "head_quant": args.head_quant,
                              "head_vocab": args.head_vocab,
                              "scale_inv_head": args.scale_inv_head,
                              "head_compose": args.head_compose,
                              "step_cpu": args.step_cpu,
                              "moe_pipeline": args.moe_pipeline,
                              "indexer_gpu_topk": args.indexer_gpu_topk,
                              "dedup": args.dedup},
                   "route_coverage": res["route_coverage"],
                   "result": res, "wall_s": wall}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
