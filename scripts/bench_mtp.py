#!/usr/bin/env python3
"""Issue #29: DSpark/MTP speculative decode baseline on M1 Max.

Compares ordinary greedy decode vs generate_spec (draft+verify) on the same
prompt, reporting acceptance rate, block stats, and accepted tok/s.

    python scripts/bench_mtp.py --prompt "Say hello." --max-tokens 16
    python scripts/bench_mtp.py --prompt "Say hello." --max-tokens 16 --json-out mtp.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"))

from encoding.encoding import encode_messages
from transformers import AutoTokenizer

from deepseeker.baseline import load_json
from deepseeker.mlx_runner import Runner


def main() -> int:
    parser = argparse.ArgumentParser(description="DSpark/MTP speculative baseline.")
    parser.add_argument("--prompt", type=str, default="Say hello.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--expert-cap", type=int, default=256,
                        help="total expert cache budget; without static pins, the LRU floor is 240")
    parser.add_argument("--expert-policy", choices=("lru", "lfu"), default="lru",
                        help="expert cache replacement: lru (default) or lfu")
    parser.add_argument("--expert-pin", type=int, default=0,
                        help="statically pin top-K hot experts per layer (0=off)")
    parser.add_argument("--expert-pin-gather", action="store_true",
                        help="EXPERIMENTAL: bank pins and use gather_qmm; may change numerics")
    parser.add_argument("--direct-pread", action="store_true",
                        help="EXPERIMENTAL: read expert MXFP4 bytes directly into MLX buffers; opt-in loader A/B")
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
    parser.add_argument("--skip-every", type=int, default=0,
                        help="bypass the MoE sublayer on layers where "
                             "layer %% N == 0 (0 = exact); cuts expert bytes "
                             "and MoE compute, degrades quality")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="only run speculative path")
    parser.add_argument("--skip-spec", action="store_true",
                        help="only run ordinary decode")
    probe_group = parser.add_mutually_exclusive_group()
    probe_group.add_argument("--spec-batch-probe", action="store_true",
                             help="run the extra diagnostic multi-token verifier (outputs are "
                                  "discarded; normal token commits still use serial target forwards)")
    probe_group.add_argument("--spec-no-batch-probe", action="store_true",
                             help="deprecated compatibility alias; diagnostic probe is off by default")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.direct_pread:
        os.environ["DEEPSEEKER_DIRECT_PREAD"] = "1"
    direct_pread = os.environ.get("DEEPSEEKER_DIRECT_PREAD") == "1"
    pre_attn_prefetch = os.environ.get("DEEPSEEKER_PREATTN_PREFETCH") == "1"
    if direct_pread and (pre_attn_prefetch or args.expert_pin_gather):
        print("WARNING: pre-attention speculative reads and pinned expert banks still use "
              "the packed loader path; direct-pread covers synchronous per-expert loads only",
              flush=True)

    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json")
    config = load_json(snapshot / "inference" / "config.json")
    model_root = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    tokenizer = AutoTokenizer.from_pretrained(str(model_root))

    prompt = encode_messages([{"role": "user", "content": args.prompt}], "chat")
    ids = tokenizer.encode(prompt)
    print(f"prompt tokens: {len(ids)} block_size={config.get('dspark_block_size')} "
          f"n_mtp={config.get('n_mtp_layers')}", flush=True)

    expert_pin: dict[int, list[int]] = {}
    eff_cap = args.expert_cap
    if args.expert_pin > 0:
        hot = load_json(REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision
                        / "hot_experts.json")
        expert_pin = {int(ly): [int(e) for e in es[:args.expert_pin]]
                      for ly, es in enumerate(hot["per_layer"])}
        n_pin = sum(len(v) for v in expert_pin.values())
        eff_cap = max(1, args.expert_cap - n_pin)  # LRU budget; pins are exempt
        print(f"PIN top-{args.expert_pin}/layer = {n_pin} experts, "
              f"lru_cap={eff_cap} total_budget={args.expert_cap}", flush=True)
    if args.expert_pin_gather and args.expert_pin <= 0:
        parser.error("--expert-pin-gather requires --expert-pin > 0")
    runner = Runner(model_root, manifest, config, expert_cache_cap=eff_cap,
                    expert_policy=args.expert_policy, expert_pin=expert_pin,
                    expert_pin_gather=args.expert_pin_gather,
                    top_k=args.top_k, no_shared=args.no_shared,
                    hc_iters=args.hc_iters, window=args.window,
                    attn_bf16=args.attn_bf16, skip_every=args.skip_every)
    if args.top_k:
        print(f"TOP-K override: {args.top_k} "
              f"(config n_activated_experts={config.get('n_activated_experts')})",
              flush=True)
    result: dict = {
        "prompt": args.prompt,
        "prompt_tokens": len(ids),
        "max_tokens": args.max_tokens,
        "block_size": config.get("dspark_block_size"),
        "n_mtp_layers": config.get("n_mtp_layers"),
        "config": {
            "dspark_target_layer_ids": config.get("dspark_target_layer_ids"),
            "dspark_n_activated_experts": config.get("dspark_n_activated_experts"),
            "dspark_noise_token_id": config.get("dspark_noise_token_id"),
            "expert_cap": args.expert_cap,
            "expert_lru_cap": eff_cap,
            "expert_policy": args.expert_policy,
            "expert_pin": args.expert_pin,
            "expert_pin_gather": args.expert_pin_gather,
            "direct_pread": direct_pread,
            "pre_attn_prefetch": pre_attn_prefetch,
            "top_k": args.top_k,
            "no_shared": args.no_shared,
            "hc_iters": args.hc_iters,
            "window": args.window,
            "attn_bf16": args.attn_bf16,
            "skip_every": args.skip_every,
            "spec_batch_probe": args.spec_batch_probe,
            "spec_no_batch_probe": args.spec_no_batch_probe,
        },
    }
    try:
        if not args.skip_baseline:
            t0 = time.perf_counter()
            base = runner.generate(ids, args.max_tokens, temperature=args.temperature)
            wall = time.perf_counter() - t0
            base_text = tokenizer.decode(base["tokens"])
            result["baseline"] = {
                "tokens": base["tokens"],
                "text": base_text,
                "timings": base["timings"],
                "wall_s": wall,
            }
            print(f"BASELINE tokens={base['timings']['tokens']} "
                  f"prefill={base['timings']['prefill_s']:.1f}s "
                  f"decode={base['timings']['decode_s']:.1f}s "
                  f"tok/s={base['timings']['tok_s']:.4f}", flush=True)
            t = base["timings"]
            if "expert_hit_rate" in t:
                print(f"BASE_CACHE loads={t['expert_loads']} "
                      f"hits={t['expert_cache_hits']} "
                      f"evict={t['expert_evictions']} "
                      f"hit_rate={t['expert_hit_rate']:.3f}", flush=True)
            ld = runner._expert_loader
            n = t["tokens"] or 1
            print(f"BASE_PHASE io_wall={ld.io_s:.1f}s ({ld.io_s / n * 1e3:.0f} ms/tok, incl pack) "
                  f"pack_sum={ld.dequant_s:.1f}s ({ld.dequant_s / n * 1e3:.0f} ms/tok) "
                  f"upload={ld.eval_s:.1f}s ({ld.eval_s / n * 1e3:.0f} ms/tok) "
                  f"read={ld.bytes_read / 2 ** 30:.2f}GiB "
                  f"({ld.bytes_read / n / 2 ** 30:.3f} GiB/tok)", flush=True)
            print(f"BASE_DECODE io={t.get('decode_io_s', 0.0):.1f}s "
                  f"({t.get('decode_io_s', 0.0) / n * 1e3:.0f} ms/tok) "
                  f"pack={t.get('decode_pack_s', 0.0):.1f}s "
                  f"({t.get('decode_pack_s', 0.0) / n * 1e3:.0f} ms/tok) "
                  f"read={t.get('decode_bytes', 0) / 2 ** 30:.2f}GiB "
                  f"({t.get('decode_bytes', 0) / n / 2 ** 30:.3f} GiB/tok) "
                  f"loads={t.get('decode_expert_loads', 0)} "
                  f"non_io={t['decode_s'] - t.get('decode_io_s', 0.0):.1f}s "
                  f"({(t['decode_s'] - t.get('decode_io_s', 0.0)) / n * 1e3:.0f} ms/tok)",
                  flush=True)
            print(f"BASE_ENGRAM engram={t.get('decode_engram_s', 0.0):.1f}s "
                  f"({t.get('decode_engram_s', 0.0) / n * 1e3:.0f} ms/tok) "
                  f"hits={t.get('engram_page_hits', 0)} "
                  f"miss={t.get('engram_page_misses', 0)} "
                  f"read={t.get('engram_bytes_read', 0) / 2 ** 20:.1f}MiB "
                  f"rows={t.get('engram_rows', 0)}", flush=True)
            steps_ms = [f"{s * 1e3:.0f}" for s in t.get("steps_s", [])]
            if steps_ms:
                print(f"BASE_STEPS ms/token: {' '.join(steps_ms)}", flush=True)
            print(f"BASE_TEXT: {base_text!r}", flush=True)
            # Baseline leaves KV/compress/index state behind; clear it so the
            # spec run sees the same clean prefill (expert cache stays warm).

        if not args.skip_spec:
            runner.reset_seq_state()
            t0 = time.perf_counter()
            spec = runner.generate_spec(
                ids, args.max_tokens, temperature=args.temperature,
                batch_probe=args.spec_batch_probe)
            wall = time.perf_counter() - t0
            spec_text = tokenizer.decode(spec["tokens"])
            result["spec"] = {
                "tokens": spec["tokens"],
                "text": spec_text,
                "timings": spec["timings"],
                "spec": spec["spec"],
                "wall_s": wall,
            }
            sp = spec["spec"]
            print(f"SPEC tokens={spec['timings']['tokens']} "
                  f"prefill={spec['timings']['prefill_s']:.1f}s "
                  f"decode={spec['timings']['decode_s']:.1f}s "
                  f"tok/s={spec['timings']['tok_s']:.4f}", flush=True)
            print(f"SPEC stats: proposed={sp['proposed']} accepted={sp['accepted']} "
                  f"acceptance={sp['acceptance']:.3f} "
                  f"full_accepts={sp['full_accepts']} rollbacks={sp['rollbacks']} "
                  f"serial_rejects={sp['serial_rejects']} "
                f"draft={sp['draft_s']:.3f}s verify={sp['verify_s']:.3f}s "
                f"target={sp['target_s']:.3f}s", flush=True)
            if sp.get("draft_n"):
                hits = sp.get("draft_hit", [])
                print("SPEC draft_by_idx: " + " ".join(
                    f"{h}/{n}" for h, n in zip(hits, sp["draft_n"])), flush=True)
            if sp.get("serial_draft_n_by_idx"):
                print("SPEC serial_draft_by_idx: " + " ".join(
                    f"{h}/{n}" for h, n in zip(
                        sp["serial_draft_hit_by_idx"], sp["serial_draft_n_by_idx"])), flush=True)
            print(f"SPEC_TEXT: {spec_text!r}", flush=True)

        if "baseline" in result and "spec" in result:
            bt = result["baseline"]["timings"]["tok_s"]
            st = result["spec"]["timings"]["tok_s"]
            result["tok_s_ratio"] = st / bt if bt else None
            same = result["baseline"]["tokens"] == result["spec"]["tokens"]
            result["lossless_match"] = same
            print(f"lossless_match={same} tok_s_ratio={result['tok_s_ratio']}", flush=True)

        if args.json_out:
            args.json_out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {args.json_out}", flush=True)
    finally:
        runner.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
