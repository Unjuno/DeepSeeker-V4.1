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
                        help="expert cache cap; min 240 forced by working set")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="only run speculative path")
    parser.add_argument("--skip-spec", action="store_true",
                        help="only run ordinary decode")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

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

    runner = Runner(model_root, manifest, config, expert_cache_cap=args.expert_cap)
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
            print(f"BASE_TEXT: {base_text!r}", flush=True)
            # Fresh caches for fair spec comparison? Keep same runner (cache warm).
            # Reset KV by reconstructing would reload 16GiB — instead note warm.

        if not args.skip_spec:
            t0 = time.perf_counter()
            spec = runner.generate_spec(ids, args.max_tokens, temperature=args.temperature)
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
                  f"draft={sp['draft_s']:.3f}s verify={sp['verify_s']:.3f}s "
                  f"target={sp['target_s']:.3f}s", flush=True)
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
