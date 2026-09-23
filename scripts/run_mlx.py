#!/usr/bin/env python3
"""MLX-native DeepSeek-V4.1-Flash runner: real tokens + real traces.

    python scripts/run_mlx.py --prompt "Say hello." --max-tokens 10
    python scripts/run_mlx.py --prompt "Say hello." --max-tokens 10 \\
        --trace-out trace.jsonl --json-out result.json

Builds the harmony chat prompt, generates with the streaming MLX
runner, records authoritative router records in deepseeker.trace/v1
(#20), and reports tok/s with timings.
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
from deepseeker.trace import make_header, read_trace, write_trace


class TraceSink:
    """Collects authoritative router events into trace records."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.records: list[dict] = []
        self.tokens_seen: list[int] = []

    def router(self, layer: int, token_start: int, idx: list, weights: list) -> None:
        for i, (experts, scores) in enumerate(zip(idx, weights)):
            self.records.append({
                "request_id": self.request_id,
                "token_pos": token_start + i,
                "layer": layer,
                "experts": [int(e) for e in experts],
                "scores": [float(s) for s in scores],
            })

    def token(self, token: int) -> None:
        self.tokens_seen.append(token)

    def engram(self, layer: int, token_start: int, hash_ids) -> None:
        self.records.append({
            "kind": "event",
            "request_id": self.request_id,
            "token_pos": token_start,
            "layer": layer,
            "extra": {"kind": "engram", "n_hashes": len(hash_ids),
                      "sample_rows": [int(r) for r in hash_ids[0][:4]]},
        })

    def kv(self, layer: int, token_start: int, compress_rows: int, topk_count: int) -> None:
        self.records.append({
            "kind": "event",
            "request_id": self.request_id,
            "token_pos": token_start,
            "layer": layer,
            "extra": {"kind": "kv", "compress_rows": compress_rows,
                      "topk_count": topk_count},
        })


def main() -> int:
    parser = argparse.ArgumentParser(description="MLX-native generation.")
    parser.add_argument("--prompt", type=str, default="Say hello.")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--expert-cap", type=int, default=None,
                        help="dequantized expert cache cap (default 256; min 240)")
    parser.add_argument("--tune", type=Path, default=None,
                        help="recommended.json from autotune.py; sets cap "
                             "from resident slots (bf16-corrected)")
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json")
    config = load_json(snapshot / "inference" / "config.json")
    model_root = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    tokenizer = AutoTokenizer.from_pretrained(str(model_root))

    request_id = f"mlx-{int(time.time())}"
    sink = TraceSink(request_id)
    expert_cap = args.expert_cap or 256
    if args.tune:
        tune = load_json(args.tune)
        # recommended.json sizes slots with encoded fp4 (~18.8MB); the cache
        # stores bf16 (~70.8MB). Convert before applying, then clamp to the
        # safe bf16 working-set range for 64GB (min 240, max ~400).
        fp4_bytes = 18_800_640
        bf16_bytes = 5120 * 2304 * 3 * 2
        slots = int(tune["config"]["resident_slots_per_layer"])
        raw_cap = slots * 40
        # If slots were computed in fp4 units, scale down by size ratio when
        # the product would exceed bf16 budget (~27GiB for 400 experts).
        if raw_cap * bf16_bytes > 30 * 1024**3:
            raw_cap = int(raw_cap * fp4_bytes / bf16_bytes)
        expert_cap = max(240, min(raw_cap, 400))
        print(f"tune: expert_cap={expert_cap} "
              f"(raw={slots}x40={slots * 40}, predictor={tune['config']['predictor']})",
              flush=True)
    runner = Runner(model_root, manifest, config, trace_hook=sink,
                    expert_cache_cap=expert_cap)
    try:
        prompt = encode_messages([{"role": "user", "content": args.prompt}], "chat")
        ids = tokenizer.encode(prompt)
        print(f"prompt tokens: {len(ids)}", flush=True)
        out = runner.generate(ids, args.max_tokens, temperature=args.temperature)
        text = tokenizer.decode(out["tokens"])
        print(f"TEXT: {text!r}", flush=True)
        print(f"tokens={out['timings']['tokens']} "
              f"prefill={out['timings']['prefill_s']:.1f}s "
              f"tok/s={out['timings']['tok_s']:.4f}", flush=True)
        result = {
            "prompt": args.prompt,
            "tokens": out["tokens"],
            "text": text,
            "timings": out["timings"],
            "expert_loads": runner.expert_loads,
            "expert_cache_hits": runner.expert_cache_hits,
            "expert_evictions": runner.expert_evictions,
            "expert_cap": expert_cap,
        }
        if args.trace_out:
            from deepseeker.trace import arch_config

            header = make_header(
                "deepseek-ai/DeepSeek-V4.1-Flash", revision, "reference-mlx",
                arch_config(manifest))
            n = write_trace(args.trace_out, header, sink.records)
            print(f"wrote {n} trace records to {args.trace_out}", flush=True)
            _back_header, back = read_trace(args.trace_out)
            print(f"trace validate: {len(back)} records ok", flush=True)
            result["trace_records"] = len(back)
        if args.json_out:
            args.json_out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {args.json_out}", flush=True)
    finally:
        runner.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
