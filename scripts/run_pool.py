#!/usr/bin/env python3
"""Issue #23: replay a trace through the real bounded pool (demand paging).

    python scripts/run_pool.py --trace /tmp/t.jsonl --slots 6 --policy lru
    python scripts/run_pool.py --trace /tmp/t.jsonl --slots 6 --max-tokens 8 --json-out pool.json

Loads REAL fp4 expert bytes from the verified checkpoint on miss and
verifies a sample byte-equal against fresh reads (quality invariant).
Trace may be synthetic until #20 unblocks (schema-identical).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.pool import ResidentPool, load_expert_bytes
from deepseeker.trace import expert_records, read_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay trace via bounded pool.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--policy", type=str, default="lru")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="cap tokens replayed (0 = all; bounds checkpoint IO)")
    parser.add_argument("--verify-every", type=int, default=50,
                        help="byte-verify every Nth served expert vs fresh read")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        _header, records = read_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    model_root = args.model_root or REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    revision = load_json(
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    per_expert: dict[tuple[int, int], int] = {}
    for tensor in manifest["tensors"]:
        if tensor.get("subsystem") == "routed-expert":
            key = (tensor["layer"], tensor["expert"])
            per_expert[key] = per_expert.get(key, 0) + tensor["bytes"]
    max_blob = max(per_expert.values())
    pool = ResidentPool(args.slots, max_blob, args.policy)
    ordered = sorted(expert_records(records),
                     key=lambda r: (r["request_id"], r["token_pos"], r["layer"]))
    seen_tokens: set[tuple[str, int]] = set()
    served = verified = 0
    start = time.perf_counter()
    for record in ordered:
        key = (record["request_id"], record["token_pos"])
        if args.max_tokens and len(seen_tokens) >= args.max_tokens and key not in seen_tokens:
            break
        seen_tokens.add(key)
        for expert in record["experts"]:
            blob = pool.demand(
                record["layer"], expert,
                lambda layer, expert: load_expert_bytes(model_root, manifest, layer, expert),
            )
            served += 1
            if served % args.verify_every == 0:
                fresh = load_expert_bytes(model_root, manifest, record["layer"], expert)
                if bytes(blob) != bytes(fresh):
                    print(f"FAIL: stale bytes at layer {record['layer']} expert {expert}")
                    return 1
                verified += 1
    elapsed = time.perf_counter() - start
    metrics = pool.metrics()
    print(f"tokens={len(seen_tokens)} served={served} verified={verified} "
          f"hit_rate={metrics['hit_rate']:.3f} evictions={metrics['evictions']} "
          f"resident={metrics['bytes_resident'] / 2**20:.0f}MB "
          f"high_water={metrics['high_water_bytes'] / 2**20:.0f}MB "
          f"elapsed={elapsed:.1f}s")
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"tokens": len(seen_tokens), "served": served,
                        "verified": verified, "elapsed_s": round(elapsed, 1),
                        **metrics}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
