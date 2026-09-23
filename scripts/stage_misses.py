#!/usr/bin/env python3
"""Issue #24: replay a miss stream through staging into the pool.

    python scripts/stage_misses.py --trace /tmp/t.jsonl --slots 6 --qd 8 --max-tokens 4
    python scripts/stage_misses.py --trace /tmp/t.jsonl --slots 6 --qd 8 --max-tokens 4 --json-out s.json

Every miss stages six real ranges and admits verified bytes; service
metrics are comparable against the #19 envelope.
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
from deepseeker.pool import ResidentPool
from deepseeker.stage import StagingEngine
from deepseeker.trace import expert_records, read_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay misses via staging.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--qd", type=int, default=8)
    parser.add_argument("--staging-mib", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=4)
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
    pool = ResidentPool(args.slots, max(per_expert.values()))
    engine = StagingEngine(model_root, manifest, max_workers=args.qd,
                           max_staging_bytes=args.staging_mib * 1024**2)
    ordered = sorted(expert_records(records),
                     key=lambda r: (r["request_id"], r["token_pos"], r["layer"]))
    seen: set[tuple[str, int]] = set()
    misses = 0
    start = time.perf_counter()
    try:
        for record in ordered:
            key = (record["request_id"], record["token_pos"])
            if args.max_tokens and len(seen) >= args.max_tokens and key not in seen:
                break
            seen.add(key)
            for expert in record["experts"]:
                if pool.lookup(record["layer"], expert) is None:
                    misses += 1
                    engine.stage_into_pool(pool, record["layer"], expert)
    finally:
        engine.shutdown()
    elapsed = time.perf_counter() - start
    pool_m = pool.metrics()
    eng_m = engine.metrics()
    print(f"tokens={len(seen)} misses={misses} hit_rate={pool_m['hit_rate']:.3f} "
          f"evictions={pool_m['evictions']} dedup={eng_m['dedup_hits']} "
          f"cancelled={eng_m['cancelled']} failures={eng_m['failures']} "
          f"mean_service={eng_m['mean_service_s']*1000:.1f}ms "
          f"mean_queue_wait={eng_m['mean_queue_wait_s']*1000:.1f}ms "
          f"elapsed={elapsed:.1f}s")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"tokens": len(seen), "misses": misses, "pool": pool_m,
             "staging": eng_m, "elapsed_s": round(elapsed, 1)}, indent=1) + "\n",
            encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
