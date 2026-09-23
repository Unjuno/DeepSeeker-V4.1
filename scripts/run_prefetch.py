#!/usr/bin/env python3
"""Issue #25: A/B demand-only vs gated prefetch replay.

    python scripts/run_prefetch.py --trace /tmp/t.jsonl --slots 6 --max-tokens 6
    python scripts/run_prefetch.py --trace /tmp/t.jsonl --slots 6 --max-tokens 6 --json-out p.json

Both arms serve byte-identical outputs (compared); the verdict is about
stalls and waste, never quality. A BLOCKED verdict is reported as data.
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
from deepseeker.prefetch import PrefetchController
from deepseeker.stage import StagingEngine
from deepseeker.trace import expert_records, read_trace


def group_tokens(records):
    groups = {}
    for record in expert_records(records):
        groups.setdefault((record["request_id"], record["token_pos"]), []).append(record)
    return [groups[k] for k in sorted(groups)]


def run_arm(token_groups, manifest, model_root, slots, policy, use_prefetch,
            predictor, budget, horizon, min_agreement, window, max_blob):
    pool = ResidentPool(slots, max_blob, policy)
    engine = StagingEngine(model_root, manifest, max_workers=8)
    controller = PrefetchController(
        pool, engine, predictor, budget, horizon,
        enabled=use_prefetch, min_agreement=min_agreement, window=window,
    )
    digest_parts = []
    stalls = 0
    start = time.perf_counter()
    try:
        for records in token_groups:
            if use_prefetch:
                controller.prefetch_ahead()
            for record in sorted(records, key=lambda r: r["layer"]):
                for expert in record["experts"]:
                    if use_prefetch:
                        out = controller.serve(
                            record["layer"], expert,
                            lambda layer, expert: load_expert_bytes(
                                model_root, manifest, layer, expert),
                        )
                    else:
                        hit = pool.lookup(record["layer"], expert)
                        if hit is None:
                            stalls += 1
                            out = pool.demand(
                                record["layer"], expert,
                                lambda layer, expert: load_expert_bytes(
                                    model_root, manifest, layer, expert),
                            )
                        else:
                            out = bytes(hit)
                    digest_parts.append((record["layer"], expert, hash(bytes(out))))
            if use_prefetch:
                controller.observe_token(records)
    finally:
        engine.shutdown()
    elapsed = time.perf_counter() - start
    report = controller.report() if use_prefetch else {}
    return {
        "digest": digest_parts,
        "stalls": stalls,
        "elapsed_s": round(elapsed, 2),
        "pool": pool.metrics(),
        "prefetch": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B prefetch replay.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--policy", type=str, default="lru")
    parser.add_argument("--predictor", type=str, default="persistence")
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-agreement", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=6)
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

    groups = group_tokens(records)
    if args.max_tokens:
        seen = set()
        picked = []
        for group in groups:
            key = (group[0]["request_id"], group[0]["token_pos"])
            if len(seen) >= args.max_tokens and key not in seen:
                break
            seen.add(key)
            picked.append(group)
        groups = picked
    common = {"manifest": manifest, "model_root": model_root, "slots": args.slots,
              "policy": args.policy, "max_blob": max_blob, "predictor": args.predictor,
              "budget": args.budget, "horizon": args.horizon,
              "min_agreement": args.min_agreement, "window": args.window}
    off = run_arm(groups, use_prefetch=False, **common)
    on = run_arm(groups, use_prefetch=True, **common)
    identical = off["digest"] == on["digest"]
    on["stalls"] = on["pool"]["misses"]  # demand-fallback serves incl. first touch
    pre = on["prefetch"]
    verdict = "GO" if pre.get("enabled") and not pre.get("auto_disabled") else "BLOCKED"
    print(f"quality identical: {identical}")
    print(f"OFF stalls={off['stalls']} hit_rate={off['pool']['hit_rate']:.3f} "
          f"elapsed={off['elapsed_s']}s")
    print(f"ON  stalls={on['stalls']} hit_rate={on['pool']['hit_rate']:.3f} "
          f"submits={pre.get('prefetch_submits')} served_prefetched={pre.get('served_prefetched')} "
          f"stale={pre.get('stale_prefetched')} trailing_agreement={pre.get('trailing_agreement')} "
          f"elapsed={on['elapsed_s']}s")
    print(f"verdict: {verdict}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"identical": identical, "verdict": verdict, "off": {k: v for k, v in off.items() if k != "digest"},
             "on": {k: v for k, v in on.items() if k != "digest"}}, indent=1) + "\n",
            encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
