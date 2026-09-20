#!/usr/bin/env python3
"""Issue #11: simulate P4 resident-pool policies on a routing trace.

    python scripts/run_resident_sim.py --trace /tmp/t.jsonl --budgets 6,12,24
    python scripts/run_resident_sim.py --trace /tmp/t.jsonl --budgets 12 \\
        --policies demand_lru,prefetch,belady --predictor persistence --json-out res.json

Budgets are per-layer resident slots. belady is the offline-optimal upper
bound; the demand_lru-to-belady gap is what prefetching can close.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.resident import POLICIES, compare
from deepseeker.simulate import PREDICTORS
from deepseeker.trace import read_trace


def expert_bytes_default() -> int:
    revision = load_json(
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    return int(manifest.get("expert_stats", {}).get("median_bytes", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate resident-pool policies.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--budgets", type=str, default="6,12,24")
    parser.add_argument("--policies", type=str, default=",".join(POLICIES))
    parser.add_argument("--predictor", type=str, default="persistence")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        header, records = read_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    policies = tuple(p for p in args.policies.split(",") if p.strip())
    unknown = [p for p in policies if p not in POLICIES]
    if unknown:
        print(f"FAIL: unknown policies {unknown}; choose from {','.join(POLICIES)}")
        return 2
    if args.predictor not in PREDICTORS:
        print(f"FAIL: unknown predictor {args.predictor!r}")
        return 2
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    if any(b < 1 for b in budgets):
        print("FAIL: budgets must be >= 1")
        return 2
    rows = compare(records, budgets, header["top_k"], args.predictor, expert_bytes_default(), policies)

    print(f"trace: {args.trace} ({len(records)} records, predictor={args.predictor})")
    print("policy        budget  hit_rate  miss/rec  missMB/rec  evict   churn")
    for row in rows:
        miss_mb = row["miss_bytes_per_record"] / 1024 / 1024
        print(
            f"{row['policy']:<13} {row['budget']:>6}  {row['hit_rate']:>7.3f}  "
            f"{row['miss_per_record']:>8.3f}  {miss_mb:>9.1f}  "
            f"{row['evictions']:>5}  {row['churn_experts']:>5}"
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
