#!/usr/bin/env python3
"""Issue #9: compare P2 baseline predictors on a routing trace.

    python scripts/run_routing_sim.py --trace /tmp/t.jsonl --budgets 6,12,24
    python scripts/run_routing_sim.py --trace /tmp/t.jsonl --budgets 6,12,24 --json-out sim.json

Budgets are per-layer resident-expert slots. Table columns are defined in
docs/SIMULATOR.md; higher recall_at_k with lower false_pos wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.simulate import PREDICTORS, compare
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
    parser = argparse.ArgumentParser(description="Compare routing predictors.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--budgets", type=str, default="6,12,24")
    parser.add_argument(
        "--predictors", type=str, default=",".join(PREDICTORS),
        help=f"subset of {','.join(PREDICTORS)}",
    )
    parser.add_argument("--expert-bytes", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        header, records = read_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    names = tuple(p for p in args.predictors.split(",") if p.strip())
    unknown = [p for p in names if p not in PREDICTORS]
    if unknown:
        print(f"FAIL: unknown predictors {unknown}; choose from {','.join(PREDICTORS)}")
        return 1
    expert_bytes = args.expert_bytes if args.expert_bytes is not None else expert_bytes_default()
    rows = compare(records, budgets, header["top_k"], expert_bytes, names)

    print(f"trace: {args.trace} ({len(records)} records, top_k={header['top_k']})")
    print("predictor       budget  recall@k  exact   fp/rec  waste/rec")
    for row in rows:
        waste_mb = row["wasted_bytes"] / 1024 / 1024
        print(
            f"{row['predictor']:<15} {row['budget']:>6}  "
            f"{row['recall_at_k']:>7.3f}  {row['exact_set']:>5.3f}  "
            f"{row['false_pos']:>6.2f}  {waste_mb:>7.1f}MB"
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
