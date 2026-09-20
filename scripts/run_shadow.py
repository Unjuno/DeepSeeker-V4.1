#!/usr/bin/env python3
"""Issue #10: shadow-run predictors and report P3 switch-on decisions.

    python scripts/run_shadow.py --trace /tmp/t.jsonl --budget 12 --horizon 1
    python scripts/run_shadow.py --trace /tmp/t.jsonl --budget 12 --window 20 \\
        --min-recall 0.7 --max-waste-mb 200 --json-out shadow.json

Exit 0 always (a BLOCKED verdict is data, not failure); exit 2 only on
bad input. The shadow never controls memory by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.shadow import run_shadow, switch_on
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
    parser = argparse.ArgumentParser(description="Shadow-run predictors.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--horizon", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--predictors", type=str, default=",".join(PREDICTORS),
        help=f"subset of {','.join(PREDICTORS)}",
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--min-recall", type=float, default=0.7)
    parser.add_argument("--max-waste-mb", type=float, default=200.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        header, records = read_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    names = tuple(p for p in args.predictors.split(",") if p.strip())
    unknown = [p for p in names if p not in PREDICTORS]
    if unknown:
        print(f"FAIL: unknown predictors {unknown}")
        return 2
    expert_bytes = expert_bytes_default()
    max_waste = args.max_waste_mb * 1024 * 1024

    print(f"trace: {args.trace} horizon={args.horizon} budget={args.budget}")
    print("predictor       mean_recall  waste/tok  switch_at")
    results = []
    for name in names:
        run = run_shadow(records, name, args.budget, header["top_k"], expert_bytes, args.horizon)
        at = switch_on(run["log"], args.window, args.min_recall, max_waste)
        verdict = f"token#{at}" if at is not None else "BLOCKED"
        print(
            f"{name:<15} {run['mean_recall']:>11.3f}  "
            f"{run['mean_waste_per_token'] / 1024 / 1024:>8.1f}MB  {verdict}"
        )
        results.append({**{k: v for k, v in run.items() if k != "log"}, "switch_at": at})
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out} (per-token log omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
