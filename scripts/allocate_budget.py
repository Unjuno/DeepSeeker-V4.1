#!/usr/bin/env python3
"""Issue #13: compare cross-layer slot allocations at a fixed total.

    python scripts/allocate_budget.py --trace /tmp/t.jsonl --total 480
    python scripts/allocate_budget.py --trace /tmp/t.jsonl --total 480 \\
        --curve-budgets 6,12,24,48 --json-out alloc.json

Scored with the Issue #11 demand_lru pool; greedy water-fills measured
hit-rate curves, proportional follows unique-expert demand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.allocate import (
    allocate_equal,
    allocate_greedy,
    allocate_proportional,
    evaluate_allocation,
    hit_curve,
    unique_demand,
)
from deepseeker.trace import read_trace

STRATEGIES = ("equal", "proportional", "greedy")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare budget allocations.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--curve-budgets", type=str, default="6,12,24,48")
    parser.add_argument("--strategies", type=str, default=",".join(STRATEGIES))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        header, records = read_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    strategies = tuple(s for s in args.strategies.split(",") if s.strip())
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        print(f"FAIL: unknown strategies {unknown}")
        return 2
    n_layers, top_k = header["n_layers"], header["top_k"]
    try:
        splits = {}
        if "equal" in strategies:
            splits["equal"] = allocate_equal(args.total, n_layers, top_k)
        if "proportional" in strategies:
            splits["proportional"] = allocate_proportional(
                args.total, unique_demand(records, n_layers), top_k
            )
        if "greedy" in strategies:
            grid = [int(b) for b in args.curve_budgets.split(",") if b.strip()]
            curves = {layer: hit_curve(records, layer, grid, top_k) for layer in range(n_layers)}
            splits["greedy"] = allocate_greedy(curves, args.total, top_k)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 2

    print(f"trace: {args.trace} total={args.total} slots over {n_layers} layers")
    rows = []
    for name, split in splits.items():
        lo, hi = min(split.values()), max(split.values())
        result = evaluate_allocation(records, split, top_k)
        print(
            f"{name:<12} hit_rate={result['hit_rate']:.3f} "
            f"miss/rec={result['miss_per_record']:.3f} range=[{lo},{hi}]"
        )
        rows.append({"strategy": name, "allocation": split, **result})
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
