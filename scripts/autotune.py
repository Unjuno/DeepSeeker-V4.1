#!/usr/bin/env python3
"""Issue #12: emit the P8 startup recommended config.

    python scripts/autotune.py
    python scripts/autotune.py --sim sim.json --shadow0 sh0.json --shadow1 sh1.json \\
        --target-recall 0.7 --out recommended.json

Profiles are read from the repo; sim/shadow JSONs come from
run_routing_sim.py / run_shadow.py --json-out. Missing inputs fall back
to documented priors (see docs/AUTOTUNE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.autotune import recommend
from deepseeker.baseline import load_json


def maybe_json(path: Path | None):
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend startup config.")
    parser.add_argument("--sim", type=Path, default=None)
    parser.add_argument("--shadow0", type=Path, default=None)
    parser.add_argument("--shadow1", type=Path, default=None)
    parser.add_argument("--target-recall", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    machine = load_json(REPO_ROOT / "profiles" / "m1-max-64gb" / "machine.json")
    memory_plan = load_json(REPO_ROOT / "profiles" / "m1-max-64gb" / "resident-plan.json")
    revision = load_json(
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    result = recommend(
        machine,
        memory_plan,
        manifest,
        sim_rows=maybe_json(args.sim),
        shadow0=maybe_json(args.shadow0),
        shadow1=maybe_json(args.shadow1),
        target_recall=args.target_recall,
    )
    print("key                       value")
    for key, value in result["config"].items():
        print(f"{key:<25} {value}")
    print("rationale:")
    for key, why in result["rationale"].items():
        print(f"  {key}: {why}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
