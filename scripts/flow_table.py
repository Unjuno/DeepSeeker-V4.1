#!/usr/bin/env python3
"""Print the analytic cheat table: tok/s over batch/MTP-accept/sparsity.

    python scripts/flow_table.py
    python scripts/flow_table.py --target 400 --json-out flow.json

Constants (fp4 bytes) come from the Issue #2 manifest; bandwidth from the
machine profile. No experiments: closed form only (see docs/FLOWMAP.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.flowmap import (
    bytes_per_forward,
    required_batch,
    required_sparsity,
    tokens_per_second,
)

CHEATS = [
    (1, 0.0, 0.0, "baseline"),
    (1, 0.0, 0.5, "+sparsity50"),
    (1, 0.0, 0.7, "+sparsity70"),
    (1, 1.0, 0.0, "+mtp1"),
    (1, 2.0, 0.0, "+mtp2"),
    (1, 2.0, 0.5, "+mtp2+spar50"),
    (8, 0.0, 0.0, "+batch8"),
    (8, 2.0, 0.5, "+b8+mtp2+sp50"),
    (32, 0.0, 0.0, "+batch32"),
    (32, 2.0, 0.5, "+b32+mtp2+sp50"),
    (128, 2.0, 0.5, "+b128+mtp2+sp50"),
    (128, 2.0, 0.7, "+b128+mtp2+sp70"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analytic cheat table.")
    parser.add_argument("--target", type=float, default=400.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    revision = load_json(
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    subs = manifest["totals"]["by_subsystem"]
    dense = sum(subs[k]["bytes"] for k in ("attention", "shared-expert", "lm-head", "router", "norm", "kv-index"))
    params = {"n_layers": 40, "n_routed": 384, "top_k": 6, "expert_bytes": 18800640}
    bandwidth = 132.7 * 1024**3

    print("B    a    s    tok/s    GiB/fwd  label")
    rows = []
    for batch, accept, sparsity, label in CHEATS:
        flow = bytes_per_forward(batch, dense, sparsity=sparsity, **params)
        rate = tokens_per_second(batch, accept, flow, bandwidth)
        rows.append(
            {"batch": batch, "mtp_accept": accept, "sparsity": sparsity,
             "tok_s": round(rate, 1), "gib_per_forward": round(flow["total_bytes"] / 2**30, 1),
             "label": label}
        )
        print(
            f"{batch:<4} {accept:<4.0f} {sparsity:<4.1f} {rate:>7.1f}  "
            f"{flow['total_bytes'] / 2**30:>8.1f}  {label}"
        )
    print(f"batch for {args.target:g} tok/s (a=2,s=0.5): "
          f"{required_batch(args.target, 2.0, 0.5, dense, bandwidth_bps=bandwidth, **params)}")
    print(f"batch for 2000 tok/s (a=2,s=0.7): "
          f"{required_batch(2000, 2.0, 0.7, dense, bandwidth_bps=bandwidth, **params)}")
    print(f"sparsity for {args.target:g} tok/s @B=32,a=2: "
          f"{required_sparsity(args.target, 32, 2.0, dense, bandwidth_bps=bandwidth, **params)}")
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
