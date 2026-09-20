#!/usr/bin/env python3
"""Print the analytic throughput table with scope and quality labels.

Default output contains only lossless-compatible sparsity=0 scenarios.
Use --include-quality-changing only to inspect counterfactual neuron-skipping
scenarios; those MUST NOT be used as evidence for the project quality target.

Batch B means concurrent sequences.  Aggregate tok/s and per-sequence tok/s
are printed separately.  MTP rows with zero verification traffic are explicit
optimistic upper bounds until real verification cost is measured.
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
    per_sequence_tokens_per_second,
    required_batch,
    required_sparsity,
    tokens_per_second,
)

SAFE_SCENARIOS = [
    (1, 0.0, 0.0, "baseline"),
    (1, 1.0, 0.0, "+mtp1 optimistic"),
    (1, 2.0, 0.0, "+mtp2 optimistic"),
    (8, 0.0, 0.0, "+batch8 aggregate"),
    (8, 2.0, 0.0, "+batch8+mtp2 optimistic"),
    (32, 0.0, 0.0, "+batch32 aggregate"),
    (32, 2.0, 0.0, "+batch32+mtp2 optimistic"),
    (128, 2.0, 0.0, "+batch128+mtp2 optimistic"),
]

QUALITY_CHANGING_SCENARIOS = [
    (1, 0.0, 0.5, "UNVERIFIED sparsity50"),
    (1, 0.0, 0.7, "UNVERIFIED sparsity70"),
    (1, 2.0, 0.5, "UNVERIFIED mtp2+sparsity50"),
    (8, 2.0, 0.5, "UNVERIFIED b8+mtp2+sparsity50"),
    (32, 2.0, 0.5, "UNVERIFIED b32+mtp2+sparsity50"),
    (128, 2.0, 0.5, "UNVERIFIED b128+mtp2+sparsity50"),
    (128, 2.0, 0.7, "UNVERIFIED b128+mtp2+sparsity70"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analytic throughput table.")
    parser.add_argument("--target", type=float, default=400.0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--include-quality-changing",
        action="store_true",
        help="include hypothetical neuron-skipping scenarios",
    )
    args = parser.parse_args()

    revision = load_json(
        REPO_ROOT
        / "upstream"
        / "DeepSeek-V4.1-Flash"
        / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    manifest = load_json(
        REPO_ROOT
        / "profiles"
        / "deepseek-v4.1-flash"
        / revision
        / "model-manifest.json"
    )
    subs = manifest["totals"]["by_subsystem"]
    dense = sum(
        subs[k]["bytes"]
        for k in (
            "attention",
            "shared-expert",
            "lm-head",
            "router",
            "norm",
            "kv-index",
        )
    )
    params = {
        "n_layers": 40,
        "n_routed": 384,
        "top_k": 6,
        "expert_bytes": 18_800_640,
    }
    bandwidth = 132.7 * 1024**3

    scenarios = list(SAFE_SCENARIOS)
    if args.include_quality_changing:
        scenarios += QUALITY_CHANGING_SCENARIOS

    print("B    a    s    aggregate    per-seq    GiB/fwd  quality       label")
    rows = []
    for batch, accept, sparsity, label in scenarios:
        flow = bytes_per_forward(batch, dense, sparsity=sparsity, **params)
        aggregate_rate = tokens_per_second(batch, accept, flow, bandwidth)
        sequence_rate = per_sequence_tokens_per_second(
            accept, flow, bandwidth
        )
        quality = (
            "lossless-model"
            if sparsity == 0.0
            else "UNVERIFIED"
        )
        rows.append(
            {
                "batch": batch,
                "mtp_accept": accept,
                "sparsity": sparsity,
                "aggregate_tok_s": round(aggregate_rate, 1),
                "per_sequence_tok_s": round(sequence_rate, 2),
                "gib_per_forward": round(flow["total_bytes"] / 2**30, 1),
                "quality": quality,
                "mtp_cost": (
                    "optimistic-zero-verify-cost"
                    if accept > 0
                    else "none"
                ),
                "label": label,
            }
        )
        print(
            f"{batch:<4} {accept:<4.0f} {sparsity:<4.1f} "
            f"{aggregate_rate:>9.1f}  {sequence_rate:>8.2f}  "
            f"{flow['total_bytes'] / 2**30:>8.1f}  "
            f"{quality:<12} {label}"
        )

    safe_batch = required_batch(
        args.target,
        2.0,
        0.0,
        dense,
        bandwidth_bps=bandwidth,
        **params,
    )
    print()
    print(
        f"aggregate batch for {args.target:g} tok/s with sparsity=0 "
        f"and optimistic a=2: {safe_batch}"
    )
    if safe_batch is not None:
        safe_flow = bytes_per_forward(safe_batch, dense, **params)
        safe_seq = per_sequence_tokens_per_second(
            2.0, safe_flow, bandwidth
        )
        print(
            f"per-sequence rate at that batch: {safe_seq:.2f} tok/s "
            "(not 400 tok/s for one conversation)"
        )

    if args.include_quality_changing:
        hypothetical_batch = required_batch(
            args.target,
            2.0,
            0.5,
            dense,
            bandwidth_bps=bandwidth,
            **params,
        )
        hypothetical_sparsity = required_sparsity(
            args.target,
            32,
            2.0,
            dense,
            bandwidth_bps=bandwidth,
            **params,
        )
        print(
            f"UNVERIFIED quality-changing counterfactual: batch="
            f"{hypothetical_batch} at sparsity=0.5"
        )
        print(
            f"UNVERIFIED quality-changing counterfactual: sparsity="
            f"{hypothetical_sparsity} at B=32,a=2"
        )

    print()
    print(
        "IMPORTANT: B>1 numbers are aggregate serving throughput. "
        "Non-zero sparsity is outside the lossless claim until exact "
        "equivalence is proved. MTP rows assume zero verification cost "
        "unless that cost is explicitly supplied elsewhere."
    )

    if args.json_out:
        args.json_out.write_text(
            json.dumps(rows, indent=1) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
