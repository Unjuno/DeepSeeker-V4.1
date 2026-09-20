#!/usr/bin/env python3
"""Issue #14: mine routing motifs + speculative-block working sets.

    python scripts/analyze_motifs.py --trace /tmp/t.jsonl --blocks 1,2,4,8 --top-motifs 10
    python scripts/analyze_motifs.py --trace /tmp/t.jsonl --json-out motifs.json

Motif coverage answers whether a small bigram table captures routing;
the reuse curve answers whether multi-token blocks share experts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.motif import block_summary, layer_bigrams, motif_coverage
from deepseeker.trace import read_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine routing motifs.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--blocks", type=str, default="1,2,4,8")
    parser.add_argument("--top-motifs", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        header, records = read_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    sizes = [int(b) for b in args.blocks.split(",") if b.strip()]
    if any(s < 1 for s in sizes):
        print("FAIL: block sizes must be >= 1")
        return 2

    pairs = layer_bigrams(records)
    top = pairs.most_common(args.top_motifs)
    motif_set = {pair for pair, _ in top}
    coverage = motif_coverage(records, motif_set)
    print(f"trace: {args.trace} ({len(records)} records, {len(pairs)} distinct bigrams)")
    print(f"top-{args.top_motifs} bigram coverage: {coverage:.3f}")
    for (src, dst), count in top[: min(5, len(top))]:
        print(f"  L{src[0]}:e{src[1]} -> L{dst[0]}:e{dst[1]}  x{count}")
    print("block  mean_unique  reuse_ratio")
    curve = block_summary(records, sizes, header["top_k"])
    for row in curve:
        print(
            f"{row['block_size']:>5}  {row['mean_unique_experts']:>11.2f}  "
            f"{row['reuse_ratio']:>11.3f}"
        )
    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                {
                    "top_motifs": [
                        {"src": src, "dst": dst, "count": c} for (src, dst), c in top
                    ],
                    "coverage": coverage,
                    "reuse_curve": curve,
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
