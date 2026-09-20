#!/usr/bin/env python3
"""Issue #15: print KV/index cache sizes and the unified P5 budget table.

    python scripts/size_kv.py --batch 1 --seqs 4096,32768,131072
    python scripts/size_kv.py --batch 1 --seqs 32768 --slots 12 --json-out kv.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.kvmodel import cache_bytes, cache_geometry, unified_table


def mib(n: float) -> str:
    return f"{n / 1024 / 1024:,.1f}MB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Size KV caches + unified budget.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqs", type=str, default="4096,32768,131072")
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.batch < 1:
        print("FAIL: batch must be >= 1")
        return 2

    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    config = load_json(snapshot / "inference" / "config.json")
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    plan = load_json(REPO_ROOT / "profiles" / "m1-max-64gb" / "resident-plan.json")
    geometry = cache_geometry(config)
    expert_bytes = int(manifest["expert_stats"]["median_bytes"])
    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]

    print(f"batch={args.batch} kv_sources={geometry['kv_source_layers']}")
    print("seq     window    compress  index     total     per_tok")
    tables = []
    for seq in seqs:
        try:
            kv = cache_bytes(geometry, args.batch, seq)
        except ValueError as exc:
            print(f"FAIL: {exc}")
            return 2
        tables.append({"seq_len": seq, **kv})
        print(
            f"{seq:<7} {mib(kv['window_bytes']):>8}  {mib(kv['compress_bytes']):>8}  "
            f"{mib(kv['index_bytes']):>7}  {mib(kv['total_bytes']):>8}  "
            f"{kv['per_token_bytes']:>8,.0f}B"
        )
    biggest = tables[-1]
    unified = unified_table(
        common_model_bytes=int(plan["common_model_bytes"]["bytes"]),
        dense_engram_bytes=int(plan["dense_engram_bytes"]["bytes"]),
        expert_slots_per_layer=args.slots,
        n_layers=geometry["n_layers"],
        expert_bytes=expert_bytes,
        kv=biggest,
        operating_bytes=int(plan["operating_bytes"]),
    )
    print(f"unified @seq={biggest['seq_len']} slots/layer={args.slots}:")
    for name, value in unified["components"].items():
        print(f"  {name:<14} {mib(value):>12}")
    print(
        f"  total {mib(unified['total_bytes'])} vs operating "
        f"{mib(unified['operating_bytes'])} -> headroom {mib(unified['headroom_bytes'])}"
    )
    print("OVER BUDGET" if unified["over_budget"] else "within budget")
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"kv_tables": tables, "unified": unified}, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
