#!/usr/bin/env python3
"""Issue #26: Engram row-cache sweep on real lookup tables.

    python scripts/run_engram_cache.py --budgets 16,64,256 --pages 4096,16384 --ops 2000
    python scripts/run_engram_cache.py --budgets 64 --pages 4096 --ops 2000 --json-out e.json

Every served row is validated byte-equal against direct preads on a
sample (exactness gate); the rest compare hit/miss/bytes across
budgets and granularities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.engram import row_ranges, table_refs_from_manifest
from deepseeker.engram_cache import EngramRowCache, access_stream


def direct_row(model_root: Path, ref, row_id: int) -> tuple[bytes, bytes]:
    info = row_ranges(ref, row_id)
    out = []
    for part in ("weight", "scale"):
        shard = info[part]["shard"]
        start, end = info[part]["range"]
        fd = os.open(model_root / shard, os.O_RDONLY)
        try:
            data = os.pread(fd, end - start, start)
        finally:
            os.close(fd)
        out.append(data)
    return out[0], out[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Engram row-cache sweep.")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--budgets", type=str, default="16,64,256")
    parser.add_argument("--pages", type=str, default="4096,16384")
    parser.add_argument("--ops", type=int, default=2000)
    parser.add_argument("--passes", type=int, default=1,
                        help="repeat the stream (recurrent context: 2nd pass measures reuse)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    model_root = args.model_root or REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    revision = load_json(
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    refs = table_refs_from_manifest(manifest)
    if args.layer not in refs:
        print(f"FAIL: no engram table for layer {args.layer}")
        return 2
    ref = refs[args.layer]
    stream = access_stream(ref.num_rows, args.ops, args.seed)

    print(f"layer={args.layer} rows={ref.num_rows} ops={args.ops}")
    print("budget page    hit_rate  read_MB  ampl   exact")
    rows = []
    for budget_mib in [int(b) for b in args.budgets.split(",") if b.strip()]:
        for page in [int(p) for p in args.pages.split(",") if p.strip()]:
            cache = EngramRowCache(model_root, refs, budget_mib * 1024**2, page)
            start = time.perf_counter()
            exact = True
            try:
                for _ in range(args.passes):
                    for i in range(0, len(stream), 64):
                        batch = stream[i : i + 64]
                        got = cache.fetch_rows(args.layer, batch)
                        if i == 0:
                            for row_id, (w, s) in zip(batch[:4], got[:4]):
                                dw, ds = direct_row(model_root, ref, row_id)
                                if w != dw or s != ds:
                                    exact = False
            finally:
                cache.shutdown()
            elapsed = time.perf_counter() - start
            metrics = cache.metrics()
            useful_mb = args.ops * args.passes * 264 / 1024**2
            ampl = metrics["bytes_read"] / 1024**2 / useful_mb if useful_mb else 0.0
            print(f"{budget_mib:>6} {page:>5}  {metrics['hit_rate']:>8.3f}  "
                  f"{metrics['bytes_read'] / 1024**2:>7.1f}  {ampl:>5.2f}x  {exact}  "
                  f"({elapsed:.1f}s)")
            rows.append({"budget_mib": budget_mib, "page_bytes": page,
                         "elapsed_s": round(elapsed, 1), "exact": exact, **metrics})
            if not exact:
                print("FAIL: row bytes differ from direct reads")
                return 1
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
