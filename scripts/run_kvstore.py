#!/usr/bin/env python3
"""Issue #27: KV/index allocator sweep with reconciliation.

    python scripts/run_kvstore.py --seqs 4096,32768,131072,524288
    python scripts/run_kvstore.py --seqs 4096 --stress --json-out kv.json

Allocates real buffers per context, grows 4K->32K, cycles reset/reuse
across requests, attempts a higher context, and probes over-budget for
a clean BudgetExceeded (no partial state).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.kvstore import BudgetExceeded, KVStore


def mib(n: float) -> str:
    return f"{n / 1024**2:,.1f}MB"


def main() -> int:
    parser = argparse.ArgumentParser(description="KV allocator sweep.")
    parser.add_argument("--seqs", type=str, default="4096,32768,131072,524288")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--budget-mib", type=int, default=4096)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    config = load_json(snapshot / "inference" / "config.json")
    store = KVStore(config, args.budget_mib * 1024**2)
    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]
    rows = []
    print(f"budget={args.budget_mib}MiB batch={args.batch}")
    for seq in seqs:
        start = time.perf_counter()
        info = store.alloc("sweep", args.batch, seq)
        alloc_s = time.perf_counter() - start
        recon = store.reconcile("sweep")
        status = "MATCH" if recon["match"] else "MISMATCH"
        print(f"seq={seq:<7} {mib(info['bytes']):>10} alloc={alloc_s * 1000:>7.1f}ms {status}")
        rows.append({"seq_len": seq, "bytes": info["bytes"],
                     "alloc_ms": round(alloc_s * 1000, 1), **recon})
        if not recon["match"]:
            print("FAIL: model-vs-actual drift")
            return 1
        store.free("sweep")

    # grow 4K -> 32K, then reset/reuse across 3 requests
    store.alloc("grow", args.batch, 4096)
    grown = store.grow("grow", 32768)
    store.write_pattern("grow", "window", 7)
    pattern_ok = store.read_sum("grow", "window") == 7 * store._requests["grow"]["buffers"]["window"].size
    store.free("grow")
    for i in range(3):
        store.alloc(f"req{i}", args.batch, 4096)
    store.reset()
    reuse_ok = store.allocated_bytes == 0 and store.telemetry()["live_requests"] == 0
    print(f"grow 4K->32K: {mib(grown['bytes'])}  pattern_roundtrip={pattern_ok}  reset_clean={reuse_ok}")
    if not (pattern_ok and reuse_ok):
        print("FAIL: lifecycle correctness")
        return 1

    stress = None
    if args.stress:
        try:
            store.alloc("huge", args.batch, 8 * 1024**3)
            print("FAIL: over-budget alloc succeeded")
            return 1
        except (BudgetExceeded, ValueError, MemoryError) as exc:
            stress = f"clean failure: {type(exc).__name__}"
            assert store.allocated_bytes == 0, "partial state after failure!"
        print(stress)
    tele = store.telemetry()
    print(f"high_water={mib(tele['high_water_bytes'])} allocations={tele['allocations']}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"rows": rows, "grow_bytes": grown["bytes"], "pattern_ok": pattern_ok,
             "reuse_ok": reuse_ok, "stress": stress, "telemetry": tele}, indent=1) + "\n",
            encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
