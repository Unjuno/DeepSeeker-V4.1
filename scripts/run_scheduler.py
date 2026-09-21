#!/usr/bin/env python3
"""Issue #28: A/B independent vs joint scheduling on a mixed workload.

    python scripts/run_scheduler.py --trace /tmp/t.jsonl --ctx 32768 --budget-mib 8192
    python scripts/run_scheduler.py --trace /tmp/t.jsonl --ctx 32768 --budget-mib 8192 \\
        --prefetch gated --max-tokens 4 --json-out sched.json

Mixed per token: expert serves (pool) + Engram rows (cache) + one KV
context. At halfway, KV grows 4x: joint steals from Engram/experts,
independent holds partitions (growth may block). Stalls = expert +
Engram misses served from storage.
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
from deepseeker.engram import table_refs_from_manifest
from deepseeker.engram_cache import EngramRowCache, access_stream
from deepseeker.kvmodel import cache_geometry
from deepseeker.kvstore import BudgetExceeded, KVStore
from deepseeker.pool import ResidentPool, load_expert_bytes
from deepseeker.prefetch import PrefetchController
from deepseeker.scheduler import UnifiedScheduler
from deepseeker.stage import StagingEngine
from deepseeker.trace import read_trace


def run_arm(token_groups, refs, manifest, model_root, config, ctx, budget_mib,
            mode, prefetch_mode, max_blob, expert_bytes):
    geometry = cache_geometry(config)
    sched = UnifiedScheduler(budget_mib * 1024**2, geometry, expert_bytes, 40, mode=mode)
    plan = sched.reserve(1, ctx)
    pool = ResidentPool(plan["expert_slots_per_layer"], max_blob)
    engram = EngramRowCache(model_root, refs, max(plan["engram_bytes"], 4096), 4096)
    kv_budget = plan["kv_bytes"] if mode == "independent" else budget_mib * 1024**2
    kv = KVStore(config, kv_budget)
    kv.alloc("req", 1, ctx)
    engine = StagingEngine(model_root, manifest, max_workers=4)
    controller = None
    if prefetch_mode == "gated":
        controller = PrefetchController(pool, engine, enabled=True, min_agreement=0.0)
    stream = access_stream(refs[1].num_rows, max(64, len(token_groups) * 8), seed=5)
    stalls = 0
    engram_uses = 0
    grown = False
    blocked_growth = 0
    freed_total = {"engram": 0, "expert_slots": 0}
    start = time.perf_counter()
    try:
        for i, records in enumerate(token_groups):
            if i == len(token_groups) // 2 and not grown:
                grown = True
                old_kv = kv.reconcile("req")["actual_bytes"]
                used = (pool.metrics()["bytes_resident"]
                        + engram.metrics()["bytes_resident"] + old_kv)
                try:
                    kv.grow("req", ctx * 4)
                    need = kv.reconcile("req")["actual_bytes"] - old_kv
                    freed = sched.steal_for_kv(need, used)
                    freed_total["engram"] += freed["engram"]
                    freed_total["expert_slots"] += freed["expert_slots"]
                    for layer in range(40):
                        pool.set_cap(layer, sched.metrics()["expert_slots_per_layer"])
                    engram.resize_capacity(max(4096, sched.engram_reservation))
                except (BudgetExceeded, RuntimeError, ValueError):
                    blocked_growth += 1
            for record in sorted(records, key=lambda r: r["layer"]):
                for expert in record["experts"]:
                    if controller is not None:
                        missed_before = pool.metrics()["misses"]
                        out = controller.serve(
                            record["layer"], expert,
                            lambda layer, expert: load_expert_bytes(
                                model_root, manifest, layer, expert))
                        stalls += pool.metrics()["misses"] - missed_before
                    else:
                        hit = pool.lookup(record["layer"], expert)
                        if hit is None:
                            stalls += 1
                            out = pool.demand(
                                record["layer"], expert,
                                lambda layer, expert: load_expert_bytes(
                                    model_root, manifest, layer, expert))
                        else:
                            out = bytes(hit)
                    del out
            rows = stream[i * 8:(i + 1) * 8]
            for w, _s in engram.fetch_rows(1, rows):
                engram_uses += 1
                del w
            if controller is not None:
                controller.observe_token(records)
    finally:
        engine.shutdown()
        engram.shutdown()
    elapsed = time.perf_counter() - start
    eng_m = engram.metrics()
    return {
        "stalls": stalls + eng_m["page_misses"],
        "expert_stalls": stalls,
        "engram_misses": eng_m["page_misses"],
        "engram_hit_rate": eng_m["hit_rate"],
        "blocked_growth": blocked_growth,
        "freed": freed_total,
        "expert_slots": sched.metrics()["expert_slots_per_layer"],
        "elapsed_s": round(elapsed, 2),
        "scheduler": sched.metrics(),
        "pool": pool.metrics(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B unified scheduling.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--budget-mib", type=int, default=8192)
    parser.add_argument("--mode", type=str, default="joint")
    parser.add_argument("--prefetch", type=str, default="demand")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        _header, records = read_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    if args.mode == "joint":
        modes = ("independent", "joint")
    else:
        modes = (args.mode,)
    model_root = args.model_root or REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    config = load_json(snapshot / "inference" / "config.json")
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    refs = table_refs_from_manifest(manifest)
    expert_bytes = int(manifest["expert_stats"]["median_bytes"])
    per_expert: dict[tuple[int, int], int] = {}
    for tensor in manifest["tensors"]:
        if tensor.get("subsystem") == "routed-expert":
            key = (tensor["layer"], tensor["expert"])
            per_expert[key] = per_expert.get(key, 0) + tensor["bytes"]
    groups: dict[tuple[str, int], list] = {}
    for record in records:
        groups.setdefault((record["request_id"], record["token_pos"]), []).append(record)
    tokens = [groups[k] for k in sorted(groups)]
    if args.max_tokens:
        tokens = tokens[: args.max_tokens]

    results = {}
    for mode in modes:
        print(f"mode={mode} prefetch={args.prefetch}...", flush=True)
        results[mode] = run_arm(tokens, refs, manifest, model_root, config, args.ctx,
                                args.budget_mib, mode, args.prefetch,
                                max(per_expert.values()), expert_bytes)
        row = results[mode]
        print(f"  stalls={row['stalls']} (expert={row['expert_stalls']}, "
              f"engram_miss={row['engram_misses']}) blocked_growth={row['blocked_growth']} "
              f"slots={row['expert_slots']} elapsed={row['elapsed_s']}s")
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
