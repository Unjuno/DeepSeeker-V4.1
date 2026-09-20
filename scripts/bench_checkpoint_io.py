#!/usr/bin/env python3
"""Issue #19: cold random-access service on the real verified checkpoint.

    python scripts/bench_checkpoint_io.py --qd 1,2,4,8,16 --jobs 8 --out /tmp/cio.json

Reads real expert/Engram file_ranges (O_RDONLY, never writes) at QD
1-16 in warm + advisory-cold (F_NOCACHE) classes, reports p50/p95 per
range, throughput, and a sustained group workload. Purge-controlled
cold is unavailable without sudo: stays UNCERTAIN by design.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.checkpoint_io import (
    engram_ranges,
    flatten_ranges,
    merge_group,
    set_nocache,
    summarize_latency,
)

CLASSES = ("warm", "advisory-cold")


def open_shards(root: Path, shards: list[str], nocache: bool) -> dict[str, int]:
    fds = {}
    for shard in shards:
        fd = os.open(root / shard, os.O_RDONLY)
        if nocache and not set_nocache(fd):
            os.close(fd)
            raise RuntimeError("F_NOCACHE unsupported on this machine")
        fds[shard] = fd
    return fds


def run_jobs(fds: dict[str, int], jobs: list[tuple[str, int, int]]) -> tuple[list[float], int]:
    """Execute pread jobs; return (per-job latencies, total bytes)."""
    latencies: list[float] = []
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        def one(job: tuple[str, int, int]) -> tuple[float, int]:
            shard, start, end = job
            size = end - start
            begin = time.perf_counter()
            data = os.pread(fds[shard], size, start)
            elapsed = time.perf_counter() - begin
            if len(data) != size:
                raise RuntimeError(f"short read {shard}@{start}: {len(data)}/{size}")
            return elapsed, size

        for elapsed, size in pool.map(one, jobs):
            latencies.append(elapsed)
            total += size
    return latencies, total


def sweep_rounds(
    root: Path,
    name: str,
    rounds: list[list[tuple[str, int, int]]],
    qds: list[int],
    cache_class: str,
) -> dict:
    """QD matrix over prebuilt per-round job lists (distinct ranges/round)."""
    if not rounds or not all(rounds):
        raise ValueError(f"empty workload: {name}")
    shards = sorted({shard for jobs in rounds for shard, _, _ in jobs})
    fds = open_shards(root, shards, nocache=cache_class == "advisory-cold")
    try:
        rows = []
        for qd in qds:
            lat_all: list[float] = []
            bytes_all = 0
            wall = 0.0
            for jobs in rounds:
                batch = (jobs * ((qd // len(jobs)) + 1))[:qd]
                begin = time.perf_counter()
                lat, nbytes = run_jobs(fds, batch)
                wall += time.perf_counter() - begin
                lat_all.extend(lat)
                bytes_all += nbytes
            summary = summarize_latency(lat_all)
            summary.update(
                {
                    "qd": qd,
                    "ranges_per_round": len(batch),
                    "rounds": len(rounds),
                    "bytes_total": bytes_all,
                    "wall_s": round(wall, 3),
                    "throughput_mib_s": round(bytes_all / wall / 1024**2, 1) if wall else 0.0,
                }
            )
            rows.append(summary)
        return {"workload": name, "cache_class": cache_class, "qd_rows": rows}
    finally:
        for fd in fds.values():
            os.close(fd)


def sweep(
    root: Path,
    name: str,
    grouped: dict[str, list[tuple[str, int, int]]],
    qds: list[int],
    repeats: int,
    cache_class: str,
) -> dict:
    jobs = flatten_ranges(grouped)
    return sweep_rounds(root, name, [jobs] * repeats, qds, cache_class)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark real checkpoint IO.")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--qd", type=str, default="1,2,4,8,16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sustained-rounds", type=int, default=120)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.model_root or REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    revision = load_json(
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    qds = [int(q) for q in args.qd.split(",") if q.strip()]
    if any(q < 1 for q in qds) or args.repeats < 1:
        print("FAIL: qd and repeats must be >= 1")
        return 2

    # Distinct ranges per repeat: re-reading the same range measures page
    # cache, not storage. Variants use disjoint experts/offsets so every
    # first touch is effectively cold (475GB vs 64GB RAM). Repeats beyond
    # the variant list cycle (documented in output).
    expert_variants = [[(5, 7)], [(15, 13)], [(25, 29)]]
    same_layer_variants = [
        [(5, e) for e in range(6)],
        [(15, e) for e in range(6, 12)],
        [(25, e) for e in range(12, 18)],
    ]
    cross_layer_variants = [
        [(layer, 3) for layer in range(5, 11)],
        [(layer, 9) for layer in range(15, 21)],
        [(layer, 15) for layer in range(25, 31)],
    ]
    engram_offsets = [0, 16 * 1024**2, 32 * 1024**2]
    plan: list[tuple[str, list[list[tuple[str, int, int]]]]] = [
        ("single-expert", [flatten_ranges(merge_group(manifest, v)) for v in expert_variants]),
        ("same-layer-group", [flatten_ranges(merge_group(manifest, v)) for v in same_layer_variants]),
        ("cross-layer-group", [flatten_ranges(merge_group(manifest, v)) for v in cross_layer_variants]),
        ("engram-batch", [
            flatten_ranges(engram_ranges(manifest, 4, base_offset=off)) for off in engram_offsets
        ]),
    ]
    results = []
    for name, variants in plan:
        # Advisory-cold FIRST per workload: warm second measures cached.
        for cls in ("advisory-cold", "warm"):
            print(f"{name} [{cls}]...", flush=True)
            rounds = [variants[i % len(variants)] for i in range(args.repeats)]
            results.append(sweep_rounds(root, name, rounds, qds, cls))

    sustained_group = merge_group(manifest, [(10, e) for e in range(6)])
    print(f"sustained same-layer-group QD8 x{args.sustained_rounds}...", flush=True)
    jobs = flatten_ranges(sustained_group)
    shards = sorted({shard for shard, _, _ in jobs})
    fds = open_shards(root, shards, nocache=True)
    try:
        round_walls = []
        for _ in range(args.sustained_rounds):
            begin = time.perf_counter()
            run_jobs(fds, jobs)
            round_walls.append(time.perf_counter() - begin)
        half = len(round_walls) // 2
        first = summarize_latency(round_walls[:half])
        second = summarize_latency(round_walls[half:])
        sustained = {
            "workload": "sustained-group",
            "cache_class": "advisory-cold",
            "rounds": len(round_walls),
            "qd": 8,
            "first_half_s_per_round": first,
            "second_half_s_per_round": second,
            "drift_ratio": second["mean_s"] / first["mean_s"] if first["mean_s"] else None,
        }
    finally:
        for fd in fds.values():
            os.close(fd)
    results.append({"workload": "sustained-group", "cache_class": "advisory-cold",
                    "qd_rows": [], "sustained": sustained})

    payload = {
        "schema": "deepseeker.checkpoint-io/v1",
        "revision": revision,
        "qds": qds,
        "repeats": args.repeats,
        "purge_class": "UNCERTAIN (no sudo; F_NOCACHE advisory only)",
        "workloads": results,
    }
    for entry in results:
        if "sustained" in entry:
            s = entry["sustained"]
            print(f"  sustained/{entry['cache_class']}: rounds={s['rounds']} "
                  f"first_half={s['first_half_s_per_round']['mean_s']:.3f}s/round "
                  f"second_half={s['second_half_s_per_round']['mean_s']:.3f}s/round "
                  f"drift={s['drift_ratio']:.3f}" if s["drift_ratio"] else "")
            continue
        for row in entry["qd_rows"]:
            print(f"  {entry['workload']}/{entry['cache_class']}/QD{row['qd']}: "
                  f"p50={row['p50_s']*1000:.2f}ms p95={row['p95_s']*1000:.2f}ms "
                  f"{row['throughput_mib_s']}MiB/s")
    if args.out:
        args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
