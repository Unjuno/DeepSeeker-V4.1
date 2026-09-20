"""Issue #19: cold random-access measurement on the real checkpoint.

Pure helpers; threads and file IO live in scripts/bench_checkpoint_io.py.
Workloads are built from Issue #2 manifest file_ranges (never invent
offsets). Cache-confidence classes:

  warm           plain O_RDONLY fd, repeated reads (page-cache served)
  advisory-cold  F_NOCACHE fd (macOS fcntl, no sudo; bypasses page cache
                 on read — advisory, hence the name)
  purge          privileged cache purge: unavailable without sudo, so any
                 claim needing it stays UNCERTAIN (issue spec).
"""

from __future__ import annotations

import statistics

SCHEMA = "deepseeker.checkpoint-io/v1"
F_NOCACHE = 48  # macOS fcntl(2); set guarded by hasattr at call sites


def set_nocache(fd: int) -> bool:
    """Enable F_NOCACHE on fd; False when unsupported (caller labels)."""
    import fcntl

    if not hasattr(fcntl, "F_NOCACHE"):
        return False
    try:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        return True
    except OSError:
        return False


def expert_ranges(manifest: dict, layer: int, expert: int) -> dict[str, list[tuple[int, int]]]:
    """{shard: [(start, end)]} for every tensor of one routed expert."""
    out: dict[str, list[tuple[int, int]]] = {}
    for tensor in manifest.get("tensors", []):
        if (
            tensor.get("subsystem") == "routed-expert"
            and tensor.get("layer") == layer
            and tensor.get("expert") == expert
        ):
            out.setdefault(tensor["shard"], []).append(tuple(tensor["file_range"]))
    for ranges in out.values():
        ranges.sort()
    return out


def flatten_ranges(grouped: dict[str, list[tuple[int, int]]]) -> list[tuple[str, int, int]]:
    """Flatten to (shard, start, end) jobs in shard order."""
    jobs = []
    for shard in sorted(grouped):
        jobs.extend((shard, start, end) for start, end in grouped[shard])
    return jobs


def merge_group(
    manifest: dict, specs: list[tuple[int, int]]
) -> dict[str, list[tuple[int, int]]]:
    """Union of expert_ranges over (layer, expert) specs (a prefetch group)."""
    merged: dict[str, list[tuple[int, int]]] = {}
    for layer, expert in specs:
        for shard, ranges in expert_ranges(manifest, layer, expert).items():
            merged.setdefault(shard, []).extend(ranges)
    for ranges in merged.values():
        ranges.sort()
    return merged


def engram_ranges(
    manifest: dict, limit: int, max_bytes: int = 16 * 1024**2, base_offset: int = 0
) -> dict[str, list[tuple[int, int]]]:
    """First `limit` engram tensors from `base_offset`, clipped to `max_bytes`.

    Full engram tensors are ~17GB; the clip keeps the benchmark bounded
    while preserving the real file offsets (page-pattern input).
    Distinct base_offsets per repeat keep first-touch reads cold.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    count = 0
    for tensor in manifest.get("tensors", []):
        if tensor.get("subsystem") != "engram":
            continue
        if count >= limit:
            break
        start, end = tensor["file_range"]
        lo = min(end, start + base_offset)
        out.setdefault(tensor["shard"], []).append((lo, min(end, lo + max_bytes)))
        count += 1
    return out


def summarize_latency(samples_s: list[float]) -> dict:
    """p50/p95/min/max over per-range latencies."""
    if not samples_s:
        raise ValueError("no latency samples")
    ordered = sorted(samples_s)
    n = len(ordered)
    def pick(q: float) -> float:
        return ordered[min(n - 1, int(q * n))]
    return {
        "n": n,
        "p50_s": pick(0.50),
        "p95_s": pick(0.95),
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "mean_s": statistics.fmean(ordered),
    }
