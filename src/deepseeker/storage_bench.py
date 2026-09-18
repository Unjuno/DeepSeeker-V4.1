"""Reusable physical-storage benchmark primitives.

Range generation from the Issue #2 tensor manifest, page alignment,
gap coalescing, and a threaded pread executor with queue-depth control.

Cache-confidence discipline: nothing here claims cold/proven. Results
are labeled by the caller; the safe default is "advisory-uncached".
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import os
import statistics
import time
from dataclasses import dataclass

F_NOCACHE = 48  # macOS fcntl, advisory only


@dataclass(frozen=True)
class Range:
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


def align_down(value: int, granularity: int) -> int:
    return (value // granularity) * granularity


def align_up(value: int, granularity: int) -> int:
    return ((value + granularity - 1) // granularity) * granularity


def align_ranges(ranges: list[Range], granularity: int) -> list[Range]:
    """Expand each range to granularity-aligned boundaries."""
    return [
        Range(
            align_down(r.offset, granularity),
            align_up(r.end, granularity) - align_down(r.offset, granularity),
        )
        for r in ranges
    ]


def coalesce_ranges(ranges: list[Range], max_gap: int) -> list[Range]:
    """Sort by offset; merge neighbors separated by <= max_gap."""
    merged: list[Range] = []
    for r in sorted(ranges, key=lambda x: x.offset):
        if merged and r.offset - merged[-1].end <= max_gap:
            last = merged[-1]
            merged[-1] = Range(last.offset, max(last.end, r.end) - last.offset)
        else:
            merged.append(r)
    return merged


def summarize_latencies(samples_s: list[float]) -> dict:
    ms = sorted(s * 1000.0 for s in samples_s)
    n = len(ms)
    return {
        "samples": n,
        "min_ms": ms[0],
        "p50_ms": ms[n // 2],
        "p95_ms": ms[max(0, min(n - 1, int(n * 0.95)))],
        "max_ms": ms[-1],
        "mean_ms": sum(ms) / n,
    }


class BenchFile:
    """Physically allocated disposable benchmark file.

    Writes incompressible-feeling pseudo-random bytes once (seeded, so
    the file is reproducible without storing it), fsyncs, and serves
    pread workers. Reads use F_NOCACHE (advisory only).
    """

    def __init__(self, path: str, size_bytes: int, seed: int = 0x5EED):
        self.path = path
        self.size_bytes = size_bytes
        self.seed = seed

    def create(self, block_bytes: int = 4 * 1024 * 1024) -> dict:
        import random

        rng = random.Random(self.seed)
        chunk = bytes(rng.randrange(256) for _ in range(1024 * 1024))
        repeats = block_bytes // len(chunk)
        block = chunk * repeats
        with open(self.path, "wb", buffering=0) as f:
            try:
                fcntl.fcntl(f.fileno(), F_NOCACHE, 1)
            except OSError:
                pass
            written = 0
            while written < self.size_bytes:
                n = min(block_bytes, self.size_bytes - written)
                os.write(f.fileno(), block[:n])
                written += n
            os.fsync(f.fileno())
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "seed": self.seed,
            "method": "seeded-random single write + fsync",
        }

    def open_nocache(self) -> int:
        fd = os.open(self.path, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, F_NOCACHE, 1)
        except OSError:
            pass
        return fd


def remap_ranges(ranges: list[Range], file_size: int, salt: int = 0) -> list[Range]:
    """Replay manifest geometry on the disposable file.

    Preserves each range's length and relative spacing (so coalescing
    and alignment behave identically) while keeping every byte inside
    the benchmark file. Deterministic for a fixed salt.
    """
    if not ranges:
        return []
    base = min(r.offset for r in ranges)
    span = max(r.end for r in ranges) - base
    if span >= file_size:
        raise ValueError("pattern span exceeds benchmark file")
    # Deterministic pseudo-random placement from salt.
    import hashlib

    digest = hashlib.sha256(str(salt).encode()).digest()
    start_max = file_size - span
    start = int.from_bytes(digest[:8], "little") % (start_max + 1)
    start = align_down(start, 4096)
    return [Range(start + (r.offset - base), r.length) for r in ranges]


def scatter_ranges(ranges: list[Range], file_size: int, seed: int = 0) -> list[Range]:
    """Replay range lengths at independent uniform random offsets.

    Used when a pattern's true span exceeds any reasonable disposable
    file (e.g. Engram rows scattered over a 98 GiB table). Lengths,
    counts, and per-job grouping are preserved exactly; absolute
    placement becomes i.i.d. random and is documented as such.
    Deterministic for a fixed seed.
    """
    import random

    rng = random.Random(seed)
    out = []
    for r in ranges:
        if r.length > file_size:
            raise ValueError("single range exceeds benchmark file")
        out.append(Range(rng.randrange(0, file_size - r.length + 1), r.length))
    return sorted(out, key=lambda x: x.offset)


def run_qd(
    bench: BenchFile,
    ranges: list[Range],
    queue_depth: int,
    repeats: int = 1,
) -> dict:
    """Execute ranges at a fixed queue depth; return latency stats."""
    fds = [bench.open_nocache() for _ in range(max(1, queue_depth))]
    latencies: list[float] = []
    total_bytes = 0
    t_start = time.perf_counter()
    try:
        if queue_depth <= 1:
            for _ in range(repeats):
                for r in ranges:
                    t0 = time.perf_counter()
                    data = os.pread(fds[0], r.length, r.offset)
                    latencies.append(time.perf_counter() - t0)
                    total_bytes += len(data)
        else:
            chunks = [ranges[i::queue_depth] for i in range(queue_depth)]

            def worker(fd: int, jobs: list[Range]) -> tuple[list[float], int]:
                local: list[float] = []
                nbytes = 0
                for _ in range(repeats):
                    for r in jobs:
                        t0 = time.perf_counter()
                        data = os.pread(fd, r.length, r.offset)
                        local.append(time.perf_counter() - t0)
                        nbytes += len(data)
                return local, nbytes

            with concurrent.futures.ThreadPoolExecutor(max_workers=queue_depth) as pool:
                futures = [pool.submit(worker, fd, jobs) for fd, jobs in zip(fds, chunks)]
                for fut in concurrent.futures.as_completed(futures):
                    local, nbytes = fut.result()
                    latencies.extend(local)
                    total_bytes += nbytes
    finally:
        for fd in fds:
            os.close(fd)
    elapsed = time.perf_counter() - t_start
    stats = summarize_latencies(latencies)
    stats["total_bytes"] = total_bytes
    stats["elapsed_s"] = elapsed
    stats["throughput_mib_s"] = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0.0
    stats["queue_depth"] = queue_depth
    stats["repeats"] = repeats
    return stats


def run_sustained(
    bench: BenchFile,
    ranges: list[Range],
    queue_depth: int,
    duration_s: float,
    bucket_s: float = 10.0,
) -> dict:
    """Loop full passes over ranges until duration_s.

    Every pass (round) is timed, so per-bucket throughput is built from
    real round measurements at any queue depth.
    """
    fds = [bench.open_nocache() for _ in range(max(1, queue_depth))]
    queue_depth = max(1, queue_depth)
    chunks = [ranges[i::queue_depth] for i in range(queue_depth)]
    rounds: list[tuple[float, int]] = []  # (elapsed_s, bytes)
    start = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, queue_depth)) as pool:
            while time.perf_counter() - start < duration_s:
                t0 = time.perf_counter()

                def job(fd: int, jobs: list[Range]) -> int:
                    n = 0
                    for r in jobs:
                        n += len(os.pread(fd, r.length, r.offset))
                    return n

                futures = [pool.submit(job, fd, jobs) for fd, jobs in zip(fds, chunks)]
                total = sum(f.result() for f in futures)
                rounds.append((time.perf_counter() - t0, total))
    finally:
        for fd in fds:
            os.close(fd)
    elapsed = time.perf_counter() - start
    total_bytes = sum(b for _, b in rounds)
    buckets: list[dict] = []
    acc_t, acc_b, bucket_start = 0.0, 0, 0.0
    for dt, nbytes in rounds:
        acc_t += dt
        acc_b += nbytes
        if acc_t - bucket_start >= bucket_s:
            span = acc_t - bucket_start
            buckets.append(
                {
                    "seconds": round(acc_t, 1),
                    "mib_s": round(acc_b / 1024 / 1024 / span, 1),
                }
            )
            bucket_start, acc_b = acc_t, 0
    if acc_b:
        span = acc_t - bucket_start
        buckets.append(
            {
                "seconds": round(acc_t, 1),
                "mib_s": round(acc_b / 1024 / 1024 / span, 1) if span > 0 else 0.0,
            }
        )
    return {
        "queue_depth": queue_depth,
        "duration_s": round(elapsed, 1),
        "rounds": len(rounds),
        "total_bytes": total_bytes,
        "avg_mib_s": round(total_bytes / 1024 / 1024 / elapsed, 1) if elapsed > 0 else 0.0,
        "buckets": buckets,
    }


def describe_case(
    name: str,
    useful_bytes: int,
    requested: list[Range],
    aligned: list[Range],
    stats: dict,
    cache_confidence: str,
) -> dict:
    requested_bytes = sum(r.length for r in requested)
    aligned_bytes = sum(r.length for r in aligned)
    assert useful_bytes <= requested_bytes <= aligned_bytes
    return {
        "case": name,
        "cache_confidence": cache_confidence,
        "useful_bytes": useful_bytes,
        "requested_ranges": len(requested),
        "requested_bytes": requested_bytes,
        "aligned_ranges": len(aligned),
        "aligned_bytes": aligned_bytes,
        "amplification": round(aligned_bytes / useful_bytes, 3) if useful_bytes else 0.0,
        "service": stats,
        "useful_mib_s": round(
            (useful_bytes * stats.get("repeats", 1)) / 1024 / 1024 / stats["elapsed_s"],
            1,
        )
        if stats.get("elapsed_s")
        else 0.0,
    }


def median_of(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0
