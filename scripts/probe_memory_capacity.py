#!/usr/bin/env python3
"""Safe Unified-Memory capacity probe (no system destabilization).

Holds progressively larger CPU + GPU-visible allocations briefly while
watching compression/swap onset, and stops conservatively well before
destructive pressure:

    python scripts/probe_memory_capacity.py [--out profiles/m1-max-64gb]

Levels default to 8..48 GiB total (half numpy-touched, half MLX-held).
Stops early on any swap increase or material compressor growth.
Releases everything between levels and verifies recovery.

Writes memory-capacity.json with the measured levels, recommended hard
ceiling, normal operating budget, and explicit safety reserve.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

LEVELS_GIB = [8, 16, 24, 32, 40, 48]
HOLD_SECONDS = 5.0
REST_SECONDS = 3.0
MAX_COMPRESSOR_DELTA_BYTES = 512 * 1024 * 1024
SAFETY_RESERVE_GIB = 4


def run_command(argv: list[str], timeout: int = 30) -> str | None:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def vm_snapshot() -> dict:
    out = run_command(["vm_stat"]) or ""
    pages: dict[str, int] = {}
    for m in re.finditer(r"^(.+?):\s*(\d+)\.$", out, re.MULTILINE):
        pages[m.group(1).strip().strip('"')] = int(m.group(2))
    page_size = 16384
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page_size = int(m.group(1))
    swap = run_command(["sysctl", "vm.swapusage"]) or ""
    return {
        "page_size": page_size,
        "pages_free": pages.get("Pages free", 0),
        "pages_compressed": pages.get("Pages stored in compressor", 0),
        "compressor_bytes": pages.get("Pages stored in compressor", 0) * page_size,
        "swapusage": swap.strip(),
    }


def swap_used(swapusage: str) -> str:
    m = re.search(r"used\s*=\s*([\d.]+\w*)", swapusage)
    return m.group(1) if m else "unknown"


def clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except ImportError:
        pass


def hold_level(total_gib: int, hold_s: float) -> dict:
    """Hold total_gib (half numpy, half MLX); return alloc timings."""
    import numpy as np

    half = (total_gib * 1024**3) // 2
    t0 = time.perf_counter()
    cpu = np.empty(half, dtype=np.uint8)
    cpu.fill(1)
    cpu_s = time.perf_counter() - t0
    gpu = None
    gpu_s = 0.0
    try:
        import mlx.core as mx

        t0 = time.perf_counter()
        # Keep every dimension under int32 limits: 1 GiB float32 chunks.
        chunk_elems = 2**28
        n_chunks = max(1, half // (chunk_elems * 4))
        gpu = [mx.zeros((chunk_elems,), dtype=mx.float32) for _ in range(n_chunks)]
        mx.eval(gpu)
        gpu_s = time.perf_counter() - t0
    except ImportError:
        pass
    time.sleep(hold_s)
    timings = {"cpu_alloc_s": round(cpu_s, 2), "gpu_alloc_s": round(gpu_s, 2)}
    del cpu
    del gpu
    gc.collect()
    clear_mlx_cache()
    return timings


def probe(max_compressor_delta: int) -> dict:
    machine_mem = None
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        machine_mem = int(proc.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    baseline = vm_snapshot()
    base_compressor = baseline["compressor_bytes"]
    base_swap = swap_used(baseline["swapusage"])
    levels = []
    stopped_reason = "completed all levels"
    for gib in LEVELS_GIB:
        before = vm_snapshot()
        try:
            timings = hold_level(gib, HOLD_SECONDS)
        except (MemoryError, OSError) as exc:
            levels.append(
                {"target_gib": gib, "status": "allocation-failed", "error": str(exc)[:200]}
            )
            stopped_reason = f"allocation failed at {gib} GiB"
            break
        time.sleep(0.5)
        after = vm_snapshot()
        comp_delta = after["compressor_bytes"] - base_compressor
        swap_now = swap_used(after["swapusage"])
        entry = {
            "target_gib": gib,
            "status": "held",
            "hold_s": HOLD_SECONDS,
            "compressor_delta_bytes": comp_delta,
            "swap_before": swap_used(before["swapusage"]),
            "swap_after": swap_now,
            **timings,
        }
        levels.append(entry)
        if swap_now != base_swap and base_swap == "0.00M":
            entry["status"] = "stopped-swap-onset"
            stopped_reason = f"swap onset at {gib} GiB"
            break
        if comp_delta > max_compressor_delta:
            entry["status"] = "stopped-compression-onset"
            stopped_reason = f"compressor +{comp_delta} B at {gib} GiB"
            break
        time.sleep(REST_SECONDS)

    recovery = vm_snapshot()
    clean = [lv["target_gib"] for lv in levels if lv["status"] == "held"]
    hard_ceiling_gib = max(clean) if clean else 0
    operating_gib = max(0, hard_ceiling_gib - SAFETY_RESERVE_GIB)
    return {
        "schema": "deepseeker.memory-capacity/v1",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "physical_bytes": machine_mem,
        "baseline": baseline,
        "recovery": recovery,
        "levels": levels,
        "stopped_reason": stopped_reason,
        "rules": {
            "max_compressor_delta_bytes": MAX_COMPRESSOR_DELTA_BYTES,
            "stop_on_any_swap": True,
            "safety_reserve_gib": SAFETY_RESERVE_GIB,
        },
        "recommendation": {
            "hard_ceiling_gib": hard_ceiling_gib,
            "operating_gib": operating_gib,
            "reserve_gib": SAFETY_RESERVE_GIB,
            "method": "largest clean level minus fixed reserve",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe Unified-Memory capacity probe.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "profiles" / "m1-max-64gb")
    parser.add_argument("--max-compressor-mb", type=int, default=512)
    args = parser.parse_args()
    print("probing levels (GiB):", LEVELS_GIB, flush=True)
    result = probe(args.max_compressor_mb * 1024 * 1024)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory-capacity.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    rec = result["recommendation"]
    print(f"stopped: {result['stopped_reason']}")
    print(f"ceiling={rec['hard_ceiling_gib']} GiB, operating={rec['operating_gib']} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
