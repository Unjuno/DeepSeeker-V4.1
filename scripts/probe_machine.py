#!/usr/bin/env python3
"""DeepSeeker target-machine probe (privacy-safe, one command).

Collects performance-relevant, privacy-safe machine/software state and
safe baseline microbenchmarks, and writes structured JSON output:

    python scripts/probe_machine.py [--out-dir profiles/m1-max-64gb]

Privacy model: system commands are allowlisted and their output is parsed
with narrow regular expressions. Only the extracted numeric/version fields
are stored. Raw command output is never written to disk or committed.

Blocked/uncertain measurements (full Xcode, purge-controlled cold SSD,
model weights) are reported as structured ``unavailable``/``blocked``
entries instead of crashing or silently disappearing.
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import json
import os
import re
import socket
import subprocess
import tempfile
from pathlib import Path

PROBE_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "profiles" / "m1-max-64gb"

# DeepSeek-relevant expert dimensions from the pinned upstream config.
HIDDEN_SIZE = 5120
MOE_INTERMEDIATE_SIZE = 2304
BENCH_BATCHES = [1, 2, 4, 8, 16]

CPU_WARMUP = 3
CPU_ITERS = 10
GPU_WARMUP = 5
GPU_ITERS = 20
STORAGE_FILE_BYTES = 256 * 1024 * 1024
STORAGE_BLOCK_BYTES = 4 * 1024 * 1024


# --------------------------------------------------------------------------
# Command execution (never raises; returns stdout or None)


def run_command(argv: list[str], timeout: int = 60) -> str | None:
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
    if proc.returncode != 0:
        return None
    return proc.stdout


def sysctl_value(key: str) -> str | None:
    out = run_command(["sysctl", "-n", key])
    return out.strip() if out else None


def sysctl_int(key: str) -> int | None:
    val = sysctl_value(key)
    try:
        return int(val) if val is not None else None
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Narrow parsers (each extracts only the needed fields)


def parse_sw_vers(text: str) -> dict:
    result: dict = {}
    m = re.search(r"^ProductVersion:\s*(.+)$", text, re.MULTILINE)
    if m:
        result["macos_version"] = m.group(1).strip()
    m = re.search(r"^BuildVersion:\s*(.+)$", text, re.MULTILINE)
    if m:
        result["macos_build"] = m.group(1).strip()
    return result


def parse_displays(text: str) -> dict:
    """Extract only chipset model, GPU core count, Metal support."""
    result: dict = {}
    m = re.search(r"Chipset Model:\s*(.+)", text)
    if m:
        result["gpu_chipset"] = m.group(1).strip()
    m = re.search(r"Total Number of Cores:\s*(\d+)", text)
    if m:
        result["gpu_cores"] = int(m.group(1))
    m = re.search(r"Metal Support:\s*(.+)", text)
    if m:
        result["metal_support"] = m.group(1).strip()
    return result


def parse_nvme(text: str) -> dict:
    """Extract only the SSD model class and capacity (no serials)."""
    result: dict = {}
    m = re.search(r"^\s*Model:\s*(.+)$", text, re.MULTILINE)
    if m:
        result["storage_model"] = m.group(1).strip()
    m = re.search(r"^\s*Capacity:\s*(.+)$", text, re.MULTILINE)
    if m:
        result["storage_capacity"] = m.group(1).strip()
    return result


def parse_df_root(text: str) -> dict:
    result: dict = {}
    for line in text.splitlines():
        parts = line.split()
        # Expect: device 1K-blocks Used Available Capacity iused ... Mounted
        if len(parts) >= 9 and parts[-1] == "/":
            try:
                result["root_capacity_kb"] = int(parts[1])
                result["root_available_kb"] = int(parts[3])
            except ValueError:
                pass
    return result


def parse_vm_stat(text: str) -> dict:
    pages: dict = {}
    for m in re.finditer(r"^(.+?):\s*(\d+)\.$", text, re.MULTILINE):
        name = m.group(1).strip().strip('"')
        pages[name] = int(m.group(2))
    result: dict = {}
    m = re.search(r"page size of (\d+) bytes", text)
    if m:
        result["page_size_bytes"] = int(m.group(1))
    keep = {
        "Pages free": "pages_free",
        "Pages active": "pages_active",
        "Pages inactive": "pages_inactive",
        "Pages speculative": "pages_speculative",
        "Pages wired down": "pages_wired",
        "Pages purgeable": "pages_purgeable",
        "Pages stored in compressor": "pages_compressed",
        "Pages occupied by compressor": "pages_compressor_used",
        "Pageins": "pageins",
    }
    for raw, clean in keep.items():
        if raw in pages:
            result[clean] = pages[raw]
    return result


def parse_swapusage(text: str) -> dict:
    result: dict = {}
    m = re.search(
        r"total\s*=\s*([\d.]+\w*)\s*used\s*=\s*([\d.]+\w*)\s*"
        r"free\s*=\s*([\d.]+\w*)",
        text,
    )
    if m:
        result["swap_total"] = m.group(1)
        result["swap_used"] = m.group(2)
        result["swap_free"] = m.group(3)
    return result


def parse_memory_pressure(text: str) -> dict:
    result: dict = {}
    m = re.search(r"(\d+)%", text)
    if m:
        result["memory_free_percent"] = int(m.group(1))
    return result


# --------------------------------------------------------------------------
# Collectors


def collect_machine() -> dict:
    machine: dict = {
        "soc": sysctl_value("machdep.cpu.brand_string"),
        "cpu_total": sysctl_int("hw.ncpu"),
        "cpu_performance_cores": sysctl_int("hw.perflevel0.physicalcpu"),
        "cpu_efficiency_cores": sysctl_int("hw.perflevel1.physicalcpu"),
        "cpu_p_l2_bytes": sysctl_int("hw.perflevel0.l2cachesize"),
        "cpu_e_l2_bytes": sysctl_int("hw.perflevel1.l2cachesize"),
        "unified_memory_bytes": sysctl_int("hw.memsize"),
    }
    displays = run_command(["system_profiler", "SPDisplaysDataType"], timeout=120)
    if displays is not None:
        machine.update(parse_displays(displays))
    nvme = run_command(["system_profiler", "SPNVMeDataType"], timeout=120)
    if nvme is not None:
        machine.update(parse_nvme(nvme))
    df = run_command(["df", "-k", "/"])
    if df is not None:
        machine.update(parse_df_root(df))
    power = run_command(["pmset", "-g"])
    if power is not None:
        m = re.search(r"^\s*powermode\s+(\d+)", power, re.MULTILINE)
        if m:
            machine["powermode"] = int(m.group(1))
    machine["schema"] = "deepseeker.machine/v1"
    return machine


def collect_software() -> dict:
    sw_vers = run_command(["sw_vers"]) or ""
    software: dict = {
        **parse_sw_vers(sw_vers),
        "kernel_release": sysctl_value("kern.osrelease"),
        "kernel_revision": sysctl_value("kern.osrevision"),
        "arch": sysctl_value("hw.machine"),
        "python_version": None,
        "clang_version": None,
    }
    import sys as _sys

    software["python_version"] = _sys.version.split()[0]

    clang = run_command(["clang", "--version"])
    if clang:
        software["clang_version"] = clang.splitlines()[0].strip()

    xcode_select = run_command(["xcode-select", "-p"])
    software["developer_dir"] = xcode_select.strip() if xcode_select else None
    software["xcode_app_installed"] = Path("/Applications/Xcode.app").exists()
    metal = run_command(["xcrun", "--find", "metal"])
    software["metal_compiler_available"] = metal is not None

    try:
        import mlx.core as mx  # type: ignore

        software["mlx"] = {
            "available": True,
            "device": str(mx.default_device()),
        }
    except ImportError:
        software["mlx"] = {
            "available": False,
            "reason": "not-installed",
        }

    commit = run_command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    software["deepseeker_commit"] = commit.strip() if commit else None

    manifest = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            software["upstream"] = {
                "available": True,
                "repo_id": data.get("repo_id"),
                "requested_revision": data.get("requested_revision"),
                "resolved_revision": data.get("resolved_revision"),
                "weights_downloaded": data.get("weights_downloaded"),
            }
        except (OSError, ValueError):
            software["upstream"] = {
                "available": False,
                "reason": "manifest-unreadable",
            }
    else:
        software["upstream"] = {
            "available": False,
            "reason": "manifest-missing",
            "hint": "run python scripts/fetch_upstream.py",
        }
    software["schema"] = "deepseeker.software/v1"
    return software


def collect_memory() -> dict:
    memory: dict = {
        "physical_bytes": sysctl_int("hw.memsize"),
    }
    vm = run_command(["vm_stat"])
    if vm is not None:
        memory.update(parse_vm_stat(vm))
    swap = run_command(["sysctl", "vm.swapusage"])
    if swap is not None:
        memory.update(parse_swapusage(swap))
    pressure = run_command(["memory_pressure"], timeout=30)
    if pressure is not None:
        memory.update(parse_memory_pressure(pressure))
    memory["schema"] = "deepseeker.memory/v1"
    return memory


# --------------------------------------------------------------------------
# Microbenchmarks


def _stats_ms(samples_s: list[float]) -> dict:
    ms = sorted(s * 1000.0 for s in samples_s)
    mid = len(ms) // 2
    return {
        "median_ms": ms[mid],
        "min_ms": ms[0],
        "max_ms": ms[-1],
    }


def bench_cpu_numpy() -> dict:
    try:
        import numpy as np
    except ImportError:
        return {"available": False, "reason": "numpy-not-installed"}
    results = []
    for size_mb in (64, 256):
        n = size_mb * 1024 * 1024 // 8
        a = np.random.rand(n)
        b = np.empty_like(a)
        for _ in range(CPU_WARMUP):
            np.copyto(b, a)
        import time as _time

        samples = []
        for _ in range(CPU_ITERS):
            t0 = _time.perf_counter()
            np.copyto(b, a)
            samples.append(_time.perf_counter() - t0)
        stats = _stats_ms(samples)
        gib = (2 * n * 8) / (stats["median_ms"] / 1000.0) / 1024**3
        results.append(
            {
                "op": "streaming-copy",
                "backend": "numpy",
                "device": "cpu",
                "dtype": "float64",
                "buffer_bytes": n * 8,
                "warmup_iters": CPU_WARMUP,
                "measured_iters": CPU_ITERS,
                **stats,
                "throughput_gib_s": round(gib, 1),
                "throughput_basis": "read+write",
            }
        )
    return {"available": True, "numpy_version": str(np.__version__), "results": results}


def bench_mlx_gpu() -> dict:
    try:
        import time as _time

        import mlx.core as mx
    except ImportError:
        return {"available": False, "reason": "mlx-not-installed"}

    def timed(fn, warmup: int, iters: int) -> dict:
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(iters):
            t0 = _time.perf_counter()
            fn()
            samples.append(_time.perf_counter() - t0)
        return _stats_ms(samples)

    results = []
    n = 256 * 1024 * 1024 // 2
    a = mx.random.uniform(shape=(n,)).astype(mx.float16)
    b = mx.random.uniform(shape=(n,)).astype(mx.float16)
    mx.eval(a, b)

    def triad() -> None:
        c = a + b * 2.0
        mx.eval(c)

    stats = timed(triad, GPU_WARMUP, GPU_ITERS)
    gib = (3 * n * 2) / (stats["median_ms"] / 1000.0) / 1024**3
    results.append(
        {
            "op": "stream-triad",
            "backend": "mlx",
            "device": str(mx.default_device()),
            "dtype": "float16",
            "buffer_bytes": n * 2,
            "warmup_iters": GPU_WARMUP,
            "measured_iters": GPU_ITERS,
            **stats,
            "throughput_gib_s": round(gib, 1),
            "throughput_basis": "2 reads + 1 write",
        }
    )

    h, inter = HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE
    w_gate = mx.random.normal(shape=(h, inter)).astype(mx.float16)
    mx.eval(w_gate)
    for batch in BENCH_BATCHES:
        x = mx.random.normal(shape=(batch, h)).astype(mx.float16)
        mx.eval(x)

        def gemm(x=x) -> None:
            y = x @ w_gate
            mx.eval(y)

        stats = timed(gemm, GPU_WARMUP, GPU_ITERS)
        flops = 2 * batch * h * inter
        results.append(
            {
                "op": "moe-gate-gemm",
                "backend": "mlx",
                "device": str(mx.default_device()),
                "dtype": "float16",
                "shape": {"batch": batch, "hidden": h, "intermediate": inter},
                "warmup_iters": GPU_WARMUP,
                "measured_iters": GPU_ITERS,
                **stats,
                "gflops": round(flops / stats["median_ms"] / 1e6, 1),
            }
        )

    g = mx.random.normal(shape=(6, inter)).astype(mx.float16)
    u = mx.random.normal(shape=(6, inter)).astype(mx.float16)
    mx.eval(g, u)

    def swiglu() -> None:
        y = g * mx.sigmoid(u) * u
        mx.eval(y)

    stats = timed(swiglu, GPU_WARMUP, GPU_ITERS)
    results.append(
        {
            "op": "swiglu-elementwise-unfused",
            "backend": "mlx",
            "device": str(mx.default_device()),
            "dtype": "float16",
            "shape": {"batch": 6, "intermediate": inter},
            "warmup_iters": GPU_WARMUP,
            "measured_iters": GPU_ITERS,
            **stats,
        }
    )
    return {"available": True, "results": results}


def bench_storage() -> dict:
    """Disposable-file baseline. Honestly labeled: advisory-uncached only.

    Uses fcntl F_NOCACHE (advisory on macOS). The file fits in the page
    cache, so a cache-resident component cannot be excluded without
    ``sudo purge``. The methodology label says exactly that.
    """
    import time as _time

    try:
        import fcntl
    except ImportError:
        return {"available": False, "reason": "fcntl-unavailable"}
    method = (
        "disposable-file with fcntl F_NOCACHE (advisory only); "
        "NOT purge-controlled; cache-resident component possible"
    )
    fd_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="deepseeker-probe-", delete=False) as tmp:
            fd_path = tmp.name
        with open(fd_path, "wb", buffering=0) as f:
            try:
                fcntl.fcntl(f.fileno(), 48, 1)
            except OSError:
                pass
            chunk = os.urandom(STORAGE_BLOCK_BYTES)
            t0 = _time.perf_counter()
            for _ in range(STORAGE_FILE_BYTES // STORAGE_BLOCK_BYTES):
                os.write(f.fileno(), chunk)
            os.fsync(f.fileno())
        write_s = _time.perf_counter() - t0

        with open(fd_path, "rb", buffering=0) as f:
            try:
                fcntl.fcntl(f.fileno(), 48, 1)
            except OSError:
                pass
            t0 = _time.perf_counter()
            total = 0
            while True:
                data = os.read(f.fileno(), STORAGE_BLOCK_BYTES)
                if not data:
                    break
                total += len(data)
        read_s = _time.perf_counter() - t0
    finally:
        if fd_path and os.path.exists(fd_path):
            os.remove(fd_path)
    mib = STORAGE_FILE_BYTES / 1024 / 1024
    return {
        "available": True,
        "method": method,
        "cache_control": "advisory-uncached",
        "cold_storage_proven": False,
        "file_bytes": STORAGE_FILE_BYTES,
        "block_bytes": STORAGE_BLOCK_BYTES,
        "sequential_write_mib_s": round(mib / write_s, 1),
        "sequential_read_mib_s": round(mib / read_s, 1),
    }


# --------------------------------------------------------------------------
# Privacy validation


def runtime_forbidden_patterns() -> list[re.Pattern]:
    patterns = [
        r"/Users/[A-Za-z0-9_.\-]+",
        (
            r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
            r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
        ),
        r"[Ss]erial [Nn]umber",
        r"Hardware UUID",
        r"Provisioning UDID",
        r"Apple ID",
    ]
    try:
        host = socket.gethostname()
        if host and host not in ("localhost",):
            # Match the hostname and its short (pre-domain) form.
            patterns.append(re.escape(host))
            patterns.append(re.escape(host.split(".")[0]))
    except OSError:
        pass
    try:
        user = getpass.getuser()
        if user and user not in ("root",):
            patterns.append(r"\b" + re.escape(user) + r"\b")
    except OSError:
        pass
    home = os.path.expanduser("~")
    if home and home != "~" and home != "/":
        patterns.append(re.escape(home))
    return [re.compile(p) for p in patterns]


def privacy_scan(payload: object) -> list[str]:
    text = json.dumps(payload, indent=2, default=str)
    hits = []
    for pat in runtime_forbidden_patterns():
        m = pat.search(text)
        if m:
            context = text[max(0, m.start() - 40) : m.end() + 40]
            hits.append(f"{pat.pattern!r} near ...{context}...")
    return hits


# --------------------------------------------------------------------------
# Output


def build_readme(machine: dict, software: dict, benchmark: dict) -> str:
    lines = [
        "# Probe baseline — m1-max-64gb (generated, sanitized)",
        "",
        "Generated by `python scripts/probe_machine.py`.",
        "All fields are parsed narrowly from allowlisted commands;",
        "raw command output is never stored.",
        "",
        "## Machine",
        "",
        f"- SoC: {machine.get('soc')}",
        (
            f"- CPU: {machine.get('cpu_total')} total "
            f"({machine.get('cpu_performance_cores')}P + "
            f"{machine.get('cpu_efficiency_cores')}E)"
        ),
        (
            f"- GPU: {machine.get('gpu_chipset')} "
            f"({machine.get('gpu_cores')} cores, "
            f"{machine.get('metal_support')})"
        ),
        f"- Unified memory: {machine.get('unified_memory_bytes')} bytes",
        (f"- Storage: {machine.get('storage_model')} ({machine.get('storage_capacity')})"),
        "",
        "## Software",
        "",
        (f"- macOS: {software.get('macos_version')} ({software.get('macos_build')})"),
        f"- Kernel: {software.get('kernel_release')}",
        f"- Python: {software.get('python_version')}",
        f"- clang: {software.get('clang_version')}",
        (
            f"- Xcode.app: {software.get('xcode_app_installed')}, "
            f"metal compiler: {software.get('metal_compiler_available')}"
        ),
        f"- DeepSeeker: {software.get('deepseeker_commit')}",
        "",
        "## Benchmarks",
        "",
        ("See `benchmark.json` for full records (warmup/iters/median/min/max/conditions)."),
        "",
        (
            "Storage results are advisory-uncached unless proven otherwise; "
            "see the `method` and `cold_storage_proven` fields."
        ),
        "",
        "## Rerun",
        "",
        "```bash",
        "python scripts/probe_machine.py",
        "```",
        "",
    ]
    _ = benchmark
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Privacy-safe DeepSeeker target-machine probe.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: profiles/m1-max-64gb).",
    )
    args = parser.parse_args()

    machine = collect_machine()
    software = collect_software()
    memory = collect_memory()
    benchmark = {
        "schema": "deepseeker.benchmark/v1",
        "probe_version": PROBE_VERSION,
        "collected_at_utc": (datetime.datetime.now(datetime.UTC).isoformat()),
        "deepseeker_commit": software.get("deepseeker_commit"),
        "upstream_resolved_revision": ((software.get("upstream") or {}).get("resolved_revision")),
        "cpu_streaming": bench_cpu_numpy(),
        "mlx_gpu": bench_mlx_gpu(),
        "storage": bench_storage(),
    }
    machine["probe_version"] = PROBE_VERSION
    software["probe_version"] = PROBE_VERSION
    memory["probe_version"] = PROBE_VERSION

    payload = {"machine": machine, "software": software, "memory": memory}
    hits = privacy_scan({**payload, "benchmark": benchmark})
    if hits:
        print("PRIVACY SCAN FAILED — refusing to write output:")
        for hit in hits:
            print(f"  - {hit}")
        return 2

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "machine.json").write_text(
        json.dumps({**machine, "software": software, "memory": memory}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(build_readme(machine, software, benchmark), encoding="utf-8")
    print(f"probe v{PROBE_VERSION} complete -> {out}")
    print("privacy scan: clean")
    mlx_status = benchmark["mlx_gpu"].get("available")
    print(f"mlx gpu baseline: {'ok' if mlx_status else 'unavailable'}")
    print(f"metal compiler: {software.get('metal_compiler_available')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
