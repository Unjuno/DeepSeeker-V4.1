"""Tests for the privacy-safe target-machine probe.

Fast and hermetic: parser unit tests use fixture strings, privacy tests
use synthetic payloads, and deliverable tests validate the committed
canonical JSON files (no benchmark rerun needed).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "scripts" / "probe_machine.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("probe_machine", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_probe()

# --------------------------------------------------------------------------
# Parser fixtures


SW_VERS = """ProductName:\tmacOS
ProductVersion:\t26.6.2
BuildVersion:\t25G83
"""

DISPLAYS = """
Graphics/Displays:

    Apple M1 Max:

      Chipset Model: Apple M1 Max
      Type: GPU
      Bus: Built-In
      Total Number of Cores: 32
      Vendor: Apple (0x106b)
      Metal Support: Metal 4
"""

NVME = """
NVMExpress:

    Apple SSD Controller:

        APPLE SSD AP4096R:

          Capacity: 4 TB (4,002,222,325,760 bytes)
          TRIM Support: Yes
          Model: APPLE SSD AP4096R
          Revision: 561.100.
          Serial Number: SHOULD-NOT-BE-PARSED
"""

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               58754.
Pages active:                           1938596.
Pages speculative:                       376427.
"Translation faults":                 264266628.
Pages stored in compressor:                   0.
Pageins:                                6068186.
"""

DF_ROOT = """Filesystem   1024-blocks      Used Available Capacity iused ifree %iused  Mounted on
/dev/disk3s1s1  3902665360 1345966320 2557005708     1%    459k   4.3G    0%   /
"""

SWAP = "vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)\n"


def test_parse_sw_vers_extracts_only_version_fields():
    parsed = probe.parse_sw_vers(SW_VERS)
    assert parsed == {"macos_version": "26.6.2", "macos_build": "25G83"}


def test_parse_displays_extracts_only_public_gpu_fields():
    parsed = probe.parse_displays(DISPLAYS)
    assert parsed == {
        "gpu_chipset": "Apple M1 Max",
        "gpu_cores": 32,
        "metal_support": "Metal 4",
    }


def test_parse_nvme_skips_serial_number():
    parsed = probe.parse_nvme(NVME)
    assert parsed["storage_model"] == "APPLE SSD AP4096R"
    assert "4 TB" in parsed["storage_capacity"]
    assert "SHOULD-NOT-BE-PARSED" not in json.dumps(parsed)


def test_parse_vm_stat_selects_curated_fields():
    parsed = probe.parse_vm_stat(VM_STAT)
    assert parsed["page_size_bytes"] == 16384
    assert parsed["pages_free"] == 58754
    assert parsed["pages_compressed"] == 0
    assert parsed["pageins"] == 6068186
    assert "Translation faults" not in json.dumps(parsed)


def test_parse_df_root_reads_capacity_only():
    parsed = probe.parse_df_root(DF_ROOT)
    assert parsed == {
        "root_capacity_kb": 3902665360,
        "root_available_kb": 2557005708,
    }


def test_parse_swapusage():
    parsed = probe.parse_swapusage(SWAP)
    assert parsed == {
        "swap_total": "0.00M",
        "swap_used": "0.00M",
        "swap_free": "0.00M",
    }


# --------------------------------------------------------------------------
# Privacy scanner


def test_privacy_scan_clean_payload():
    payload = {
        "soc": "Apple M1 Max",
        "gpu_cores": 32,
        "macos_version": "26.6.2",
        "deepseeker_commit": "a07fb03ac4427113707582922028b02f29272d58",
    }
    assert probe.privacy_scan(payload) == []


def test_privacy_scan_catches_home_path_username_uuid_and_serial():
    import getpass
    import os

    payload = {
        "note": f"stored under {os.path.expanduser('~')}/data",
        "owner": getpass.getuser(),
        "device": "7D7D2573-CAE5-5528-9D10-85183B945A77",
        "label": "Serial Number: C7Y92J96K1",
    }
    hits = probe.privacy_scan(payload)
    assert len(hits) >= 3, hits


# --------------------------------------------------------------------------
# Committed deliverables (canonical profile, if present)


def _canonical_dir() -> Path:
    return REPO_ROOT / "profiles" / "m1-max-64gb"


def test_canonical_machine_json_schema_and_privacy():
    machine_path = _canonical_dir() / "machine.json"
    if not machine_path.exists():
        probe_pytest_skip("canonical profile not generated yet")
    data = json.loads(machine_path.read_text(encoding="utf-8"))
    for key in ("soc", "cpu_total", "unified_memory_bytes", "software", "memory"):
        assert key in data, f"missing machine key: {key}"
    sw = data["software"]
    for key in (
        "macos_version",
        "kernel_release",
        "python_version",
        "deepseeker_commit",
        "upstream",
    ):
        assert key in sw, f"missing software key: {key}"
    assert probe.privacy_scan(data) == []


def test_canonical_benchmark_json_schema_and_conditions():
    bench_path = _canonical_dir() / "benchmark.json"
    if not bench_path.exists():
        probe_pytest_skip("canonical profile not generated yet")
    data = json.loads(bench_path.read_text(encoding="utf-8"))
    assert data["upstream_resolved_revision"]
    assert data["deepseeker_commit"]
    for section in ("cpu_streaming", "mlx_gpu", "storage"):
        assert section in data, f"missing benchmark section: {section}"
    for entry in data["mlx_gpu"].get("results", []):
        for key in (
            "op",
            "backend",
            "device",
            "dtype",
            "warmup_iters",
            "measured_iters",
            "median_ms",
            "min_ms",
            "max_ms",
        ):
            assert key in entry, f"missing condition {key} in {entry}"
    batches = [
        r["shape"]["batch"]
        for r in data["mlx_gpu"].get("results", [])
        if r["op"] == "moe-gate-gemm"
    ]
    assert batches == [1, 2, 4, 8, 16], batches
    assert data["storage"]["cold_storage_proven"] is False
    assert probe.privacy_scan(data) == []


def probe_pytest_skip(reason: str):
    try:
        import pytest

        pytest.skip(reason)
    except ImportError:
        raise AssertionError(reason)
