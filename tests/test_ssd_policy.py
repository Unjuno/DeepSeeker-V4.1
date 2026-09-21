"""Tests for Issue #41 SSD policy (static audit + monitor + helper)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.ssd_policy import (
    WRITE_ALLOWLIST,
    WRITE_CLASSES,
    SwapMonitor,
    open_backing_file,
    parse_swapusage,
    parse_vm_stat,
)


def test_write_classes_complete():
    assert set(WRITE_CLASSES) == {"required", "optional", "forbidden"}
    assert any("writeback" in item for item in WRITE_CLASSES["forbidden"])


def test_no_write_opens_outside_allowlist():
    """Static audit: src may only open files read-only, except allowlist."""
    offenders = []
    for path in (REPO_ROOT / "src" / "deepseeker").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name in WRITE_ALLOWLIST:
            continue
        for marker in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "open(",):
            if marker == "open(":
                continue  # open_backing_file/os.open(O_RDONLY) handled below
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert offenders == []
    # os.open calls that remain must all pass O_RDONLY explicitly
    for path in (REPO_ROOT / "src" / "deepseeker").glob("*.py"):
        if path.name in WRITE_ALLOWLIST:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "os.open(" in line:
                assert "O_RDONLY" in line, f"{path.name}:{i}: {line.strip()}"


def test_open_backing_file_guards(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    (root / "a.safetensors").write_bytes(b"0123456789")
    fd = open_backing_file(root, "a.safetensors")
    assert os.pread(fd, 4, 1) == b"1234"
    os.close(fd)
    with pytest.raises(ValueError):
        open_backing_file(root, "../escape.safetensors")
    with pytest.raises(ValueError):
        open_backing_file(root, "/abs/path.safetensors")
    with pytest.raises(FileNotFoundError):
        open_backing_file(root, "missing.safetensors")


def test_parsers():
    vm = parse_vm_stat("Pages free: 123.\nPages occupied by compressor: 456.\nTranslation faults: 789.")
    assert vm["Pages free"] == 123
    assert vm["Pages occupied by compressor"] == 456
    swap = parse_swapusage("vm.swapusage: total = 2048.00M  used = 10.00M  free = 2038.00M  (encrypted)")
    assert swap == {"total_mb": 2048.0, "used_mb": 10.0, "free_mb": 2038.0}


def test_monitor_roundtrip_live():
    monitor = SwapMonitor()
    before = monitor.begin()
    assert "swap_used_mb" in before
    report = monitor.end()
    assert report["verdict"] in ("OK", "WARN", "STOP")
    assert set(report) >= {"before", "after", "delta_swap_mb", "verdict", "advice"}
