"""Issue #41: read-mostly SSD policy + swap guard (cross-cutting).

Write classes: REQUIRED (logs, user outputs, small metadata like the
checkpoint state JSON), OPTIONAL (cache persistence, predictor tables,
temp converted tensors — all opt-in, never automatic), FORBIDDEN
(evicted-weight writeback, Engram writeback, weight repacking per
run, transient checkpoint copies, steady-state swap dependence).

Enforcement: open_backing_file() opens checkpoint payload read-only
and rejects path escapes; a static test forbids write-flag opens in
src (except the allowlisted state-JSON writer). SwapMonitor samples
vm_stat/swapusage around a run and reports onset per thresholds.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SCHEMA = "deepseeker.ssd-policy/v1"

WRITE_CLASSES = {
    "required": (
        "logs and small telemetry",
        "explicit user-requested outputs",
        "small sanitized benchmark/profile metadata",
        "checkpoint state JSON (resume cursor, bytes not weights)",
    ),
    "optional": (
        "cache persistence",
        "prefetch state persistence",
        "predictor tables",
        "temporary converted tensors",
    ),
    "forbidden": (
        "evicted expert weight writeback",
        "Engram row/page writeback",
        "per-run immutable weight repacking",
        "transient checkpoint-shard copies",
        "steady-state swap dependence",
    ),
}

# src files allowed to open files for writing (state/metadata only).
WRITE_ALLOWLIST = ("checkpoint.py",)


def open_backing_file(model_root: Path | str, shard: str) -> int:
    """Open a checkpoint payload strictly read-only.

    Rejects absolute shard names and `..` escapes so a bad manifest or
    caller can never redirect a write (or a read) outside the model root.
    Callers must os.close() (or use the staging fd cache).
    """
    if os.path.isabs(shard) or ".." in Path(shard).parts:
        raise ValueError(f"refusing backing path outside model root: {shard!r}")
    path = Path(model_root) / shard
    if not path.is_file():
        raise FileNotFoundError(f"backing file missing: {path}")
    return os.open(path, os.O_RDONLY)


def parse_vm_stat(text: str) -> dict:
    """Parse `vm_stat` output into {key: int} (pages, faults as given)."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"\s*(.+?):\s+([\d.]+)\.?", line)
        if match:
            name, value = match.groups()
            try:
                out[name.strip()] = int(float(value))
            except ValueError:
                continue
    return out


def parse_swapusage(text: str) -> dict:
    """Parse `sysctl vm.swapusage` (values in MB)."""
    out = {"total_mb": 0.0, "used_mb": 0.0, "free_mb": 0.0}
    for key in ("total", "used", "free"):
        match = re.search(rf"{key}\s*=\s*([\d.]+)M", text)
        if match:
            out[f"{key}_mb"] = float(match.group(1))
    return out


class SwapMonitor:
    """Before/after sampler for swap + compression pressure."""

    def __init__(self, warn_swap_mb: float = 1.0, stop_swap_mb: float = 2048.0) -> None:
        self._warn = warn_swap_mb
        self._stop = stop_swap_mb
        self.before: dict | None = None
        self.after: dict | None = None

    @staticmethod
    def _sample_once() -> dict:
        vm = parse_vm_stat(
            subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=30,
                           check=False).stdout
        )
        swap = parse_swapusage(
            subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True,
                           timeout=30, check=False).stdout
        )
        page = vm.get("page size of 16384 bytes", 16384)
        return {
            "swap_used_mb": swap["used_mb"],
            "compressor_pages": vm.get("Pages occupied by compressor", 0),
            "page_bytes": page,
        }

    def begin(self) -> dict:
        self.before = self._sample_once()
        return self.before

    def end(self) -> dict:
        self.after = self._sample_once()
        if self.before is None:
            raise RuntimeError("begin() was not called")
        delta_swap = self.after["swap_used_mb"] - self.before["swap_used_mb"]
        delta_comp = self.after["compressor_pages"] - self.before["compressor_pages"]
        if self.after["swap_used_mb"] >= self._stop or delta_swap >= self._stop:
            verdict = "STOP"
        elif self.after["swap_used_mb"] >= self._warn or delta_swap > 0 or delta_comp > 0:
            verdict = "WARN"
        else:
            verdict = "OK"
        return {
            "schema": SCHEMA,
            "before": self.before,
            "after": self.after,
            "delta_swap_mb": round(delta_swap, 2),
            "delta_compressor_pages": delta_comp,
            "verdict": verdict,
            "advice": {
                "OK": "no pressure observed",
                "WARN": "reduce discretionary resident/prefetch pressure",
                "STOP": "do not report benchmark numbers; relieve memory first",
            }[verdict],
        }
