"""Issue #27: KV/sparse-index runtime memory manager (roadmap P5).

Allocates the Issue #15 cache geometry as real buffers (numpy uint8 =
raw fp8/fp4 bytes; values are opaque until P7 kernels interpret them),
with alloc/free/grow/reset lifecycle, hard byte budgets, and telemetry.
Model-vs-actual reconciliation: allocated bytes must equal the
kvmodel.cache_bytes formula exactly (zero allocator overhead by
construction — NumPy has no per-buffer padding here; any drift FAILs).

No attention semantics change: this manages memory, never math.
"""

from __future__ import annotations

import numpy as np

from deepseeker.kvmodel import cache_bytes, cache_geometry

SCHEMA = "deepseeker.kvstore/v1"

# Element sizes mirror kvmodel: window fp8, compress/index fp4 (bytes).
WINDOW_ELEMS = ("window",)
ELEM_BYTES = {"window": 1.0, "compress": 0.5, "index": 0.5}


class BudgetExceeded(RuntimeError):
    """Allocation would break the hard byte ceiling: fail clean, no partial."""


class KVStore:
    """Per-request KV/index store with explicit byte accounting."""

    def __init__(self, config: dict, budget_bytes: int) -> None:
        if budget_bytes < 1:
            raise ValueError("budget_bytes must be >= 1")
        self._geometry = cache_geometry(config)
        self._budget = budget_bytes
        self._requests: dict[str, dict] = {}
        self.allocated_bytes = 0
        self.high_water_bytes = 0
        self.allocations = 0
        self.frees = 0

    @property
    def geometry(self) -> dict:
        return self._geometry

    def _request_bytes(self, batch: int, seq_len: int) -> int:
        return cache_bytes(self._geometry, batch, seq_len)["total_bytes"]

    def _buffers(self, batch: int, seq_len: int) -> dict:
        # fp4 classes pack 2 elements per byte: halved last dim keeps
        # allocated nbytes exactly equal to the formula (zero overhead).
        geometry = self._geometry
        head = geometry["head_dim"]
        win = geometry["window_size"]
        buffers: dict[str, np.ndarray] = {
            "window": np.zeros(
                (geometry["n_layers"], batch, win, head), dtype=np.uint8
            )
        }
        for layer in geometry["kv_source_layers"]:
            ratio = geometry["compress_ratios"][layer]
            n = seq_len // ratio
            buffers[f"compress:{layer}"] = np.zeros((batch, n, head // 2), dtype=np.uint8)
            buffers[f"index:{layer}"] = np.zeros(
                (batch, n, geometry["index_head_dim"] // 2), dtype=np.uint8
            )
        return buffers

    def _check(self, additional: int) -> None:
        if self.allocated_bytes + additional > self._budget:
            raise BudgetExceeded(
                f"{self.allocated_bytes}B + {additional}B > budget {self._budget}B"
            )

    def alloc(self, request_id: str, batch: int, seq_len: int) -> dict:
        """Allocate a request's full cache set; replaces any prior state."""
        if request_id in self._requests:
            self.free(request_id)
        want = self._request_bytes(batch, seq_len)
        self._check(want)
        entry = {"batch": batch, "seq_len": seq_len, "buffers": self._buffers(batch, seq_len),
                 "bytes": want}
        self._requests[request_id] = entry
        self.allocated_bytes += want
        self.high_water_bytes = max(self.high_water_bytes, self.allocated_bytes)
        self.allocations += 1
        return {"request_id": request_id, "bytes": want, "seq_len": seq_len}

    def grow(self, request_id: str, seq_len: int) -> dict:
        """Extend a request to a longer context (reallocates that request)."""
        entry = self._requests.get(request_id)
        if entry is None:
            raise KeyError(f"unknown request {request_id!r}")
        if seq_len < entry["seq_len"]:
            raise ValueError("grow-only; use reset to shrink")
        batch = entry["batch"]
        self.free(request_id)
        return self.alloc(request_id, batch, seq_len)

    def free(self, request_id: str) -> int:
        """Release a request; returns freed bytes."""
        entry = self._requests.pop(request_id, None)
        if entry is None:
            return 0
        self.allocated_bytes -= entry["bytes"]
        self.frees += 1
        return entry["bytes"]

    def reset(self) -> None:
        """Release everything (between benchmark cases)."""
        self._requests.clear()
        self.allocated_bytes = 0

    def write_pattern(self, request_id: str, key: str, value: int) -> None:
        """Write a fill byte into a buffer (roundtrip/alias check)."""
        entry = self._requests[request_id]
        entry["buffers"][key].fill(value)

    def read_sum(self, request_id: str, key: str) -> int:
        entry = self._requests[request_id]
        return int(entry["buffers"][key].sum())

    def reconcile(self, request_id: str) -> dict:
        """Actual vs formula bytes for one live request."""
        entry = self._requests[request_id]
        actual = sum(buf.nbytes for buf in entry["buffers"].values())
        expected = self._request_bytes(entry["batch"], entry["seq_len"])
        return {
            "request_id": request_id,
            "actual_bytes": actual,
            "formula_bytes": expected,
            "match": actual == expected,
        }

    def telemetry(self) -> dict:
        return {
            "schema": SCHEMA,
            "budget_bytes": self._budget,
            "allocated_bytes": self.allocated_bytes,
            "high_water_bytes": self.high_water_bytes,
            "live_requests": len(self._requests),
            "allocations": self.allocations,
            "frees": self.frees,
        }
