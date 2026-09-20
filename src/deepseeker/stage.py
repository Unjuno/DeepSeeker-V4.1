"""Issue #24: asynchronous six-range expert staging engine (roadmap P4).

Fetches one routed expert as its real six tensor ranges (w1/w2/w3 +
scales, never span-coalesced across gaps), with QD control, dedup,
cancellation, bounded staging buffers, and manifest validation. Hands
verified bytes to the Issue #23 pool without blocking the control plane:
submit() returns immediately; wait() collects.

Demand-driven only: no prediction, no router changes, no kernels.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = "deepseeker.staging/v1"


class StagingError(RuntimeError):
    """Fetch failure: caller decides retry vs fallback (engine stays usable)."""


class Ticket:
    """Opaque handle for one in-flight expert fetch."""

    _ids = 0
    _lock = threading.Lock()

    def __init__(self, layer: int, expert: int) -> None:
        with Ticket._lock:
            Ticket._ids += 1
            self.id = Ticket._ids
        self.layer = layer
        self.expert = expert
        self.submitted_at = time.perf_counter()


class StagingEngine:
    """Bounded async staging with per-range pread workers."""

    def __init__(
        self,
        model_root: Path | str,
        manifest: dict,
        max_workers: int = 8,
        max_staging_bytes: int = 2 * 1024**3,
        reader=None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._root = Path(model_root)
        self._manifest = manifest
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers
        self._max_staging = max_staging_bytes
        self._reader = reader or self._cached_pread
        self._fd_cache: dict[Path, int] = {}
        self._fd_lock = threading.Lock()
        self._index = self._build_index(manifest)
        self._lock = threading.Lock()
        self._inflight: dict[tuple[int, int], concurrent.futures.Future] = {}
        self._tickets: dict[tuple[int, int], Ticket] = {}
        self._staged_bytes = 0
        self.submitted = 0
        self.completed = 0
        self.cancelled = 0
        self.dedup_hits = 0
        self.failures = 0
        self.queue_wait_s = 0.0
        self.service_s = 0.0
        self._shutdown = False

    @staticmethod
    def _build_index(manifest: dict) -> dict[tuple[int, int], list[tuple[str, int, int, int]]]:
        """(layer, expert) -> ordered ranges, built once (submit is hot)."""
        index: dict[tuple[int, int], list[tuple[str, int, int, int]]] = {}
        for tensor in manifest.get("tensors", []):
            if tensor.get("subsystem") == "routed-expert":
                start, end = tensor["file_range"]
                index.setdefault((tensor["layer"], tensor["expert"]), []).append(
                    (tensor["shard"], start, end, tensor["bytes"])
                )
        return index

    def expert_plan(self, layer: int, expert: int) -> list[tuple[str, int, int, int]]:
        """Ordered (shard, start, end, expected_bytes) ranges for one expert."""
        try:
            return self._index[(layer, expert)]
        except KeyError:
            raise KeyError(f"no tensors for expert {(layer, expert)}") from None

    @staticmethod
    def _pread_range(path: Path, start: int, size: int) -> bytes:
        fd = os.open(path, os.O_RDONLY)
        try:
            data = os.pread(fd, size, start)
        finally:
            os.close(fd)
        if len(data) != size:
            raise StagingError(f"short read {path.name}@{start}: {len(data)}/{size}")
        return data

    def _cached_pread(self, path: Path, start: int, size: int) -> bytes:
        """pread on a persistent per-shard fd (pread is thread-safe).

        open/close per range cost ~3x service time in measurement;
        the cache closes fds at shutdown.
        """
        with self._fd_lock:
            fd = self._fd_cache.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._fd_cache[path] = fd
        data = os.pread(fd, size, start)
        if len(data) != size:
            raise StagingError(f"short read {path.name}@{start}: {len(data)}/{size}")
        return data

    def _fetch(self, ticket: Ticket) -> np.ndarray:
        plan = self.expert_plan(ticket.layer, ticket.expert)
        parts = []
        for shard, start, end, expected in plan:
            data = self._reader(self._root / shard, start, end - start)
            if len(data) != expected:
                raise StagingError(
                    f"range mismatch {(ticket.layer, ticket.expert)}: "
                    f"got {len(data)}, manifest says {expected}"
                )
            parts.append(data)
        return np.frombuffer(b"".join(parts), dtype=np.uint8).copy()

    def plan_bytes(self, layer: int, expert: int) -> int:
        return sum(expected for _, _, _, expected in self.expert_plan(layer, expert))

    def submit(self, layer: int, expert: int) -> Ticket:
        """Queue a fetch; dedup to the in-flight ticket for the same expert.

        Reserves plan bytes up front: raises RuntimeError when the
        staging bound would break (caller waits for completions, then
        retries — never blocks inside the lock).
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("engine shut down")
            key = (layer, expert)
            future = self._inflight.get(key)
            if future is not None and not future.done():
                self.dedup_hits += 1
                return self._tickets[key]
            size = self.plan_bytes(layer, expert)
            if self._staged_bytes + size > self._max_staging:
                raise RuntimeError(
                    f"staging full ({self._staged_bytes}B + {size}B > {self._max_staging}B)"
                )
            ticket = Ticket(layer, expert)
            self._tickets[key] = ticket
            self._inflight[key] = self._pool.submit(self._run, ticket)
            self._staged_bytes += size
            self.submitted += 1
            return ticket

    def _run(self, ticket: Ticket) -> np.ndarray:
        queued = time.perf_counter() - ticket.submitted_at
        with self._lock:
            self.queue_wait_s += queued
        start = time.perf_counter()
        try:
            blob = self._fetch(ticket)
        except Exception as exc:
            with self._lock:
                self.failures += 1
            raise StagingError(str(exc)) from exc
        with self._lock:
            self.service_s += time.perf_counter() - start
            self.completed += 1
        return blob

    def wait(self, ticket: Ticket, timeout: float | None = None) -> np.ndarray:
        """Block for completion; releases the byte reservation on settle."""
        key = (ticket.layer, ticket.expert)
        with self._lock:
            future = self._inflight.get(key)
            if future is None or self._tickets.get(key) is not ticket:
                raise KeyError(f"unknown or settled ticket {ticket.id}")
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(f"ticket {ticket.id} timed out") from exc
        finally:
            with self._lock:
                if self._inflight.get(key) is future and future.done():
                    size = 0
                    try:
                        size = self.plan_bytes(ticket.layer, ticket.expert)
                    except KeyError:
                        pass
                    self._staged_bytes = max(0, self._staged_bytes - size)
                    del self._inflight[key]
                    self._tickets.pop(key, None)

    def cancel(self, ticket: Ticket) -> bool:
        """Best-effort cancel: unstarted work is dropped; in-flight runs to
        completion but its bytes are discarded (never admitted)."""
        key = (ticket.layer, ticket.expert)
        with self._lock:
            future = self._inflight.get(key)
            if future is None or self._tickets.get(key) is not ticket:
                return False
            cancelled = future.cancel()
            if cancelled:
                try:
                    size = self.plan_bytes(ticket.layer, ticket.expert)
                except KeyError:
                    size = 0
                self._staged_bytes = max(0, self._staged_bytes - size)
                del self._inflight[key]
                self._tickets.pop(key, None)
                self.cancelled += 1
            return cancelled

    def stage_into_pool(self, pool, layer: int, expert: int, timeout: float | None = None,
                        full_retries: int = 300):
        """Fetch and admit into an Issue #23 pool; returns evicted expert.

        A full staging bound spins briefly (background workers keep
        draining) instead of deadlocking the control plane.
        """
        ticket = None
        for _ in range(full_retries):
            try:
                ticket = self.submit(layer, expert)
                break
            except RuntimeError:
                time.sleep(0.1)
        if ticket is None:
            raise RuntimeError("staging full: retries exhausted")
        blob = self.wait(ticket, timeout=timeout)
        return pool.admit(layer, expert, blob)

    def metrics(self) -> dict:
        with self._lock:
            return {
                "schema": SCHEMA,
                "max_workers": self._max_workers,
                "submitted": self.submitted,
                "completed": self.completed,
                "cancelled": self.cancelled,
                "dedup_hits": self.dedup_hits,
                "failures": self.failures,
                "in_flight": len(self._inflight),
                "mean_queue_wait_s": self.queue_wait_s / max(1, self.completed),
                "mean_service_s": self.service_s / max(1, self.completed),
            }

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._fd_lock:
            for fd in self._fd_cache.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fd_cache.clear()
