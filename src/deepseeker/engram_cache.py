"""Issue #26: bounded Engram row/page cache runtime (roadmap P5).

Serves exact (weight, scale) row pairs from the real 189 GiB lookup
tables through a page-granular LRU cache (4 KiB / 16 KiB modes).
Row geometry comes from Issue #3 refs (manifest-derived, never
hardcoded); Engram math and hashes are untouched.

A row is 256 fp8 weight bytes + 8 scale bytes = 264 useful bytes,
usually spanning two pages: page cache + coalesced reads are the
whole game. Async fetch via worker pool; validation compares
assembled rows against direct preads.
"""

from __future__ import annotations

import concurrent.futures
import os
from collections import OrderedDict
from pathlib import Path

from deepseeker.engram import TableRef, page_id, row_ranges

SCHEMA = "deepseeker.engram-cache/v1"


class EngramRowCache:
    """LRU page cache fronting exact row assembly."""

    def __init__(
        self,
        model_root: Path | str,
        refs: dict[int, TableRef],
        capacity_bytes: int = 64 * 1024**2,
        page_bytes: int = 4096,
        max_workers: int = 4,
        reader=None,
    ) -> None:
        if capacity_bytes < page_bytes:
            raise ValueError("capacity must hold at least one page")
        if page_bytes not in (4096, 16384):
            raise ValueError("page_bytes must be 4096 or 16384")
        self._root = Path(model_root)
        self._refs = refs
        self._capacity = capacity_bytes
        self._page = page_bytes
        self._reader = reader or self._pread
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._pages: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._page_bytes_resident = 0
        self.page_hits = 0
        self.page_misses = 0
        self.bytes_read = 0
        self.rows_served = 0

    def _pread(self, shard: str, start: int, size: int) -> bytes:
        fd = os.open(self._root / shard, os.O_RDONLY)
        try:
            data = os.pread(fd, size, start)
        finally:
            os.close(fd)
        if len(data) != size:
            raise RuntimeError(f"short read {shard}@{start}")
        return data

    def _page_key(self, shard: str, offset: int) -> tuple[str, int]:
        return (shard, page_id(offset, self._page))

    def _get_page(self, shard: str, offset: int) -> bytes:
        key = self._page_key(shard, offset)
        page = self._pages.get(key)
        if page is not None:
            self.page_hits += 1
            self._pages.move_to_end(key)
            return page
        self.page_misses += 1
        base = key[1] * self._page
        page = self._reader(shard, base, self._page)
        self._pages[key] = page
        self._page_bytes_resident += len(page)
        while self._page_bytes_resident > self._capacity:
            _, old = self._pages.popitem(last=False)
            self._page_bytes_resident -= len(old)
        self.bytes_read += len(page)
        return page

    def fetch_rows(self, layer: int, row_ids: list[int]) -> list[tuple[bytes, bytes]]:
        """Assemble exact (weight, scale) pairs, fetching missing pages async.

        Hit accounting counts only reuse: pages already resident BEFORE
        this call. Freshly fetched pages count as misses even though the
        subsequent slice finds them present.
        """
        ref = self._refs[layer]
        resident_before = set(self._pages)
        needed: dict[tuple[str, int], None] = {}
        pages_per_row: list[list[tuple[str, int]]] = []
        for row_id in row_ids:
            info = row_ranges(ref, row_id)
            row_pages = []
            for part in ("weight", "scale"):
                shard = info[part]["shard"]
                start, end = info[part]["range"]
                first, last = start // self._page, (end - 1) // self._page
                for page in range(first, last + 1):
                    key = (shard, page)
                    row_pages.append(key)
                    if key not in self._pages and key not in needed:
                        needed[key] = None
            pages_per_row.append(row_pages)
        if needed:
            futures = {
                self._pool.submit(self._get_page_uncached, shard, page): (shard, page)
                for shard, page in needed
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
            self.page_misses += len(needed)
            self._page_bytes_resident = sum(map(len, self._pages.values()))
            self._enforce_capacity()
        out = []
        for row_id, row_pages in zip(row_ids, pages_per_row):
            info = row_ranges(ref, row_id)
            w = self._slice_present(info["weight"]["shard"], *info["weight"]["range"])
            s = self._slice_present(info["scale"]["shard"], *info["scale"]["range"])
            out.append((w, s))
            for key in row_pages:
                if key in resident_before:
                    self.page_hits += 1
        self.rows_served += len(row_ids)
        return out

    def _get_page_uncached(self, shard: str, page: int) -> None:
        # Worker-side insert only (distinct keys per batch); capacity is
        # enforced on the calling thread after completion (see fetch_rows),
        # so the bound holds at every call boundary.
        base = page * self._page
        self._pages[(shard, page)] = self._reader(shard, base, self._page)
        self.bytes_read += self._page

    def _enforce_capacity(self) -> None:
        while self._page_bytes_resident > self._capacity and self._pages:
            _, old = self._pages.popitem(last=False)
            self._page_bytes_resident -= len(old)

    def _slice_present(self, shard: str, start: int, end: int) -> bytes:
        """Assemble from resident pages (all prefetched by the caller)."""
        first, last = start // self._page, (end - 1) // self._page
        chunks = []
        for page in range(first, last + 1):
            base = page * self._page
            try:
                page_bytes = self._pages[(shard, page)]
            except KeyError:
                # Evicted between fetch and slice (call exceeds capacity):
                # serve directly without polluting reuse accounting.
                page_bytes = self._reader(shard, base, self._page)
                self.bytes_read += len(page_bytes)
            lo = max(start - base, 0)
            hi = min(end - base, self._page)
            chunks.append(page_bytes[lo:hi])
        return b"".join(chunks)

    def _slice(self, shard: str, start: int, end: int) -> bytes:
        first, last = start // self._page, (end - 1) // self._page
        chunks = []
        for page in range(first, last + 1):
            base = page * self._page
            page_bytes = self._get_page(shard, base)
            lo = max(start - base, 0)
            hi = min(end - base, self._page)
            chunks.append(page_bytes[lo:hi])
        return b"".join(chunks)

    def resize_capacity(self, capacity_bytes: int) -> None:
        """Lower (or raise) the byte ceiling, evicting LRU excess now."""
        if capacity_bytes < self._page:
            raise ValueError("capacity must hold at least one page")
        self._capacity = capacity_bytes
        self._enforce_capacity()

    def metrics(self) -> dict:
        total = self.page_hits + self.page_misses
        return {
            "schema": SCHEMA,
            "capacity_bytes": self._capacity,
            "page_bytes": self._page,
            "page_hits": self.page_hits,
            "page_misses": self.page_misses,
            "hit_rate": self.page_hits / total if total else 0.0,
            "bytes_read": self.bytes_read,
            "rows_served": self.rows_served,
            "bytes_resident": self._page_bytes_resident,
        }

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


def access_stream(num_rows: int, ops: int, seed: int, hotspot_frac: float = 0.2) -> list[int]:
    """Deterministic row stream: hotspot_frac of ops hit 1% of rows."""
    import random

    rng = random.Random(seed)
    hot = max(1, num_rows // 100)
    out = []
    for _ in range(ops):
        if rng.random() < hotspot_frac:
            out.append(rng.randrange(hot))
        else:
            out.append(rng.randrange(num_rows))
    return out
