"""Issue #23: bounded Unified Memory resident expert pool (roadmap P4).

Unlike the Issue #11 simulator, this pool holds REAL expert bytes read
from the verified checkpoint via manifest file_ranges. Numpy uint8
buffers live in the same unified physical memory the GPU will later
read; GPU-side residency arrives in P7.

Guarantees (all tested):
  hard ceiling  bytes_resident never exceeds slots x max_expert_bytes
  no stale use  lookup returns bytes byte-equal to a fresh pread, and
                evicted entries are unreachable
  pin safety    pinned entries are never evicted; unpin re-admits them
                to normal eviction order
  miss fallback demand paging: miss -> loader -> admit -> serve

Demand-only policies: LRU (default) / LFU / FIFO by insertion order.
No prefetch, no router changes, no kernels.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

POLICIES = ("lru", "lfu", "fifo")


class ResidentPool:
    """Fixed per-layer slot pool over opaque expert byte blobs."""

    def __init__(
        self,
        slots_per_layer: int | dict[int, int],
        max_expert_bytes: int,
        policy: str = "lru",
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
        if isinstance(slots_per_layer, int):
            if slots_per_layer < 1:
                raise ValueError("slots_per_layer must be >= 1")
            self._caps: dict[int, int] = {}
            self._default = slots_per_layer
        else:
            if any(s < 1 for s in slots_per_layer.values()):
                raise ValueError("per-layer slots must be >= 1")
            self._caps = dict(slots_per_layer)
            self._default = 1
        if max_expert_bytes < 1:
            raise ValueError("max_expert_bytes must be >= 1")
        self._max_bytes = max_expert_bytes
        self._policy = policy
        self._slots: dict[int, OrderedDict[int, np.ndarray]] = {}
        self._pinned: set[tuple[int, int]] = set()
        self._freq: dict[tuple[int, int], int] = {}
        self._order: list[tuple[int, int]] = []
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bytes_resident = 0
        self.high_water_bytes = 0

    def cap(self, layer: int) -> int:
        return self._caps.get(layer, self._default)

    def set_cap(self, layer: int, cap: int) -> int:
        """Lower (or raise) a layer cap, evicting LRU excess. Returns evicted count."""
        if cap < 1:
            raise ValueError("cap must be >= 1")
        self._caps[layer] = cap
        evicted = 0
        bucket = self._slots.get(layer, OrderedDict())
        while len(bucket) > cap:
            victim = self._victim(layer)
            if victim is None:
                break
            self._evict(layer, victim)
            evicted += 1
        self._check_ceiling()
        return evicted

    def _ceiling(self) -> int:
        return sum(self.cap(layer) for layer in self._slots) * self._max_bytes

    def _check_ceiling(self) -> None:
        assert self.bytes_resident <= self._ceiling(), "byte ceiling violated"

    def contains(self, layer: int, expert: int) -> bool:
        """Non-counting residency probe (prefetch skip-checks must not pollute metrics)."""
        bucket = self._slots.get(layer)
        return bucket is not None and expert in bucket

    def lookup(self, layer: int, expert: int) -> np.ndarray | None:
        """Return resident bytes or None (miss). Touches recency."""
        bucket = self._slots.get(layer)
        if bucket is None or expert not in bucket:
            self.misses += 1
            return None
        self.hits += 1
        if self._policy == "lru":
            bucket.move_to_end(expert)
        self._freq[(layer, expert)] = self._freq.get((layer, expert), 0) + 1
        return bucket[expert]

    def _victim(self, layer: int) -> int | None:
        bucket = self._slots.get(layer, OrderedDict())
        cands = [e for e in bucket if (layer, e) not in self._pinned]
        if not cands:
            return None
        if self._policy == "lfu":
            return min(cands, key=lambda e: (self._freq.get((layer, e), 0), e))
        if self._policy == "fifo":
            for key in self._order:
                if key[0] == layer and key[1] in bucket and key not in self._pinned:
                    return key[1]
            return cands[0]
        return cands[0]  # lru: OrderedDict front is least recent

    def admit(self, layer: int, expert: int, blob: np.ndarray) -> int | None:
        """Store blob; evict one victim if full. Returns evicted expert."""
        if blob.nbytes > self._max_bytes:
            raise ValueError(f"blob {blob.nbytes}B exceeds max {self._max_bytes}B")
        bucket = self._slots.setdefault(layer, OrderedDict())
        if expert in bucket:
            if self._policy == "lru":
                bucket.move_to_end(expert)
            return None
        evicted = None
        if len(bucket) >= self.cap(layer):
            evicted = self._victim(layer)
            if evicted is None:
                raise RuntimeError(f"layer {layer}: all slots pinned, cannot admit")
            self._evict(layer, evicted)
        bucket[expert] = blob
        self._order.append((layer, expert))
        self.bytes_resident += int(blob.nbytes)
        self.high_water_bytes = max(self.high_water_bytes, self.bytes_resident)
        self._check_ceiling()
        return evicted

    def _evict(self, layer: int, expert: int) -> None:
        blob = self._slots[layer].pop(expert)
        self.bytes_resident -= int(blob.nbytes)
        self.evictions += 1
        self._freq.pop((layer, expert), None)
        self._order = [key for key in self._order if key != (layer, expert)]

    def pin(self, layer: int, expert: int) -> None:
        if expert not in self._slots.get(layer, {}):
            raise KeyError(f"cannot pin non-resident {(layer, expert)}")
        self._pinned.add((layer, expert))

    def unpin(self, layer: int, expert: int) -> None:
        self._pinned.discard((layer, expert))

    def demand(self, layer: int, expert: int, loader) -> np.ndarray:
        """Lookup with miss fallback: load, admit, serve."""
        hit = self.lookup(layer, expert)
        if hit is not None:
            return hit
        return self._admit_loaded(layer, expert, loader(layer, expert))

    def _admit_loaded(self, layer: int, expert: int, blob: np.ndarray) -> np.ndarray:
        self.admit(layer, expert, blob)
        found = self.lookup(layer, expert)
        assert found is not None
        return found

    def metrics(self) -> dict:
        total = self.hits + self.misses
        return {
            "policy": self._policy,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "evictions": self.evictions,
            "bytes_resident": self.bytes_resident,
            "high_water_bytes": self.high_water_bytes,
            "pinned": len(self._pinned),
        }


def load_expert_bytes(model_root, manifest: dict, layer: int, expert: int):
    """Read one expert's full tensor bytes from checkpoint shards."""
    import os
    from pathlib import Path

    blobs = []
    for tensor in manifest.get("tensors", []):
        if (
            tensor.get("subsystem") == "routed-expert"
            and tensor.get("layer") == layer
            and tensor.get("expert") == expert
        ):
            start, end = tensor["file_range"]
            fd = os.open(Path(model_root) / tensor["shard"], os.O_RDONLY)
            try:
                data = os.pread(fd, end - start, start)
            finally:
                os.close(fd)
            if len(data) != end - start:
                raise RuntimeError(f"short read {tensor['name']}")
            blobs.append(data)
    if not blobs:
        raise KeyError(f"no tensors for expert {(layer, expert)}")
    out = np.frombuffer(b"".join(blobs), dtype=np.uint8).copy()
    return out
