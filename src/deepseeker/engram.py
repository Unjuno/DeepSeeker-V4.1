"""Engram row-level access model for DeepSeek-V4.1-Flash.

Pure-numpy reimplementation of the pinned upstream ``NgramHashState``
semantics (``inference/engram.py``), plus row -> safetensors file-range
mapping derived from the Issue #2 tensor manifest, plus working-set /
cache / read-amplification simulation.

No model weights are required. All row geometry is derived from manifest
dtype/shape metadata, never hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEAD = -1


# --------------------------------------------------------------------------
# Deterministic primality (exact for all n < 2^32 via bases 2/7/61)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small:
        return True
    if any(n % p == 0 for p in small):
        return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while candidate in seen or not is_prime(candidate):
        candidate += 1
    return candidate


# --------------------------------------------------------------------------
# Geometry (values carried in, asserted against upstream config)


@dataclass(frozen=True)
class EngramGeometry:
    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    num_embeddings: tuple[int, ...]
    vocab_size: int  # bucket prime search start (engram_vocab_size)
    compressed_vocab_size: int
    pad_id: int  # compressed pad id

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads


def build_primes(geometry: EngramGeometry) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Per-layer, per-ngram (2..max), per-head bucket moduli.

    Mirrors ``EngramLayout.from_args`` prime allocation order exactly,
    including the nested-tuple structure.
    """
    layers = []
    seen: set[int] = set()
    for _ in geometry.layer_ids:
        per_ngram = []
        for _ in range(geometry.max_ngram_size - 1):
            current = geometry.vocab_size - 1
            sizes = []
            for _ in range(geometry.n_heads):
                current = next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        layers.append(tuple(per_ngram))
    return tuple(layers)


def build_offsets(
    primes: tuple[tuple[tuple[int, ...], ...], ...],
) -> list[list[int]]:
    offsets = []
    for layer in primes:
        flat = [p for per_ngram in layer for p in per_ngram]
        acc = []
        total = 0
        for i, size in enumerate(flat):
            if i > 0:
                total += flat[i - 1]
            acc.append(total)
        offsets.append(acc)
    return offsets


def build_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab: int
) -> np.ndarray:
    """One odd multiplier per (layer, lookback); mirrors upstream."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // compressed_vocab) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows)


# --------------------------------------------------------------------------
# Hashing (mirrors NgramHashState.forward for one batch row)


def hash_positions(
    compressed: list[int],
    geometry: EngramGeometry,
    multipliers: np.ndarray,
    primes: list[list[list[int]]],
    offsets: list[list[int]],
    dead: set[int] | None = None,
) -> list[list[list[int]]]:
    """Return [position][layer][24 col] hash row ids.

    ``compressed`` is the full history of compressed token ids;
    positions before history start or marked dead hash with pad fill,
    exactly like the official cache/gather logic for start_pos=0.
    """
    dead = dead or set()
    n = len(compressed)
    out = []
    for pos in range(n):
        tokens = []
        blocked = False
        for shift in range(geometry.max_ngram_size):
            src_pos = pos - shift
            if src_pos < 0:
                blocked = True
                tokens.append(geometry.pad_id)
                continue
            src = compressed[src_pos]
            if src_pos in dead:
                blocked = True
                tokens.append(geometry.pad_id)
                continue
            if blocked:
                tokens.append(geometry.pad_id)
            else:
                tokens.append(src)
        pos_hashes = []
        for li in range(len(geometry.layer_ids)):
            rolling = tokens[0] * int(multipliers[li, 0])
            cols = []
            for i in range(1, geometry.max_ngram_size):
                rolling = rolling ^ (tokens[i] * int(multipliers[li, i]))
                for head in range(geometry.n_heads):
                    prime = primes[li][i - 1][head]
                    col = (i - 1) * geometry.n_heads + head
                    cols.append((rolling % prime) + offsets[li][col])
            pos_hashes.append(cols)
        out.append(pos_hashes)
    return out


# --------------------------------------------------------------------------
# Row -> manifest tensor / file-range mapping


@dataclass(frozen=True)
class TableRef:
    layer_id: int
    weight_name: str
    scale_name: str
    num_rows: int
    weight_row_bytes: int
    scale_row_bytes: int
    weight_data_start: int  # absolute file offset of data section start
    scale_data_start: int
    weight_shard: str
    scale_shard: str
    weight_file_range: tuple[int, int]
    scale_file_range: tuple[int, int]


def table_refs_from_manifest(manifest: dict) -> dict[int, TableRef]:
    """Derive per-layer lookup-table refs from manifest metadata."""
    by_name = {t["name"]: t for t in manifest["tensors"]}
    refs = {}
    layers = sorted(
        {
            t["layer"]
            for t in manifest["tensors"]
            if t["subsystem"] == "engram" and t["layer"] is not None and ".embed." in t["name"]
        }
    )
    for layer in layers:
        w = by_name[f"layers.{layer}.engram.embed.weight"]
        s = by_name[f"layers.{layer}.engram.embed.scale"]
        w_rows, w_cols = w["shape"]
        s_rows, s_cols = s["shape"]
        assert w_rows == s_rows, (w_rows, s_rows)
        assert w_cols == s_cols * 32, (w_cols, s_cols)
        w_row = w["bytes"] // w_rows
        s_row = s["bytes"] // s_rows
        assert w_row * w_rows == w["bytes"]
        assert s_row * s_rows == s["bytes"]
        refs[layer] = TableRef(
            layer_id=layer,
            weight_name=w["name"],
            scale_name=s["name"],
            num_rows=w_rows,
            weight_row_bytes=w_row,
            scale_row_bytes=s_row,
            weight_data_start=w["file_range"][0],
            scale_data_start=s["file_range"][0],
            weight_shard=w["shard"],
            scale_shard=s["shard"],
            weight_file_range=tuple(w["file_range"]),
            scale_file_range=tuple(s["file_range"]),
        )
    return refs


def row_ranges(ref: TableRef, row_id: int) -> dict:
    """Exact byte ranges for one (weight, scale) row pair."""
    if not 0 <= row_id < ref.num_rows:
        raise ValueError(f"row {row_id} out of range [0, {ref.num_rows})")
    w0 = ref.weight_data_start + row_id * ref.weight_row_bytes
    s0 = ref.scale_data_start + row_id * ref.scale_row_bytes
    return {
        "layer": ref.layer_id,
        "row": row_id,
        "weight": {
            "shard": ref.weight_shard,
            "range": [w0, w0 + ref.weight_row_bytes],
        },
        "scale": {
            "shard": ref.scale_shard,
            "range": [s0, s0 + ref.scale_row_bytes],
        },
        "useful_bytes": ref.weight_row_bytes + ref.scale_row_bytes,
    }


# --------------------------------------------------------------------------
# Locality / cache / amplification simulation


def reuse_distances(rows: list[int]) -> list[int | None]:
    """Per-access distance since previous identical access (None = first)."""
    last: dict[int, int] = {}
    out = []
    for i, row in enumerate(rows):
        out.append(None if row not in last else i - last[row])
        last[row] = i
    return out


def lru_hit_rate(requests: list, capacity_items: int) -> float:
    """Hit rate of an LRU cache holding capacity_items entries."""
    if capacity_items <= 0 or not requests:
        return 0.0
    from collections import OrderedDict

    cache: OrderedDict = OrderedDict()
    hits = 0
    for req in requests:
        if req in cache:
            hits += 1
            cache.move_to_end(req)
        else:
            if len(cache) >= capacity_items:
                cache.popitem(last=False)
            cache[req] = None
    return hits / len(requests)


def page_id(offset: int, page_bytes: int) -> int:
    return offset // page_bytes


def amplified_bytes(ranges: list[tuple[int, int]], page_bytes: int) -> tuple[int, int]:
    """Return (page_count, byte_count) covering ranges with aligned pages."""
    pages = set()
    for start, end in ranges:
        first, last = page_id(start, page_bytes), page_id(end - 1, page_bytes)
        pages.update(range(first, last + 1))
    return len(pages), len(pages) * page_bytes


def coalesce(ranges: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Sort by offset and merge ranges separated by <= max_gap bytes."""
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]
