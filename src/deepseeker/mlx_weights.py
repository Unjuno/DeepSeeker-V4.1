"""MLX-native runner weight layer: exact dequant + streaming loads.

On-disk formats (verified against inference/convert.py):
  fp4 experts (I8)   two e2m1fn nibbles/byte (low first) via FP4_TABLE,
                     one ue8m0 scale per 32 values along the last dim.
  fp8 dense (F8_E4M3) standard E4M3 values, one ue8m0 scale per 32x32
                     block (weight[i,j] uses scale[i//32, j//32]).
  ue8m0 scales       2^(bits-127) float; 0xFF -> NaN (unused in practice).
  bf16/f32           direct.

Strategy: dense (~9.5GB fp8 -> ~19GB bf16) dequantized once at startup
(fits); routed experts streamed per token through staged dequant
(#33 Metal kernel or this numpy reference); Engram rows dequantized
on lookup (#26 cache serves raw rows).
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from deepseeker.metal_fp4 import FP4_TABLE

E4M3_TABLE = np.zeros(256, dtype=np.float32)
for _b in range(256):
    _s = (_b >> 7) & 1
    _e = (_b >> 3) & 0xF
    _m = _b & 0x7
    if _e == 0xF:
        # OCP E4M3 (torch e4m3fn ground truth): only all-mantissa-ones
        # is NaN; the rest of the exp-15 row is 256*(1+m/8) (bias 7).
        E4M3_TABLE[_b] = np.nan if _m == 0x7 else ((-1) ** _s) * 256.0 * (1.0 + _m / 8.0)
    elif _e == 0:
        E4M3_TABLE[_b] = ((-1) ** _s) * (_m / 8.0) * 2.0**-6
    else:
        E4M3_TABLE[_b] = ((-1) ** _s) * (1.0 + _m / 8.0) * 2.0 ** (_e - 7)
del _b, _s, _e, _m


def _check_e4m3_table() -> None:
    """Cross-check the full table against torch once at import."""
    import torch as _torch

    want = _torch.frombuffer(bytearray(range(256)), dtype=_torch.float8_e4m3fn).float().numpy()
    got = E4M3_TABLE
    both_nan = np.isnan(want) & np.isnan(got)
    same = both_nan | (want == got)
    if not same.all():
        bad = np.where(~same)[0]
        raise RuntimeError(f"E4M3_TABLE mismatch at bytes {bad[:8].tolist()}")


_check_e4m3_table()


def decode_u8m0(raw: np.ndarray) -> np.ndarray:
    """ue8m0 bytes -> float32 powers of two."""
    bits = raw.astype(np.int32)
    out = np.exp2(bits - 127).astype(np.float32)
    out[bits == 0xFF] = np.nan
    return out


def dequant_fp8_block(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """[R, C] e4m3 bytes + [R//32, C//32] ue8m0 scales -> float32."""
    rows, cols = packed.shape
    assert rows % 32 == 0 and cols % 32 == 0, (packed.shape, scales.shape)
    assert scales.shape == (rows // 32, cols // 32), (packed.shape, scales.shape)
    values = E4M3_TABLE[packed.astype(np.int64)]
    gain = np.repeat(np.repeat(decode_u8m0(scales), 32, axis=0), 32, axis=1)
    return values * gain


def dequant_fp4_rowmajor(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Flat fp4 pairs (low nibble first) + per-32 ue8m0 scales -> float32."""
    n = packed.size * 2
    assert scales.size == (n + 31) // 32, (packed.size, scales.size)
    raw = packed.astype(np.int64)
    vals = np.empty(n, dtype=np.float32)
    vals[0::2] = FP4_TABLE[raw & 0x0F]
    vals[1::2] = FP4_TABLE[(raw >> 4) & 0x0F]
    return vals * np.repeat(decode_u8m0(scales), 32)[:n]


def read_bytes(model_root: Path, shard: str, start: int, end: int) -> bytes:
    from deepseeker.ssd_policy import open_backing_file

    size = end - start
    fd = open_backing_file(model_root, shard)
    try:
        data = os.pread(fd, size, start)
    finally:
        os.close(fd)
    if len(data) != size:
        raise RuntimeError(f"short read {shard}@{start}")
    return data


class ExpertLoader:
    """On-demand fp4 expert -> bf16 MLX arrays (w1/w2/w3 folded scales)."""

    def __init__(self, model_root: Path | str, manifest: dict) -> None:
        self._root = Path(model_root)
        by_name = {t["name"]: t for t in manifest.get("tensors", [])}
        self._by_name = by_name

    def _tensor(self, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
        entry = self._by_name[name]
        data = read_bytes(self._root, entry["shard"], *entry["file_range"])
        return np.frombuffer(data, dtype=np.uint8).copy(), tuple(entry["shape"])

    def load_expert(self, layer: int, expert: int) -> dict[str, mx.array]:
        """Dequantized {w1, w2, w3} bf16 arrays with scales folded in."""
        out = {}
        for role in ("w1", "w2", "w3"):
            w = self._by_name[f"layers.{layer}.ffn.experts.{expert}.{role}.weight"]
            s = self._by_name[f"layers.{layer}.ffn.experts.{expert}.{role}.scale"]
            wdata = read_bytes(self._root, w["shard"], *w["file_range"])
            sdata = read_bytes(self._root, s["shard"], *s["file_range"])
            rows, pair_cols = w["shape"]
            vals = dequant_fp4_rowmajor(
                np.frombuffer(wdata, dtype=np.uint8).copy(),
                np.frombuffer(sdata, dtype=np.uint8).copy(),
            ).reshape(rows, pair_cols * 2)
            out[role] = mx.array(vals).astype(mx.bfloat16)
        mx.eval(list(out.values()))
        return out


class DenseLoader:
    """One-shot dequant of resident weights to bf16 (skips streamed tables)."""

    SKIP_SUBSYSTEMS = ("routed-expert", "mtp-expert", "vision")

    def __init__(self, model_root: Path | str, manifest: dict) -> None:
        self._root = Path(model_root)
        self._manifest = manifest

    def load_all(self) -> dict[str, mx.array]:
        out: dict[str, mx.array] = {}
        for tensor in self._manifest.get("tensors", []):
            if tensor.get("subsystem") in self.SKIP_SUBSYSTEMS:
                continue
            if ".engram.embed." in tensor.get("name", ""):
                continue  # streamed rows via EngramRowCache, never resident
            if tensor["name"].endswith(".scale"):
                continue  # consumed with their fp8/fp4 weight, never standalone
            out[tensor["name"]] = self.load_one(tensor["name"])
        mx.eval(list(out.values()))
        return out

    def load_one(self, name: str) -> mx.array:
        by_name = {t["name"]: t for t in self._manifest.get("tensors", [])}
        entry = by_name[name]
        data = read_bytes(self._root, entry["shard"], *entry["file_range"])
        dtype = entry["dtype"]
        shape = tuple(entry["shape"])
        if dtype == "BF16":
            t = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape)
            return mx.array(t.float().numpy()).astype(mx.bfloat16)
        if dtype == "F32":
            return mx.array(np.frombuffer(data, dtype="<f4").copy().reshape(shape))
        if dtype == "F8_E4M3":
            scale_name = name.replace(".weight", ".scale")
            scale = by_name.get(scale_name)
            if scale is None:
                raise KeyError(f"no scale for {name}")
            sdata = read_bytes(self._root, scale["shard"], *scale["file_range"])
            vals = dequant_fp8_block(
                np.frombuffer(data, dtype=np.uint8).copy().reshape(shape),
                np.frombuffer(sdata, dtype=np.uint8).copy().reshape(scale["shape"]),
            )
            return mx.array(vals).astype(mx.bfloat16)
        raise ValueError(f"unsupported dtype {dtype} for {name}")
