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
    """Flat fp4 pairs (low nibble first) + per-32 ue8m0 scales -> float32.

    Block multiply without materializing a full np.repeat scale vector.
    """
    n = packed.size * 2
    n_scales = (n + 31) // 32
    assert scales.size == n_scales, (packed.size, scales.size, n)
    raw = packed.astype(np.int32, copy=False)
    vals = np.empty(n, dtype=np.float32)
    vals[0::2] = FP4_TABLE[raw & 0x0F]
    vals[1::2] = FP4_TABLE[(raw >> 4) & 0x0F]
    gain = decode_u8m0(scales)
    # Fold scales per 32-element block without a full repeat vector.
    n_blocks = n // 32
    vals[: n_blocks * 32] *= np.repeat(gain[:n_blocks], 32)
    if n_blocks * 32 < n:
        vals[n_blocks * 32:] *= gain[n_blocks]
    return vals


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
        self._metal_kernel = None
        self._metal_lut = None

    def _kernel(self):
        if self._metal_kernel is None:
            from deepseeker.metal_fp4 import _kernel_fn
            self._metal_kernel, self._metal_lut = _kernel_fn()
        return self._metal_kernel, self._metal_lut

    def _tensor(self, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
        entry = self._by_name[name]
        data = read_bytes(self._root, entry["shard"], *entry["file_range"])
        return np.frombuffer(data, dtype=np.uint8).copy(), tuple(entry["shape"])

    def _read_role(self, layer: int, expert: int, prefix: str, role: str):
        base = f"{prefix}.{layer}.ffn.experts.{expert}.{role}" if prefix == "layers" \
            else f"mtp.{layer}.ffn.experts.{expert}.{role}"
        w = self._by_name[f"{base}.weight"]
        s = self._by_name[f"{base}.scale"]
        wdata = read_bytes(self._root, w["shard"], *w["file_range"])
        sdata = read_bytes(self._root, s["shard"], *s["file_range"])
        return role, wdata, sdata, tuple(w["shape"])

    def _dequant_metal(self, wdata: bytes, sdata: bytes,
                       rows: int, pair_cols: int) -> mx.array:
        """fp4 bytes -> bf16 on GPU (Metal LUT kernel; ~10x numpy)."""
        from deepseeker.metal_fp4 import decode_scales_u8m0

        n = rows * pair_cols * 2
        kernel, lut = self._kernel()
        mp = mx.array(np.frombuffer(wdata, dtype=np.uint8))
        scales = decode_scales_u8m0(sdata)
        ms = mx.array(np.ascontiguousarray(scales, dtype=np.float32))
        outs = kernel(
            inputs=[mp, ms, lut],
            output_shapes=[(n,)],
            output_dtypes=[mx.float32],
            grid=(n, 1, 1),
            threadgroup=(min(n, 256), 1, 1),
        )
        return outs[0].reshape(rows, pair_cols * 2).astype(mx.bfloat16)

    def load_expert(self, layer: int, expert: int, prefix: str = "layers") -> dict[str, mx.array]:
        """Dequantized {w1, w2, w3} bf16 arrays with scales folded in.

        prefix='layers' for backbone experts; prefix='mtp' uses mtp.{layer}.ffn.
        Parallel I/O for the three tensors, Metal dequant on GPU.
        """
        from concurrent.futures import ThreadPoolExecutor

        roles = ("w1", "w2", "w3")
        with ThreadPoolExecutor(max_workers=3) as pool:
            parts = list(pool.map(
                lambda r: self._read_role(layer, expert, prefix, r), roles))
        out: dict[str, mx.array] = {}
        for role, wdata, sdata, (rows, pair_cols) in parts:
            out[role] = self._dequant_metal(wdata, sdata, rows, pair_cols)
        mx.eval(list(out.values()))
        return out

    def load_experts_parallel(self, specs: list[tuple[int, int, str]], workers: int = 8) -> dict[tuple[str, int, int], dict[str, mx.array]]:
        """Load many experts: batch I/O then sequential Metal dequant.

        specs: list of (layer, expert, prefix). Returns key=(prefix, layer, expert) -> dict.
        Metal kernels serialize on the GPU, so overlap disk reads across
        experts first, then dequant one expert at a time.
        """
        if not specs:
            return {}
        from concurrent.futures import ThreadPoolExecutor

        def _read(spec: tuple[int, int, str]):
            layer, expert, prefix = spec
            parts = [
                self._read_role(layer, expert, prefix, r)
                for r in ("w1", "w2", "w3")
            ]
            return (prefix, layer, expert), parts

        workers = max(1, min(workers, len(specs)))
        if workers == 1:
            reads = [_read(s) for s in specs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                reads = list(pool.map(_read, specs))
        out: dict[tuple[str, int, int], dict[str, mx.array]] = {}
        for key, parts in reads:
            expert_out: dict[str, mx.array] = {}
            for role, wdata, sdata, (rows, pair_cols) in parts:
                expert_out[role] = self._dequant_metal(wdata, sdata, rows, pair_cols)
            out[key] = expert_out
            mx.eval(list(expert_out.values()))
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
