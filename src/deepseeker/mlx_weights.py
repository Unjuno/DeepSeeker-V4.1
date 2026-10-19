"""MLX-native runner weight layer: exact dequant + streaming loads.

On-disk formats (verified against inference/convert.py):
  fp4 experts (I8)   two e2m1fn nibbles/byte (low first) via FP4_TABLE,
                     one ue8m0 scale per 32 values along the last dim.
  fp8 dense (F8_E4M3) standard E4M3 values, one ue8m0 scale per 32x32
                     block (weight[i,j] uses scale[i//32, j//32]).
  ue8m0 scales       2^(bits-127) float; 0xFF -> NaN (unused in practice).
  bf16/f32           direct.

Strategy: dense (~9.5GB fp8 -> ~19GB bf16) dequantized once at startup
(fits); routed experts streamed per token in packed mxfp4 and consumed
directly by mx.quantized_matmul (no dequant, 3.76x smaller cache footprint);
Engram rows dequantized on lookup (#26 cache serves raw rows).
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


_U8M0_LUT: np.ndarray | None = None


def decode_u8m0(raw: np.ndarray) -> np.ndarray:
    """ue8m0 bytes -> float32 powers of two."""
    global _U8M0_LUT
    if raw.dtype == np.uint8:
        if _U8M0_LUT is None:
            bits = np.arange(256, dtype=np.int32)
            lut = np.exp2(bits - 127)
            lut[0xFF] = np.nan
            _U8M0_LUT = lut.astype(np.float32)
        return _U8M0_LUT[raw]
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


def pack_fp4_mxfp4(wdata: bytes, sdata: bytes, rows: int,
                   pair_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """On-disk fp4 bytes -> MLX mxfp4 operands (uint32 weights, uint8 scales).

    The checkpoint stores 8 e2m1 values per 4 bytes (low nibble first) with
    one ue8m0 scale per 32 values along the last dim. MLX mxfp4 packs exactly
    the same way -- 8 values per uint32 word, first value in the low nibble,
    one uint8 e8m0 scale per 32 values -- so the weights only need a
    little-endian reinterpretation, no arithmetic. Verified byte-exact against
    `reference_dequant` + `mx.dequantize(mode="mxfp4")` (max abs diff 0).
    """
    if pair_cols % 4:
        raise ValueError(f"pair_cols {pair_cols} not divisible by 4")
    w = np.frombuffer(wdata, dtype=np.uint8).reshape(rows, pair_cols)
    n_scales = pair_cols // 16  # pair_cols*2 values / 32 per scale
    s = np.frombuffer(sdata, dtype=np.uint8)
    if s.size != rows * n_scales:
        raise ValueError(f"scale bytes {s.size} != {rows * n_scales}")
    return w.view(np.uint32), s.reshape(rows, n_scales)


def dequant_packed(packed: mx.array, scales: mx.array) -> mx.array:
    """mxfp4 operands -> bf16 (staging path for A/B profiles and tests)."""
    return mx.dequantize(packed, scales=scales, group_size=32, bits=4,
                         mode="mxfp4").astype(mx.bfloat16)


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


_FD_CACHE: dict[tuple[str, str], int] = {}
_FD_LOCK = threading.Lock()


def read_bytes(model_root: Path, shard: str, start: int, end: int) -> bytes:
    from deepseeker.ssd_policy import open_backing_file

    size = end - start
    key = (str(model_root), shard)
    fd = _FD_CACHE.get(key)
    if fd is None:
        with _FD_LOCK:
            fd = _FD_CACHE.get(key)
            if fd is None:
                fd = open_backing_file(model_root, shard)
                _FD_CACHE[key] = fd
    data = os.pread(fd, size, start)
    if len(data) != size:
        raise RuntimeError(f"short read {shard}@{start}")
    return data


def read_into(model_root: Path, shard: str, start: int,
              target: memoryview) -> int:
    """Read a shard range directly into a writable contiguous buffer."""
    view = memoryview(target)
    if view.readonly or not view.c_contiguous:
        raise ValueError("read target must be writable and C-contiguous")
    view = view.cast("B")
    key = (str(model_root), shard)
    fd = _FD_CACHE.get(key)
    if fd is None:
        with _FD_LOCK:
            fd = _FD_CACHE.get(key)
            if fd is None:
                from deepseeker.ssd_policy import open_backing_file
                fd = open_backing_file(model_root, shard)
                _FD_CACHE[key] = fd
    done = 0
    while done < view.nbytes:
        if hasattr(os, "preadv"):
            n = os.preadv(fd, [view[done:]], start + done)
        else:
            chunk = os.pread(fd, view.nbytes - done, start + done)
            n = len(chunk)
            view[done:done + n] = chunk
        if n <= 0:
            raise RuntimeError(f"short read {shard}@{start + done}")
        done += n
    return done


class ExpertLoader:
    """On-demand fp4 expert -> packed mxfp4 operand pairs (w1/w2/w3)."""

    def __init__(self, model_root: Path | str, manifest: dict) -> None:
        self._root = Path(model_root)
        by_name = {t["name"]: t for t in manifest.get("tensors", [])}
        self._by_name = by_name
        self._metal_kernel = None
        self._metal_lut = None
        self._metal_kernel_bf16 = None
        io_threads = int(os.environ.get("DEEPSEEKER_IO_THREADS", "16"))
        self._io_pool = ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="expert-io")
        self.direct_pread = os.environ.get("DEEPSEEKER_DIRECT_PREAD") == "1"
        self._run_cache: dict[tuple[str, int, int], list[dict]] = {}
        self.io_s = 0.0
        self.dequant_s = 0.0
        # Worker-summed timings for the opt-in per-expert async MoE path.
        # These are aggregate worker seconds (and may overlap), not wall time.
        self.pipeline_read_worker_s = 0.0
        self.pipeline_pack_worker_s = 0.0
        self.pre_route_read_worker_s = 0.0
        self.pre_route_pack_worker_s = 0.0
        self.eval_s = 0.0
        self.bytes_read = 0
        self.load_calls = 0
        self.eval_times: list[float] = []
        self.batch_sizes: list[int] = []

    def _kernel(self):
        if self._metal_kernel is None:
            from deepseeker.metal_fp4 import _kernel_fn
            self._metal_kernel, self._metal_lut = _kernel_fn()
        return self._metal_kernel, self._metal_lut

    def _kernel_bf16(self):
        if self._metal_kernel_bf16 is None:
            from deepseeker.metal_fp4 import _kernel_bf16_u8_fn
            (self._metal_kernel_bf16,
             self._metal_lut,
             self._metal_scale_lut) = _kernel_bf16_u8_fn()
        return self._metal_kernel_bf16, self._metal_lut, self._metal_scale_lut

    def _tensor(self, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
        entry = self._by_name[name]
        data = read_bytes(self._root, entry["shard"], *entry["file_range"])
        return np.frombuffer(data, dtype=np.uint8).copy(), tuple(entry["shape"])

    def _expert_name(self, layer: int, expert: int, prefix: str, role: str) -> str:
        if prefix == "layers":
            return f"layers.{layer}.ffn.experts.{expert}.{role}"
        return f"mtp.{layer}.ffn.experts.{expert}.{role}"

    def _expert_runs(self, prefix: str, layer: int, expert: int) -> list[dict]:
        """Contiguous on-disk byte runs covering one expert.

        The checkpoint stores an expert's three weights back to back and its
        three scales back to back, so the six tensor reads collapse into two
        `pread`s (one ~17.7 MiB weight run, one ~1.1 MiB scale run) instead of
        six syscalls. Ranges that are not contiguous stay in their own run.
        """
        key = (prefix, layer, expert)
        cached = self._run_cache.get(key)
        if cached is not None:
            return cached
        ents: list[tuple[str, str, dict]] = []
        for role in ("w1", "w2", "w3"):
            base = self._expert_name(layer, expert, prefix, role)
            ents.append((role, "weight", self._by_name[f"{base}.weight"]))
            ents.append((role, "scale", self._by_name[f"{base}.scale"]))
        ents.sort(key=lambda t: (t[2]["shard"], t[2]["file_range"][0]))
        runs: list[dict] = []
        for role, kind, e in ents:
            start, end = e["file_range"]
            if runs and runs[-1]["shard"] == e["shard"] and runs[-1]["end"] == start:
                runs[-1]["end"] = end
                runs[-1]["parts"].append((role, kind, start - runs[-1]["start"], e))
            else:
                runs.append({"shard": e["shard"], "start": start, "end": end,
                             "parts": [(role, kind, 0, e)]})
        self._run_cache[key] = runs
        return runs

    def _read_expert_parts(self, layer: int, expert: int, prefix: str,
                           parallel: bool = False):
        """(role, weight_mv, scale_mv, shape) for w1/w2/w3 with merged reads."""
        runs = self._expert_runs(prefix, layer, expert)
        if parallel and len(runs) > 1:
            blobs = list(self._io_pool.map(
                lambda r: read_bytes(self._root, r["shard"],
                                     r["start"], r["end"]), runs))
        else:
            blobs = [read_bytes(self._root, r["shard"],
                                r["start"], r["end"]) for r in runs]
        out: dict[str, dict] = {}
        for run, blob in zip(runs, blobs):
            mv = memoryview(blob)
            for role, kind, off, e in run["parts"]:
                slot = out.setdefault(role, {})
                slot[kind] = mv[off:off + e["bytes"]]
                if kind == "weight":
                    slot["shape"] = tuple(e["shape"])
        return [(r, out[r]["weight"], out[r]["scale"], out[r]["shape"])
                for r in ("w1", "w2", "w3")]

    def _read_role(self, layer: int, expert: int, prefix: str, role: str):
        base = self._expert_name(layer, expert, prefix, role)
        w = self._by_name[f"{base}.weight"]
        s = self._by_name[f"{base}.scale"]
        wdata = read_bytes(self._root, w["shard"], *w["file_range"])
        sdata = read_bytes(self._root, s["shard"], *s["file_range"])
        return role, wdata, sdata, tuple(w["shape"])

    def _dequant_metal(self, wdata: bytes, sdata: bytes,
                       rows: int, pair_cols: int) -> mx.array:
        """fp4 bytes -> bf16 on GPU (staging path for A/B profiles/tests).

        The decode path no longer dequantizes (packed experts go straight to
        quantized_matmul); this stays as the ground truth the packed path is
        compared against. The ue8m0 scale bytes go up raw and are expanded by
        a 256-entry LUT on the device, so there is no host-side decode.
        """
        n = rows * pair_cols * 2
        kernel, lut, slut = self._kernel_bf16()
        mp = mx.array(np.frombuffer(wdata, dtype=np.uint8))
        ms = mx.array(np.frombuffer(sdata, dtype=np.uint8))
        outs = kernel(
            inputs=[mp, ms, lut, slut],
            output_shapes=[(n,)],
            output_dtypes=[mx.bfloat16],
            grid=(n, 1, 1),
            threadgroup=(min(n, 256), 1, 1),
        )
        return outs[0].reshape(rows, pair_cols * 2)

    def _pack_expert(self, wdata: bytes, sdata: bytes,
                     rows: int, pair_cols: int) -> tuple[mx.array, mx.array]:
        """fp4 bytes -> (uint32 weights, uint8 scales) for quantized_matmul."""
        w32, scales = pack_fp4_mxfp4(wdata, sdata, rows, pair_cols)
        return mx.array(w32), mx.array(scales)

    def _allocate_direct_runs(self, layer: int, expert: int,
                              prefix: str) -> list[tuple[dict, mx.array]]:
        runs = self._expert_runs(prefix, layer, expert)
        plan: list[tuple[dict, mx.array]] = []
        for run in runs:
            nbytes = run["end"] - run["start"]
            kinds = {kind for _, kind, _, _ in run["parts"]}
            if kinds == {"weight"}:
                if nbytes % 4:
                    raise ValueError("packed weight run is not uint32 aligned")
                buf = mx.empty((nbytes // 4,), dtype=mx.uint32)
            elif kinds == {"scale"}:
                buf = mx.empty((nbytes,), dtype=mx.uint8)
            else:
                raise ValueError("mixed weight/scale run is unsupported")
            plan.append((run, buf))
        return plan

    @staticmethod
    def _direct_views(plan: list[tuple[dict, mx.array]]
                      ) -> list[tuple[dict, mx.array, memoryview]]:
        return [(run, buf, memoryview(np.array(buf, copy=False)).cast("B"))
                for run, buf in plan]

    def _fill_direct_runs(self,
                          plan: list[tuple[dict, mx.array, memoryview]]) -> None:
        for run, _buf, target in plan:
            read_into(self._root, run["shard"], run["start"], target)

    def _direct_plan_to_expert(
            self, plan: list[tuple[dict, mx.array, memoryview]]
            ) -> dict[str, tuple[mx.array, mx.array]]:

        out: dict[str, dict[str, mx.array]] = {}
        for run, buf, _host_view in plan:
            for role, kind, offset, entry in run["parts"]:
                if kind == "weight":
                    begin = offset // 4
                    count = entry["bytes"] // 4
                    shape = (int(entry["shape"][0]),
                             int(entry["shape"][1]) // 4)
                else:
                    begin = offset
                    count = entry["bytes"]
                    shape = tuple(entry["shape"])
                arr = buf[begin:begin + count].reshape(shape)
                out.setdefault(role, {})[kind] = arr
        return {role: (out[role]["weight"], out[role]["scale"])
                for role in ("w1", "w2", "w3")}

    def _load_expert_direct(self, layer: int, expert: int,
                            prefix: str) -> dict[str, tuple[mx.array, mx.array]]:
        """pread merged expert runs into MLX-owned shared buffers (opt-in)."""
        plan = self._allocate_direct_runs(layer, expert, prefix)
        mx.eval([buf for _, buf in plan])
        plan = self._direct_views(plan)
        self._fill_direct_runs(plan)
        return self._direct_plan_to_expert(plan)

    def load_expert(self, layer: int, expert: int, prefix: str = "layers") -> dict[str, tuple[mx.array, mx.array]]:
        """Packed mxfp4 {w1, w2, w3} operand pairs (uint32 + ue8m0 scales).

        prefix='layers' for backbone experts; prefix='mtp' uses mtp.{layer}.ffn.
        Two merged preads (weight run + scale run) replace six per-tensor reads.
        """
        t0 = time.perf_counter()
        if self.direct_pread:
            out = self._load_expert_direct(layer, expert, prefix)
            mx.eval([t for pair in out.values() for t in pair])
            self.io_s += time.perf_counter() - t0
            self.bytes_read += sum(run["end"] - run["start"]
                                   for run in self._expert_runs(prefix, layer, expert))
            self.load_calls += 1
            return out
        parts = self._read_expert_parts(layer, expert, prefix, parallel=True)
        out: dict[str, tuple[mx.array, mx.array]] = {}
        for role, wdata, sdata, (rows, pair_cols) in parts:
            out[role] = self._pack_expert(wdata, sdata, rows, pair_cols)
        mx.eval([t for pair in out.values() for t in pair])
        self.io_s += time.perf_counter() - t0
        self.bytes_read += sum(len(w) + len(s) for _, w, s, _ in parts)
        self.load_calls += 1
        return out

    def load_experts_parallel(self, specs: list[tuple[int, int, str]], workers: int = 8) -> dict[tuple[str, int, int], dict[str, tuple[mx.array, mx.array]]]:
        """Load many experts with read+pack overlapped inside each worker.

        specs: list of (layer, expert, prefix). Returns key=(prefix, layer, expert) -> dict.
        Packing now runs on the IO threads so memcpy hides behind pread;
        the trailing eval uploads every operand in one shot.
        """
        if not specs:
            return {}

        direct_plans: dict[tuple[str, int, int],
                           list[tuple[dict, mx.array, memoryview]]] = {}
        if self.direct_pread:
            allocated = {
                (prefix, layer, expert): self._allocate_direct_runs(
                    layer, expert, prefix)
                for layer, expert, prefix in specs
            }
            mx.eval([buf for plan in allocated.values() for _, buf in plan])
            direct_plans = {key: self._direct_views(plan)
                            for key, plan in allocated.items()}

        def _read(spec: tuple[int, int, str]):
            layer, expert, prefix = spec
            if self.direct_pread:
                key = (prefix, layer, expert)
                plan = direct_plans[key]
                self._fill_direct_runs(plan)
                nbytes = sum(run["end"] - run["start"]
                             for run in self._expert_runs(prefix, layer, expert))
                return key, plan, 0.0, nbytes
            parts = self._read_expert_parts(layer, expert, prefix)
            tp0 = time.perf_counter()
            packed = [
                (role, self._pack_expert(wdata, sdata, rows, pair_cols))
                for role, wdata, sdata, (rows, pair_cols) in parts
            ]
            pack_s = time.perf_counter() - tp0
            nbytes = sum(len(w) + len(s) for _, w, s, _ in parts)
            return (prefix, layer, expert), packed, pack_s, nbytes

        t0 = time.perf_counter()
        reads = list(self._io_pool.map(_read, specs))
        t1 = time.perf_counter()
        out: dict[tuple[str, int, int], dict[str, tuple[mx.array, mx.array]]] = {}
        pending: list[mx.array] = []
        pack_wall = 0.0
        for key, packed, pack_s, _ in reads:
            if self.direct_pread:
                packed = list(self._direct_plan_to_expert(packed).items())
            out[key] = dict(packed)
            for _, pair in packed:
                pending.extend(pair)
            pack_wall += pack_s
        if pending:
            mx.eval(pending)
        t3 = time.perf_counter()
        self.eval_times.append(t3 - t1)
        self.batch_sizes.append(len(reads))
        self.io_s += t1 - t0
        self.dequant_s += pack_wall
        self.eval_s += max(0.0, t3 - t1 - min(pack_wall, t1 - t0))
        self.bytes_read += sum(nbytes for _, _, _, nbytes in reads)
        self.load_calls += 1
        return out

    def load_expert_banks(self, specs: list[tuple[int, int, str]], workers: int = 8
                          ) -> dict[int, dict]:
        """Load pinned experts into per-layer contiguous MXFP4 banks.

        Unlike stacking cached per-expert MLX arrays later, this stacks the
        packed NumPy buffers before upload. It keeps one device allocation per
        projection bank and avoids a second resident copy of the pinned set.
        Specs must be backbone experts; each layer is uploaded separately to
        bound host-side staging memory.
        """
        if not specs:
            return {}
        grouped: dict[int, list[tuple[int, int, str]]] = {}
        for layer, expert, prefix in specs:
            if prefix != "layers":
                raise ValueError("expert banks currently support backbone layers only")
            grouped.setdefault(int(layer), []).append((int(layer), int(expert), prefix))

        def _read(spec: tuple[int, int, str]):
            layer, expert, prefix = spec
            parts = self._read_expert_parts(layer, expert, prefix)
            t0 = time.perf_counter()
            packed = {
                role: pack_fp4_mxfp4(wdata, sdata, shape[0], shape[1])
                for role, wdata, sdata, shape in parts
            }
            pack_s = time.perf_counter() - t0
            nbytes = sum(len(w) + len(s) for _, w, s, _ in parts)
            return expert, packed, pack_s, nbytes

        out: dict[int, dict] = {}
        pack_wall = 0.0
        read_wall = 0.0
        upload_wall = 0.0
        total_bytes = 0
        for layer in sorted(grouped):
            layer_specs = grouped[layer]
            t_read = time.perf_counter()
            reads = list(self._io_pool.map(_read, layer_specs))
            read_wall += time.perf_counter() - t_read
            total_bytes += sum(nbytes for _, _, _, nbytes in reads)
            pack_wall += sum(pack_s for _, _, pack_s, _ in reads)
            experts = [expert for expert, _, _, _ in reads]
            banks: dict[str, tuple[mx.array, mx.array]] = {}
            pending: list[mx.array] = []
            for role in ("w1", "w2", "w3"):
                ws = np.stack([packed[role][0] for _, packed, _, _ in reads])
                ss = np.stack([packed[role][1] for _, packed, _, _ in reads])
                mw, ms = mx.array(ws), mx.array(ss)
                banks[role] = (mw, ms)
                pending.extend((mw, ms))
            t_upload = time.perf_counter()
            mx.eval(pending)
            upload_wall += time.perf_counter() - t_upload
            out[layer] = {
                "banks": banks,
                "slots": {expert: slot for slot, expert in enumerate(experts)},
            }
            # Drop this layer's CPU staging before assembling the next bank.
            del reads, pending

        self.io_s += read_wall
        self.dequant_s += pack_wall
        self.eval_s += upload_wall
        self.bytes_read += total_bytes
        self.load_calls += 1
        self.batch_sizes.append(len(specs))
        return out


class DenseLoader:
    """One-shot dequant of resident weights to bf16 (skips streamed tables)."""

    SKIP_SUBSYSTEMS = ("routed-expert", "mtp-expert", "vision")

    def __init__(self, model_root: Path | str, manifest: dict) -> None:
        self._root = Path(model_root)
        self._manifest = manifest

    def load_all(self) -> dict[str, mx.array]:
        out: dict[str, mx.array] = {}
        # F8 -> F32 -> BF16 dequant graphs for every dense tensor otherwise
        # remain queued until the final eval, temporarily retaining both the
        # float32 inputs and BF16 outputs. An opt-in eval batch bounds that
        # startup peak without changing any tensor values.
        eval_chunk = max(0, int(os.environ.get(
            "DEEPSEEKER_DENSE_EVAL_CHUNK", "0")))
        pending: list[mx.array] = []
        for tensor in self._manifest.get("tensors", []):
            if tensor.get("subsystem") in self.SKIP_SUBSYSTEMS:
                continue
            if ".engram.embed." in tensor.get("name", ""):
                continue  # streamed rows via EngramRowCache, never resident
            if tensor["name"].endswith(".scale"):
                continue  # consumed with their fp8/fp4 weight, never standalone
            value = self.load_one(tensor["name"])
            out[tensor["name"]] = value
            if eval_chunk:
                pending.append(value)
                if len(pending) >= eval_chunk:
                    mx.eval(pending)
                    pending.clear()
        if eval_chunk:
            if pending:
                mx.eval(pending)
        else:
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
