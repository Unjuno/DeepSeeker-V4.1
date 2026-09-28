"""Issue #33: e2m1 FP4 dequant Metal kernel + staged GEMM path (roadmap P7).

True on-disk layout from `inference/convert.py`: two e2m1fn nibbles per
byte (low nibble first) mapped through
  FP4_TABLE = [0,.5,1,1.5,2,3,4,6, 0,-.5,-1,-1.5,-2,-3,-4,-6]
with one ue8m0 power-of-two scale per 32 elements. Host decodes scales
to float32 (2^(bits-127); 0xFF -> NaN); the device kernel does LUT +
multiply only, keeping it exactly testable against the torch reference
`reference_dequant` below.

Staged path: dequant once per expert load, then MLX bf16 GEMM (the
framework fallback IS the compute path until a fused kernel lands).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch

FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
BLOCK = 32

_DEQUANT_SOURCE = """
uint elem = thread_position_in_grid.x;
uint byte = packed[elem >> 1];
uint nib = (elem & 1) ? (byte >> 4) & 0xFu : byte & 0xFu;
uint s = elem >> 5;
out[elem] = float(lut[nib]) * scales[s];
"""

# Direct bf16 store: skips the f32 intermediate write+read+rewrite (~3x less
# output traffic). Metal's `bfloat(v)` matches MLX `.astype(mx.bfloat16)`
# round-to-nearest-even on all inputs incl. NaN (see test_metal_bf16_store).
_DEQUANT_BF16_SOURCE = """
uint elem = thread_position_in_grid.x;
uint byte = packed[elem >> 1];
uint nib = (elem & 1) ? (byte >> 4) & 0xFu : byte & 0xFu;
uint s = elem >> 5;
float v = float(lut[nib]) * scales[s];
out[elem] = bfloat(v);
"""

_kernel = None
_LUT = None
_kernel_bf16 = None
_kernel_bf16_u8 = None
_SLUT: mx.array | None = None
_SCALE_LUT: np.ndarray | None = None

# ue8m0 byte -> power of two, on device. Indexing this LUT on the GPU removes
# the host-side fancy-index (the dominant cost of a load call) and shrinks the
# scale upload 4x (raw bytes instead of f32).  0xFF -> NaN per OCP, matching
# decode_scales_u8m0 exactly.
_DEQUANT_BF16_U8_SOURCE = """
uint elem = thread_position_in_grid.x;
uint byte = packed[elem >> 1];
uint nib = (elem & 1) ? (byte >> 4) & 0xFu : byte & 0xFu;
uint s = elem >> 5;
float v = float(lut[nib]) * slut[scales[s]];
out[elem] = bfloat(v);
"""


def _kernel_fn():
    global _kernel, _LUT
    if _kernel is None:
        _LUT = mx.array(FP4_TABLE, dtype=mx.float32)
        _kernel = mx.fast.metal_kernel(
            name="e2m1_dequant",
            input_names=["packed", "scales", "lut"],
            output_names=["out"],
            source=_DEQUANT_SOURCE,
        )
    return _kernel, _LUT


def _kernel_bf16_fn():
    global _kernel_bf16, _LUT
    if _kernel_bf16 is None:
        if _LUT is None:
            _LUT = mx.array(FP4_TABLE, dtype=mx.float32)
        _kernel_bf16 = mx.fast.metal_kernel(
            name="e2m1_dequant_bf16",
            input_names=["packed", "scales", "lut"],
            output_names=["out"],
            source=_DEQUANT_BF16_SOURCE,
        )
    return _kernel_bf16, _LUT


def _scale_lut_np() -> np.ndarray:
    global _SCALE_LUT
    if _SCALE_LUT is None:
        bits = np.arange(256, dtype=np.int32)
        lut = np.exp2(bits - 127)
        lut[0xFF] = np.nan
        _SCALE_LUT = lut.astype(np.float32)
    return _SCALE_LUT


def _kernel_bf16_u8_fn():
    global _kernel_bf16_u8, _LUT, _SLUT
    if _kernel_bf16_u8 is None:
        if _LUT is None:
            _LUT = mx.array(FP4_TABLE, dtype=mx.float32)
        _kernel_bf16_u8 = mx.fast.metal_kernel(
            name="e2m1_dequant_bf16_u8",
            input_names=["packed", "scales", "lut", "slut"],
            output_names=["out"],
            source=_DEQUANT_BF16_U8_SOURCE,
        )
    if _SLUT is None:
        _SLUT = mx.array(_scale_lut_np(), dtype=mx.float32)
    return _kernel_bf16_u8, _LUT, _SLUT


def reference_dequant(packed: np.ndarray, scales_f32: np.ndarray) -> np.ndarray:
    """Torch/numpy reference: identical math to the Metal kernel."""
    raw = np.frombuffer(packed.tobytes(), dtype=np.uint8)
    low = (raw & 0x0F).astype(np.int64)
    high = ((raw >> 4) & 0x0F).astype(np.int64)
    vals = np.empty(raw.size * 2, dtype=np.float32)
    vals[0::2] = FP4_TABLE[low]
    vals[1::2] = FP4_TABLE[high]
    rep = np.repeat(scales_f32.astype(np.float32), BLOCK)
    return vals * rep


def decode_scales_u8m0(raw: bytes) -> np.ndarray:
    """ue8m0 bytes -> float32 powers of two (0xFF -> NaN per OCP)."""
    return _scale_lut_np()[np.frombuffer(raw, dtype=np.uint8)]


def metal_dequant(packed: np.ndarray, scales_f32: np.ndarray) -> np.ndarray:
    """Device e2m1 dequant; returns float32 host array, exact vs reference."""
    n = packed.size * 2
    if scales_f32.size != (n + BLOCK - 1) // BLOCK:
        raise ValueError("scales must cover ceil(n/32) blocks")
    kernel, lut = _kernel_fn()
    mp = mx.array(np.frombuffer(packed.tobytes(), dtype=np.uint8))
    ms = mx.array(np.ascontiguousarray(scales_f32, dtype=np.float32))
    outs = kernel(
        inputs=[mp, ms, lut],
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
    )
    mx.eval(outs)
    return np.asarray(outs[0])


def torch_reference(packed: np.ndarray, scales_f32: np.ndarray) -> torch.Tensor:
    """Independent torch reimplementation (cross-checks numpy + Metal)."""
    table = torch.tensor(FP4_TABLE)
    raw = torch.frombuffer(bytearray(packed.tobytes()), dtype=torch.uint8)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()
    vals = torch.empty(raw.numel() * 2)
    vals[0::2] = table[low]
    vals[1::2] = table[high]
    rep = torch.from_numpy(np.ascontiguousarray(scales_f32, dtype=np.float32)).repeat_interleave(BLOCK)
    return vals * rep
