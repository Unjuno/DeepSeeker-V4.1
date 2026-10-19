"""Tests for runner weight loading (exactness on real bytes)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.mlx_weights import (
    E4M3_TABLE,
    DenseLoader,
    ExpertLoader,
    decode_u8m0,
    dequant_fp4_rowmajor,
    dequant_fp8_block,
)

MODEL_ROOT = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
REVISION = load_json(
    REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
)["resolved_revision"]
MANIFEST = load_json(
    REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / REVISION / "model-manifest.json"
)


def test_e4m3_table_spots():
    import torch

    assert E4M3_TABLE[0x00] == 0.0
    assert E4M3_TABLE[0x40] == pytest.approx(2.0)  # 0 1000 000, bias 7
    assert E4M3_TABLE[0x3C] == pytest.approx(1.5)  # 0 0111 100
    assert E4M3_TABLE[0xC0] == pytest.approx(-2.0)
    assert E4M3_TABLE[0x78] == pytest.approx(256.0)  # extended range, not NaN
    assert E4M3_TABLE[0x7E] == pytest.approx(448.0)
    assert np.isnan(E4M3_TABLE[0x7F])
    assert np.isnan(E4M3_TABLE[0xFF])
    # full table == torch ground truth (also enforced at import)
    want = torch.frombuffer(bytearray(range(256)), dtype=torch.float8_e4m3fn).float().numpy()
    same = (np.isnan(want) & np.isnan(E4M3_TABLE)) | (want == E4M3_TABLE)
    assert same.all()


def test_fp8_block_against_torch():
    import torch

    rng = np.random.default_rng(1)
    packed = rng.integers(0, 256, size=(32, 64), dtype=np.uint8)
    scales = rng.integers(120, 135, size=(1, 2), dtype=np.uint8)
    got = dequant_fp8_block(packed, scales)
    table = torch.tensor(E4M3_TABLE)
    vals = table[torch.from_numpy(packed.astype(np.int64))]
    gain = torch.from_numpy(decode_u8m0(scales)).repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    np.testing.assert_allclose(got, (vals * gain).numpy(), rtol=1e-6)


def test_fp4_rowmajor_matches_metal_kernel():
    from deepseeker.metal_fp4 import reference_dequant
    from deepseeker.mlx_weights import decode_u8m0

    rng = np.random.default_rng(2)
    packed = rng.integers(0, 256, size=512, dtype=np.uint8)
    raw_scales = rng.integers(120, 135, size=32, dtype=np.uint8)
    np.testing.assert_array_equal(
        dequant_fp4_rowmajor(packed, raw_scales),
        reference_dequant(packed, decode_u8m0(raw_scales)),
    )


def test_expert_loader_real_shapes():
    loader = ExpertLoader(MODEL_ROOT, MANIFEST)
    expert = loader.load_expert(5, 7)
    assert set(expert) == {"w1", "w2", "w3"}
    w, scales = expert["w1"]
    assert w.shape == (2304, 640) and w.dtype == mx.uint32   # 5120 / 8
    assert scales.shape == (2304, 160) and scales.dtype == mx.uint8  # / 32
    assert expert["w2"][0].shape == (5120, 288)             # 2304 / 8
    mx.eval(w, scales)
    assert bool((w != 0).any().item())


def test_packed_qmm_matches_staged_bf16_gemm():
    """quantized_matmul on packed bytes == dequant-then-bf16-GEMM."""
    from deepseeker.mlx_forward import qmm

    loader = ExpertLoader(MODEL_ROOT, MANIFEST)
    _, wdata, sdata, (rows, pair_cols) = loader._read_role(5, 7, "layers", "w1")
    packed = loader._pack_expert(wdata, sdata, rows, pair_cols)
    staged = loader._dequant_metal(wdata, sdata, rows, pair_cols)
    mx.eval(*packed, staged)

    rng = np.random.default_rng(4)
    x = mx.array(rng.standard_normal((3, pair_cols * 2)).astype(np.float32))
    ref = x @ staged.astype(mx.float32).T
    got = qmm(x, packed)
    mx.eval(ref, got)
    a, b = np.asarray(ref), np.asarray(got)
    assert np.max(np.abs(a - b)) <= 1e-5 * np.max(np.abs(a))


def test_dense_loader_spot():
    loader = DenseLoader(MODEL_ROOT, MANIFEST)
    w = loader.load_one("layers.0.attn.wq_a.weight")
    assert tuple(w.shape) == (1280, 5120)
    assert w.dtype == mx.bfloat16
    norm = loader.load_one("layers.0.attn.q_norm.weight")
    assert norm.dtype == mx.bfloat16
