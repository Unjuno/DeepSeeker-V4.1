"""Tests for Issue #33 Metal e2m1 dequant (exact vs references)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.metal_fp4 import (
    FP4_TABLE,
    decode_scales_u8m0,
    metal_dequant,
    reference_dequant,
    torch_reference,
)

MODEL_ROOT = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"


def test_table_spot_values():
    assert FP4_TABLE[0x7] == 6.0
    assert FP4_TABLE[0xF] == -6.0
    assert FP4_TABLE[0x8] == 0.0
    assert decode_scales_u8m0(bytes([127]))[0] == 1.0
    assert decode_scales_u8m0(bytes([128]))[0] == 2.0
    assert np.isnan(decode_scales_u8m0(bytes([0xFF]))[0])


def test_metal_matches_references_random():
    rng = np.random.default_rng(3)
    packed = rng.integers(0, 256, size=512, dtype=np.uint8)  # 1024 elems
    scales = np.exp2(rng.integers(-3, 3, size=32).astype(np.float32))
    ref = reference_dequant(packed, scales)
    got = metal_dequant(packed, scales)
    assert got.shape == (1024,)
    np.testing.assert_array_equal(got, ref)
    torch_vals = torch_reference(packed, scales).numpy()
    np.testing.assert_array_equal(got, torch_vals)


def test_metal_on_real_expert_bytes():
    import os

    revision = load_json(
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    w = next(t for t in manifest["tensors"] if t["name"] == "layers.5.ffn.experts.0.w1.weight")
    s = next(t for t in manifest["tensors"] if t["name"] == "layers.5.ffn.experts.0.w1.scale")
    assert w["bytes"] % 256 == 0
    fd = os.open(MODEL_ROOT / w["shard"], os.O_RDONLY)
    try:
        raw = os.pread(fd, 8192, w["file_range"][0])
    finally:
        os.close(fd)
    fds = os.open(MODEL_ROOT / s["shard"], os.O_RDONLY)
    try:
        sraw = os.pread(fds, 512, s["file_range"][0])
    finally:
        os.close(fds)
    scales = decode_scales_u8m0(sraw)
    assert scales.size == 512  # 8192 bytes = 16384 nibbles / 32 per block
    packed = np.frombuffer(raw, dtype=np.uint8)
    got = metal_dequant(packed, scales)
    ref = reference_dequant(packed, scales)
    np.testing.assert_array_equal(got, ref)
    finite = np.isfinite(got)
    assert finite.mean() > 0.99
    assert np.abs(got[finite]).max() < 448.0  # e4m3fn range per convert.py comment


def test_bad_scales_rejected():
    packed = np.zeros(128, dtype=np.uint8)
    with pytest.raises(ValueError):
        metal_dequant(packed, np.ones(3, dtype=np.float32))
