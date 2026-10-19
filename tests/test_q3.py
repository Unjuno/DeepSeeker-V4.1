"""Tests for the Q3 pilot plumbing: requant writer, variant paths, decode.

Hermetic: synthetic tensors and dirs; the artifact test skips if the
tracked Q3 manifest is not present.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlx.core as mx

from deepseeker import mlx_weights as W
from deepseeker.variant_q3 import is_q3, paths_for_variant


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


requant = _load_script(REPO_ROOT / "scripts" / "requant_experts_3bit.py", "requant_experts_3bit")


def test_parse_layers_ranges_and_lists():
    assert requant.parse_layers("20-23") == [20, 21, 22, 23]
    assert requant.parse_layers("2,5,7") == [2, 5, 7]
    assert requant.parse_layers("0,6-8") == [0, 6, 7, 8]


def test_parse_experts_defaults_to_all():
    assert requant.parse_experts(None) == list(range(384))
    assert requant.parse_experts("3-5") == [3, 4, 5]


def test_is_q3_variant_names():
    assert is_q3("1017-Q3")
    assert is_q3("q3")
    assert is_q3("1017-q3")
    assert not is_q3("1017")
    assert not is_q3("")
    assert not is_q3(None)


def test_paths_for_variant_manifest_selection(tmp_path):
    revision = "abc123"
    profile = tmp_path / "profiles" / "deepseek-v4.1-flash" / revision
    profile.mkdir(parents=True)
    (profile / "model-manifest.json").write_text("{}")
    (profile / "model-manifest-q3.json").write_text("{}")
    (tmp_path / "models" / "DeepSeek-V4.1-Flash-Q3").mkdir(parents=True)

    root, manifest = paths_for_variant(revision, tmp_path, "DeepSeek-V4.1-Flash")
    assert manifest == profile / "model-manifest.json"
    assert root == tmp_path / "models" / "DeepSeek-V4.1-Flash"

    old = os.environ.get("DEEPSEEKER_Q3_MANIFEST")
    try:
        os.environ.pop("DEEPSEEKER_Q3_MANIFEST", None)
        root, manifest = paths_for_variant(
            revision, tmp_path, "DeepSeek-V4.1-Flash", variant="1017-Q3"
        )
        assert manifest == profile / "model-manifest-q3.json"
        assert root == tmp_path / "models" / "DeepSeek-V4.1-Flash-Q3"

        os.environ["DEEPSEEKER_Q3_MANIFEST"] = "model-manifest-qprobe.json"
        _root, manifest = paths_for_variant(revision, tmp_path, "DeepSeek-V4.1-Flash", variant="q3")
        assert manifest == profile / "model-manifest-qprobe.json"
    finally:
        if old is not None:
            os.environ["DEEPSEEKER_Q3_MANIFEST"] = old
        else:
            os.environ.pop("DEEPSEEKER_Q3_MANIFEST", None)


def test_convert_tensor_roundtrip_and_geometry():
    rng = np.random.default_rng(7)
    vals = (rng.standard_normal(8 * 256) * 0.05).astype(np.float32)
    words, sraw, braw, rms = requant.convert_tensor(vals, 8, 256, bits=3)
    # 256 cols * 3 bits = 768 bits -> 24 words/row; 4 groups per row.
    assert len(words) == 8 * 24 * 4  # uint32 bytes
    assert len(sraw) == 8 * 4 * 2  # fp16 bytes
    assert len(braw) == 8 * 4 * 2
    assert 0.0 < rms < 0.08  # affine 3-bit gs=64 on O(0.05) scale values
    # Decode through the same path the runner uses and recompute the error.
    packed = mx.array(np.frombuffer(words, dtype="<u4").reshape(8, 24))
    scales = mx.array(np.frombuffer(sraw, dtype="<f2").reshape(8, 4))
    biases = mx.array(np.frombuffer(braw, dtype="<f2").reshape(8, 4))
    rt = mx.dequantize(packed, scales=scales, biases=biases, group_size=64, bits=3)
    err = np.asarray(mx.array(vals.reshape(8, 256)), np.float64) - np.asarray(rt, np.float64)
    rms2 = float(np.sqrt(np.mean(err**2)) / max(float(np.abs(mx.array(vals)).max()), 1e-9))
    assert abs(rms - rms2) < 1e-6


def test_unpack_int3_affine_rejects_shape_mismatch():
    words = np.zeros((4, 24), dtype="<u4").tobytes()
    scales = np.zeros((4, 4), dtype="<f2").tobytes()
    biases = np.zeros((4, 5), dtype="<f2").tobytes()
    with pytest.raises(ValueError):
        W.unpack_int3_affine(words, scales, biases, (4, 24), (4, 4))
    biases = np.zeros((4, 4), dtype="<f2").tobytes()
    scales_bad = np.zeros((3, 4), dtype="<f2").tobytes()
    with pytest.raises(ValueError):
        W.unpack_int3_affine(words, scales_bad, biases, (4, 24), (4, 4))


def test_unpack_int3_affine_passes_shape_check():
    rows, cols, gs, bits = 2, 128, 64, 3
    words_row = (cols * bits + 31) // 32
    groups = cols // gs
    words = np.zeros((rows, words_row), dtype="<u4").tobytes()
    scales = np.full((rows, groups), 120, dtype="<f2").tobytes()
    biases = np.zeros((rows, groups), dtype="<f2").tobytes()
    w, s, b = W.unpack_int3_affine(words, scales, biases, (rows, words_row), (rows, groups))
    assert w.shape == (rows, words_row)
    assert s.shape == (rows, groups) and b.shape == (rows, groups)


def test_dequant_fp4_rowmajor_interleaving_and_scale():
    # Two bytes -> four nibble slots, low nibble first.
    # fp4 table index 3 == 1.0 (E2M1), index 1 == 0.5.
    raw = np.array([0x13], dtype=np.uint8)  # low=3, high=1
    gain = W.decode_u8m0(np.array([128], dtype=np.uint8))  # 2.0
    vals = W.dequant_fp4_rowmajor(raw, np.array([128], dtype=np.uint8))
    expect = np.array([W.FP4_TABLE[3], W.FP4_TABLE[1]], dtype=np.float32)
    assert float(gain[0]) == 2.0
    np.testing.assert_allclose(vals, expect * 2.0, rtol=0, atol=0)


MANIFEST_Q3 = (
    REPO_ROOT
    / "profiles"
    / "deepseek-v4.1-flash"
    / "1bae0e1088993b9b82de08f59de5be5a34e42170"
    / "model-manifest-q3.json"
)


@pytest.mark.skipif(not MANIFEST_Q3.exists(), reason="Q3 manifest not tracked")
def test_manifest_q3_artifact_consistency():
    m = json.loads(MANIFEST_Q3.read_text())
    q = m["q3_requant"]
    assert q["bits"] == 3 and q["group_size"] == 64
    assert q["layer_range"] == [20, 39]
    assert q["experts_converted"] == len(range(20, 40)) * 384
    expert_bytes = q["q3_expert_bytes"]
    assert 15_000_000 < expert_bytes < 16_500_000  # vs 18.8 MB FP4 (-17.6%)
    assert q["payload_bytes_written"] > 100_000_000_000
    by = {t["name"]: t for t in m["tensors"]}
    for layer in (20, 25, 39):
        base = f"layers.{layer}.ffn.experts.0"
        assert by[f"{base}.w1.weight"]["dtype"] == "U32_Q3_AFFINE"
        assert by[f"{base}.w2.weight"]["dtype"] == "U32_Q3_AFFINE"
        assert by[f"{base}.w1.bias"]["dtype"] == "F16"
        assert by[f"{base}.w2.bias"]["shard"] == q["payload"]
        assert by[f"{base}.w3.bias"]["shard"] == q["payload"]
    # payload bytes actually referenced by manifest entries match the writer log
    assert q["payload_bytes_written"] == sum(
        t["bytes"] for t in m["tensors"] if t.get("shard") == q["payload"]
    )
