"""Tests for the canonical tensor-manifest builder.

Hermetic: synthetic safetensors headers are built in-memory; only the
classification tests use real tensor names from the pinned upstream
metadata (names only, no weights).
"""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_model_manifest.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_model_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_builder()


def synth_header_bytes(entries: dict) -> bytes:
    raw = json.dumps(entries).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def test_parse_shard_tensors_offsets_and_metadata_skip():
    header = {
        "__metadata__": {"format": "pt"},
        "a.weight": {
            "dtype": "F16",
            "shape": [4, 8],
            "data_offsets": [0, 64],
        },
        "b.scale": {
            "dtype": "U8",
            "shape": [4],
            "data_offsets": [64, 68],
        },
    }
    parsed = builder.parse_shard_tensors(header)
    assert set(parsed) == {"a.weight", "b.scale"}
    assert parsed["a.weight"]["data_range"] == [0, 64]
    assert parsed["b.scale"]["shape"] == [4]


def test_file_range_convention():
    # header_len known; file_range must be 8 + header_len + relative.
    header_len = 100
    start, end = 16, 80
    assert 8 + header_len + start == 124
    assert 8 + header_len + end == 188


def test_build_manifest_accounting_and_ranges():
    entries = {
        "layers.0.ffn.experts.0.w1.weight": {
            "dtype": "U8",
            "shape": [2304, 5120],
            "data_offsets": [0, 100],
        },
        "layers.0.ffn.experts.0.w1.scale": {
            "dtype": "U8",
            "shape": [8],
            "data_offsets": [100, 108],
        },
        "mystery.blob": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [108, 124],
        },
    }
    raw = json.dumps(entries).encode("utf-8")
    header_len = len(raw)
    header = json.loads(raw.decode("utf-8"))
    weight_map = {k: "shard.safetensors" for k in entries}
    manifest = builder.build_manifest(
        "rev",
        weight_map,
        {"shard.safetensors": (header, header_len, 100000)},
        {"source": "test"},
        123,
        45,
    )
    assert manifest["totals"]["tensors"] == 3
    assert manifest["totals"]["tensor_bytes"] == 124
    by_sub = manifest["totals"]["by_subsystem"]
    assert sum(b["bytes"] for b in by_sub.values()) == manifest["totals"]["tensor_bytes"]
    assert manifest["validation"]["accounting_reconciled"] is True
    assert manifest["validation"]["missing_from_headers"] == []
    assert by_sub["unknown"]["tensors"] == 1
    first = manifest["tensors"][0]
    assert first["file_range"][0] == 8 + header_len + first["data_range"][0]
    assert manifest["per_expert_bytes"] == {"0/0": 108}
    assert manifest["transport"]["network_bytes"] == 123


def test_build_manifest_twice_identical_modulo_timestamp():
    entries = {
        "embed.weight": {
            "dtype": "BF16",
            "shape": [4],
            "data_offsets": [0, 8],
        },
    }
    raw = json.dumps(entries).encode("utf-8")
    header = json.loads(raw.decode("utf-8"))
    kwargs = {
        "weight_map": {"embed.weight": "s"},
        "shard_headers": {"s": (header, len(raw), 1000)},
        "architecture": {},
        "network_bytes": 1,
        "index_bytes": 2,
    }
    first = builder.build_manifest("rev", **kwargs)
    second = builder.build_manifest("rev", **kwargs)
    first.pop("generated_at_utc")
    second.pop("generated_at_utc")
    assert first == second


# --------------------------------------------------------------------------
# Classification with real upstream tensor names


def test_classify_backbone_routed_expert():
    cls = builder.classify("layers.12.ffn.experts.305.w3.scale")
    assert cls["subsystem"] == "routed-expert"
    assert cls["layer"] == 12
    assert cls["expert"] == 305
    assert cls["routed"] is True


def test_classify_shared_expert_and_router():
    cls = builder.classify("layers.0.ffn.shared_experts.w2.weight")
    assert cls["subsystem"] == "shared-expert"
    assert cls["routed"] is False
    cls = builder.classify("layers.3.ffn.gate.weight")
    assert cls["subsystem"] == "router"
    assert cls["layer"] == 3


def test_classify_attention_and_kv_index():
    cls = builder.classify("layers.0.attn.wkv.weight")
    assert cls["subsystem"] == "attention"
    cls = builder.classify("layers.8.attn.indexer.wk.weight")
    assert cls["subsystem"] == "kv-index"
    cls = builder.classify("layers.8.attn.compressor.wkv.weight")
    assert cls["subsystem"] == "kv-index"


def test_classify_norms_embedding_head():
    assert builder.classify("layers.5.ffn_norm.weight")["subsystem"] == "norm"
    assert builder.classify("norm.weight")["detail"] == "final-norm"
    assert builder.classify("embed.weight")["subsystem"] == "embedding"
    assert builder.classify("head.weight")["subsystem"] == "lm-head"
    assert builder.classify("image_start")["subsystem"] == "embedding"


def test_classify_engram_mtp_vision():
    cls = builder.classify("layers.14.engram.embed.weight")
    assert cls["subsystem"] == "engram"
    assert cls["layer"] == 14
    cls = builder.classify("mtp.2.ffn.experts.100.w1.weight")
    assert cls["subsystem"] == "mtp-expert"
    assert cls["routed"] is True
    assert builder.classify("mtp.1.ffn.gate.weight")["subsystem"] == ("mtp-router")
    assert builder.classify("mtp.2.markov_head.embed.weight")["subsystem"] == "mtp"
    assert builder.classify("vision.blocks.3.mlp.w1.weight")["subsystem"] == "vision"
    assert builder.classify("aligner.w1.weight")["subsystem"] == "vision"


def test_classify_unknown_fallback_never_drops():
    for name in (
        "layers.0.hc_attn_base",
        "mtp.1.hc_ffn_scale",
        "totally_new_tensor",
    ):
        cls = builder.classify(name)
        assert cls["subsystem"] == "unknown", name
        assert cls["detail"] == name
    # confidence_head lives under an MTP layer: mtp, not unknown.
    assert builder.classify("mtp.2.confidence_head.proj.weight")["subsystem"] == "mtp"
