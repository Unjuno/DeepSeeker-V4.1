"""Tests for Issue #8 trace schema and offline tooling (no weights)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.trace import (
    arch_config,
    emit_synthetic,
    layer_histogram,
    make_header,
    read_trace,
    trace_stats,
    validate_record,
    write_trace,
)

TINY_CFG = {"n_layers": 2, "n_routed_experts": 16, "top_k": 2}


def tiny_header(source="synthetic"):
    return make_header("repo", "rev", source, TINY_CFG, seed=1)


def test_arch_config_from_manifest():
    from deepseeker.baseline import load_json

    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    cfg = arch_config(manifest)
    assert cfg == {"n_layers": 40, "n_routed_experts": 384, "top_k": 6}


def test_header_rejects_bad_geometry():
    with pytest.raises(ValueError):
        make_header("repo", "rev", "synthetic", {**TINY_CFG, "n_layers": 0})


def test_validate_accepts_good_record():
    record = {"request_id": "r", "token_pos": 0, "layer": 1, "experts": [3, 7]}
    assert validate_record(record, tiny_header()) == []


def test_validate_rejects_bad_records():
    header = tiny_header()
    bad_layer = {"request_id": "r", "token_pos": 0, "layer": 9, "experts": [1, 2]}
    assert validate_record(bad_layer, header)
    bad_count = {"request_id": "r", "token_pos": 0, "layer": 0, "experts": [1]}
    assert validate_record(bad_count, header)
    dupes = {"request_id": "r", "token_pos": 0, "layer": 0, "experts": [5, 5]}
    assert validate_record(dupes, header)
    bad_id = {"request_id": "r", "token_pos": 0, "layer": 0, "experts": [1, 99]}
    assert validate_record(bad_id, header)
    bad_scores = {
        "request_id": "r",
        "token_pos": 0,
        "layer": 0,
        "experts": [1, 2],
        "scores": [0.5],
    }
    assert validate_record(bad_scores, header)


def test_event_records_validate():
    header = tiny_header()
    good_engram = {"kind": "event", "request_id": "r", "token_pos": 3, "layer": 1,
                   "extra": {"kind": "engram", "n_hashes": 5}}
    good_kv = {"kind": "event", "request_id": "r", "token_pos": 3, "layer": 1,
               "extra": {"kind": "kv", "compress_rows": 2, "topk_count": 64}}
    assert validate_record(good_engram, header) == []
    assert validate_record(good_kv, header) == []
    bad = dict(good_engram, token_pos=-1)
    assert validate_record(bad, header)
    bad2 = dict(good_kv, extra={"kind": "bogus"})
    assert validate_record(bad2, header)


def test_roundtrip_with_events(tmp_path):
    header = tiny_header()
    records = emit_synthetic(header, n_requests=2, tokens_per_request=3, seed=1)
    records.append({"kind": "event", "request_id": "synthetic-1-0", "token_pos": 0,
                    "layer": 1, "extra": {"kind": "kv", "compress_rows": 1, "topk_count": 4}})
    assert len(records) == 2 * 3 * 2 + 1
    path = tmp_path / "trace.jsonl"
    assert write_trace(path, header, records) == len(records)
    back_header, back = read_trace(path)
    assert back_header["schema"] == "deepseeker.trace/v1"
    assert back == records


def test_read_rejects_corrupt(tmp_path):
    header = tiny_header()
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "nope"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_trace(path)
    path.write_text(json.dumps(header) + "\n" + '{"kind": "record"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_trace(path)


def test_synthetic_deterministic_and_shaped():
    header = tiny_header()
    first = emit_synthetic(header, 2, 4, seed=7)
    second = emit_synthetic(header, 2, 4, seed=7)
    assert first == second
    assert all(len(r["experts"]) == 2 for r in first)
    assert all(0 <= r["layer"] < 2 for r in first)


def test_stats_and_histogram():
    header = tiny_header()
    records = emit_synthetic(header, 3, 5, seed=2)
    stats = trace_stats(header, records)
    assert stats["records"] == 3 * 5 * 2
    assert stats["requests"] == 3
    assert 0 < stats["mean_unique"] <= 16
    assert stats["adjacent_same_set_ratio"] is not None
    hist = layer_histogram(records, 2)
    assert sum(sum(b.values()) for b in hist.values()) == 3 * 5 * 2 * 2
