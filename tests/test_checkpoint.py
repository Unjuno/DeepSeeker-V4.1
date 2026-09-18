"""Tests for checkpoint management (no weights, no network by default).

Resume/Range tests run against a local Range-capable HTTP server.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.checkpoint import (
    atomic_write_json,
    download_range,
    expected_files,
    git_ignored,
    git_safety_scan,
    new_state,
    sanitized_status,
    verify_shard_against_manifest,
)

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"


def _manifest():
    path = REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / REVISION / "model-manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_expected_files_from_manifest():
    manifest = _manifest()
    weight_map = {t["name"]: t["shard"] for t in manifest["tensors"]}
    files = expected_files(weight_map)
    shards = [f for f in files if f.endswith(".safetensors")]
    assert len(shards) == 48
    assert files[-4:] == [
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
    ]


def test_pinned_revision_enforced():
    spec = importlib.util.spec_from_file_location(
        "manage_checkpoint",
        REPO_ROOT / "scripts" / "manage_checkpoint.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.pinned_revision() == REVISION
    assert module.REPO_ID == "deepseek-ai/DeepSeek-V4.1-Flash"


def test_model_root_must_be_ignored():
    assert git_ignored(REPO_ROOT, REPO_ROOT / "models" / "x")
    assert git_ignored(REPO_ROOT, REPO_ROOT / ".cache" / "x")
    assert not git_ignored(REPO_ROOT, REPO_ROOT / "scripts" / "x.py")


def test_git_safety_clean_and_detects_payload(tmp_path):
    assert git_safety_scan(REPO_ROOT)["ok"] is True
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    fake = tmp_path / "weights.safetensors"
    fake.write_bytes(b"fake")
    subprocess.run(["git", "-C", str(tmp_path), "add", "weights.safetensors"], check=True)
    result = git_safety_scan(tmp_path)
    assert result["ok"] is False
    assert any("weights.safetensors" in i for i in result["issues"])


def _synth_shard(path: Path, tensors: dict) -> None:
    raw = json.dumps(tensors).encode()
    data = b"\x00" * sum(
        v["data_offsets"][1] - v["data_offsets"][0] for v in tensors.values() if isinstance(v, dict)
    )
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def test_verify_shard_ok_and_mismatch(tmp_path):
    manifest = {
        "tensors": [
            {
                "name": "a.weight",
                "shape": [4],
                "dtype": "F16",
                "bytes": 8,
                "shard": "s",
                "data_range": [0, 8],
                "file_range": [100, 108],
            },
        ]
    }
    header = {"a.weight": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]}}
    raw = json.dumps(header).encode()
    pad = 100 - 8 - len(raw)
    path = tmp_path / "s"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * pad + b"\x01" * 8)
    result = verify_shard_against_manifest(path, "s", manifest)
    assert result["ok"] is True
    assert result["overhead_bytes"] == pad
    manifest["tensors"][0]["shape"] = [8]
    result = verify_shard_against_manifest(path, "s", manifest)
    assert result["ok"] is False


class _RangeHandler(BaseHTTPRequestHandler):
    body = b""

    def do_GET(self):
        total = len(type(self).body)
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.split("=")[1].split("-")[0])
            payload = type(self).body[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
        else:
            payload = type(self).body
            self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_resume_download_range(tmp_path):
    _RangeHandler.body = bytes(range(256)) * 4096  # 1 MiB
    server = HTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/f"
    stop = threading.Event()
    part = tmp_path / "f.part"
    part.write_bytes(_RangeHandler.body[:400000])
    hasher = hashlib.sha256(_RangeHandler.body[:400000])
    received = download_range(url, part, 400000, hasher, stop)
    server.shutdown()
    assert received == len(_RangeHandler.body) - 400000
    assert part.read_bytes() == _RangeHandler.body
    assert hasher.hexdigest() == hashlib.sha256(_RangeHandler.body).hexdigest()


class _StallHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"x" * 1024
        self.send_response(206)
        self.send_header("Content-Range", "bytes 0-1023/1048576")
        self.send_header("Content-Length", "1048576")
        self.end_headers()
        self.wfile.write(body)
        import time as _time

        _time.sleep(30)  # then stall: never send the rest

    def log_message(self, *args):
        pass


def test_download_range_watchdog_aborts_stall(tmp_path):
    import pytest as _pytest

    server = HTTPServer(("127.0.0.1", 0), _StallHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/f"
    stop = threading.Event()
    part = tmp_path / "f.part"
    with _pytest.raises(TimeoutError):
        download_range(
            url, part, 0, None, stop, timeout=10, min_bytes_per_s=10**12, stall_grace_s=1.0
        )
    server.shutdown()


def test_state_and_status_helpers(tmp_path):
    state = new_state("repo", REVISION)
    state["files"]["a.safetensors"] = {"status": "verified", "verified_bytes": 10}
    atomic_write_json(tmp_path / "s.json", state)
    assert json.loads((tmp_path / "s.json").read_text())["revision"] == (REVISION)
    status = sanitized_status(state, {"totals": {"tensor_bytes": 10}})
    assert status["complete"] is True
    assert status["shard_verified"] == 1
    assert "/" not in status.get("repo_id", "repo")
