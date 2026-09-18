"""Local checkpoint management: preflight/download/verify/status.

One physical copy, direct HTTP Range downloads with resume, SHA-256
verified against upstream LFS metadata while downloading (no second
verification read required), manifest-driven header reconciliation.

Nothing here touches Git history with weight bytes: model roots must
resolve inside ignored paths (checked, not assumed).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024
STATE_FILENAME = ".deepseeker-checkpoint.json"
SMALL_FILES = (
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
)


def run_command(argv: list[str], timeout: int = 30) -> str | None:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_ignored(repo_root: Path, path: Path) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "-q", str(path)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return proc.returncode == 0


def disk_avail_bytes(path: Path) -> int | None:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def expected_files(weight_map: dict[str, str]) -> list[str]:
    shards = sorted(set(weight_map.values()))
    return [*shards, *SMALL_FILES]


def new_state(repo_id: str, revision: str, tool: str = "manage_checkpoint/1") -> dict:
    return {
        "schema": "deepseeker.checkpoint-state/v1",
        "repo_id": repo_id,
        "revision": revision,
        "tool": tool,
        "files": {},
    }


def download_range(
    url: str,
    dest_part: Path,
    offset: int,
    hasher: hashlib._Hash | None,
    stop: threading.Event,
    timeout: int = 120,
    min_bytes_per_s: int = 500 * 1024,
    stall_grace_s: float = 60.0,
) -> int:
    """Append url[offset:] to dest_part; feed hasher. Returns new bytes.

    Raises TimeoutError when throughput collapses (stalled CDN edge):
    after 30 s, sustained rate must stay above min_bytes_per_s.
    """
    import time

    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
    received = 0
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 206 and not (resp.status == 200 and offset == 0):
            raise RuntimeError(f"unexpected status {resp.status}")
        with open(dest_part, "ab") as f:
            while not stop.is_set():
                chunk = resp.read(CHUNK_BYTES)
                if not chunk:
                    break
                f.write(chunk)
                if hasher is not None:
                    hasher.update(chunk)
                received += len(chunk)
                elapsed = time.perf_counter() - started
                if elapsed > stall_grace_s and received / elapsed < min_bytes_per_s:
                    raise TimeoutError(
                        f"throughput collapsed: {received} B in {elapsed:.0f}s from {url[:80]}"
                    )
    return received


def hash_prefix(path: Path, length: int) -> hashlib._Hash:
    """SHA-256 over the first length bytes (resume support)."""
    hasher = hashlib.sha256()
    remaining = length
    with open(path, "rb") as f:
        while remaining > 0:
            chunk = f.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                break
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError("file too short for safetensors header")
        (header_len,) = struct.unpack("<Q", prefix[:8])
        raw = f.read(header_len)
    if len(raw) < header_len:
        raise ValueError("truncated safetensors header")
    return json.loads(raw.decode("utf-8")), header_len


def verify_shard_against_manifest(path: Path, shard: str, manifest: dict) -> dict:
    """Check size, header, tensor names/shapes/dtypes/offsets, bounds."""
    expected = [t for t in manifest["tensors"] if t["shard"] == shard]
    file_size = path.stat().st_size
    header, header_len = read_header(path)
    header_tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    manifest_names = {t["name"] for t in expected}
    problems = []
    if set(header_tensors) != manifest_names:
        missing = sorted(manifest_names - set(header_tensors))
        extra = sorted(set(header_tensors) - manifest_names)
        problems.append(f"name mismatch: missing={missing[:5]} extra={extra[:5]}")
    for t in expected:
        info = header_tensors.get(t["name"])
        if info is None:
            continue
        if (
            list(info["shape"]) != t["shape"]
            or info["dtype"] != t["dtype"]
            or list(info["data_offsets"]) != t["data_range"]
        ):
            problems.append(f"metadata mismatch: {t['name']}")
            continue
        start, end = t["data_range"]
        if not 0 <= start < end:
            problems.append(f"bad range: {t['name']}")
            continue
        abs_end = 8 + header_len + end
        if abs_end > file_size:
            problems.append(f"out of bounds: {t['name']}")
    tensor_bytes = sum(t["bytes"] for t in expected)
    return {
        "shard": shard,
        "file_bytes": file_size,
        "header_bytes": 8 + header_len,
        "tensors": len(expected),
        "tensor_bytes": tensor_bytes,
        "overhead_bytes": file_size - 8 - header_len - tensor_bytes,
        "ok": not problems,
        "problems": problems,
    }


def git_safety_scan(repo_root: Path) -> dict:
    """Fail if any weight/payload/cache artifact is tracked or staged."""
    issues = []
    suspicious = run_command(
        ["git", "-C", str(repo_root), "ls-files", "*.safetensors", "models/", ".cache/"], timeout=60
    )
    if suspicious and suspicious.strip():
        issues.append(f"tracked payload files: {suspicious.strip()[:500]}")
    staged = run_command(
        ["git", "-C", str(repo_root), "diff", "--cached", "--name-only"], timeout=60
    )
    big_allowed = {
        (
            "profiles/deepseek-v4.1-flash/"
            "dba1be0a40aa45a94ad051997016db3960a90277/model-manifest.json"
        ),
    }
    if staged:
        for name in staged.strip().splitlines():
            if name.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")):
                issues.append(f"staged payload: {name}")
            if name.startswith(("models/", ".cache/")):
                issues.append(f"staged cache/model path: {name}")
            p = repo_root / name
            if p.exists() and p.stat().st_size > 100 * 1024 * 1024 and name not in big_allowed:
                issues.append(f"staged file >100MB: {name}")
    return {"ok": not issues, "issues": issues}


def sanitized_status(state: dict, manifest: dict | None) -> dict:
    files = state.get("files", {})
    done = sum(1 for f in files.values() if f.get("status") == "verified")
    return {
        "schema": "deepseeker.checkpoint-status/v1",
        "repo_id": state.get("repo_id"),
        "revision": state.get("revision"),
        "shard_total": len([f for f in files if f.endswith(".safetensors")]),
        "shard_verified": done,
        "bytes_verified": sum(f.get("verified_bytes", 0) for f in files.values()),
        "expected_tensor_bytes": (manifest["totals"]["tensor_bytes"] if manifest else None),
        "complete": (done == len([f for f in files if f.endswith(".safetensors")]) and bool(files)),
        "verification": "sha256-vs-upstream-lfs + manifest reconciliation",
    }


def temp_path_final_swap(part: Path, final: Path) -> None:
    os.replace(part, final)


def atomic_write_json(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def parse_swap_used(swapusage: str) -> str:
    m = re.search(r"used\s*=\s*([\d.]+\w*)", swapusage)
    return m.group(1) if m else "unknown"
