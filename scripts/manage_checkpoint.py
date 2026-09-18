#!/usr/bin/env python3
"""Local checkpoint manager for the pinned DeepSeek-V4.1-Flash revision.

    python scripts/manage_checkpoint.py preflight
    python scripts/manage_checkpoint.py download
    python scripts/manage_checkpoint.py verify
    python scripts/manage_checkpoint.py status

One physical copy under an ignored model root (default
``models/DeepSeek-V4.1-Flash``). Downloads resume per shard, hash
SHA-256 while streaming, and reconcile headers against the Issue #2
manifest. Never touches Git history with weight bytes.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.checkpoint import (
    STATE_FILENAME,
    atomic_write_json,
    disk_avail_bytes,
    download_range,
    expected_files,
    git_ignored,
    hash_prefix,
    new_state,
    sanitized_status,
    temp_path_final_swap,
    verify_shard_against_manifest,
)

REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
CONCURRENCY = 3


def pinned_revision() -> str | None:
    path = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / ("DEEPSEEKER_UPSTREAM.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("resolved_revision")
    except (OSError, ValueError):
        return None


def model_root(args) -> Path:
    import os

    if args.model_root:
        return Path(args.model_root)
    env = os.environ.get("DEEPSEEKER_MODEL_ROOT")
    if env:
        return Path(env)
    return REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"


def load_manifest() -> dict:
    revision = pinned_revision()
    path = REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def state_path(root: Path) -> Path:
    return root / STATE_FILENAME


def load_state(root: Path, revision: str) -> dict:
    path = state_path(root)
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("revision") == revision:
                return state
        except (OSError, ValueError):
            pass
    return new_state(REPO_ID, revision)


def remote_file_info(revision: str, filename: str) -> dict:
    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(REPO_ID, paths=[filename], revision=revision, expand=True)
    if not infos:
        raise RuntimeError(f"no upstream info for {filename}")
    info = infos[0]
    lfs = getattr(info, "lfs", None)
    return {
        "size": getattr(info, "size", None),
        "sha256": getattr(lfs, "sha256", None) if lfs else None,
    }


def cmd_preflight(args) -> int:
    revision = pinned_revision()
    if revision is None:
        print("FAIL: pinned revision unknown (run fetch_upstream first)")
        return 1
    root = model_root(args)
    manifest = load_manifest()
    weight_map = {t["name"]: t["shard"] for t in manifest["tensors"]}
    files = expected_files(weight_map)
    print(f"pinned revision: {revision}")
    print(f"model root: {root} ({'ignored' if git_ignored(REPO_ROOT, root) else 'NOT IGNORED'})")
    if not git_ignored(REPO_ROOT, root):
        print("FAIL: model root is not Git-ignored")
        return 1
    remote_total = 0
    for name in files:
        info = remote_file_info(revision, name)
        if info["size"] is None:
            print(f"FAIL: no upstream size for {name}")
            return 1
        remote_total += info["size"]
    avail = disk_avail_bytes(root.parent if root.exists() else REPO_ROOT)
    print(f"expected files: {len(files)} ({len(files) - 4} shards + 4 small)")
    print(f"expected remote bytes: {remote_total}")
    print(f"disk available bytes: {avail}")
    headroom = (avail or 0) - remote_total
    print(f"headroom bytes: {headroom}")
    if avail is None or headroom < 100 * 1024**3:
        print("FAIL: need >=100 GiB headroom beyond checkpoint")
        return 1
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(REPO_ID, revision=revision)
        print("upstream reachable: yes (public, no auth needed)")
    except Exception as exc:  # noqa: BLE001 - network errors vary
        print(f"FAIL: upstream unreachable: {exc}")
        return 1
    state = load_state(root, revision)
    done = [f for f, s in state.get("files", {}).items() if s.get("status") == "verified"]
    print(f"already verified: {len(done)}/{len(files) - 4} shards")
    print("preflight: PASS")
    return 0


def _download_small(root: Path, revision: str, name: str, stop: threading.Event) -> dict:
    dest = root / name
    if dest.exists() and dest.stat().st_size > 0:
        return {"file": name, "status": "present", "bytes": dest.stat().st_size}
    # Revision-pinned URL; small enough to fetch directly, no cache copy.
    url = f"https://huggingface.co/{REPO_ID}/resolve/{revision}/{name}"
    req = urllib.request.Request(url)
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        while not stop.is_set():
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return {
        "file": name,
        "status": "downloaded",
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def _download_shard(
    root: Path,
    revision: str,
    shard: str,
    size: int,
    sha256: str | None,
    state: dict,
    stop: threading.Event,
) -> dict:
    dest = root / shard
    part = root / f"{shard}.part"
    entry = state["files"].get(shard, {})
    if (
        dest.exists()
        and dest.stat().st_size == size
        and entry.get("status") == "verified"
        and entry.get("sha256") == sha256
    ):
        return {"file": shard, "status": "reused-verified", "bytes": 0}
    if dest.exists() and dest.stat().st_size == size and not part.exists():
        # Complete file on disk but not yet adopted (previous run or
        # copy): hash in place instead of re-downloading.
        digest = hash_prefix(dest, size).hexdigest()
        if sha256 and digest != sha256:
            dest.unlink()
        else:
            return {
                "file": shard,
                "status": "downloaded",
                "bytes": 0,
                "resumed_bytes": size,
                "sha256": digest,
                "elapsed_s": 0.0,
                "adopted": True,
            }
    offset = part.stat().st_size if part.exists() else 0
    if dest.exists() and not part.exists() and dest.stat().st_size != size:
        dest.unlink()  # stale complete-size mismatch: restart shard
        offset = 0
    hasher = hash_prefix(part, offset) if offset else hashlib.sha256()
    resumed = offset
    url = f"https://huggingface.co/{REPO_ID}/resolve/{revision}/{shard}"
    started = time.perf_counter()
    received = download_range(url, part, offset, hasher, stop)
    if stop.is_set():
        return {"file": shard, "status": "interrupted", "bytes": received}
    digest = hasher.hexdigest()
    if sha256 and digest != sha256:
        part.unlink(missing_ok=True)
        return {"file": shard, "status": "checksum-mismatch", "expected": sha256, "got": digest}
    final_size = part.stat().st_size
    if final_size != size:
        return {"file": shard, "status": "size-mismatch", "expected": size, "got": final_size}
    temp_path_final_swap(part, dest)
    return {
        "file": shard,
        "status": "downloaded",
        "bytes": received,
        "resumed_bytes": resumed,
        "sha256": digest,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def cmd_download(args) -> int:
    import concurrent.futures
    import signal

    revision = pinned_revision()
    root = model_root(args)
    if not git_ignored(REPO_ROOT, root):
        print("FAIL: model root is not Git-ignored")
        return 1
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    weight_map = {t["name"]: t["shard"] for t in manifest["tensors"]}
    shards = sorted(set(weight_map.values()))
    state = load_state(root, revision)
    stop = threading.Event()

    def _request_stop(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    pass_number = 0
    started = time.perf_counter()
    total_new = 0
    try:
        from huggingface_hub import HfApi

        infos = {}
        for i in range(0, len(shards), 20):
            batch = HfApi().get_paths_info(
                REPO_ID, paths=shards[i : i + 20], revision=revision, expand=True
            )
            for info in batch:
                lfs = getattr(info, "lfs", None)
                infos[info.path] = {
                    "size": info.size,
                    "sha256": lfs.sha256 if lfs else None,
                }

        def handle_result(shard: str, result: dict) -> None:
            nonlocal total_new
            entry = state["files"].get(shard, {})
            if result["status"] == "downloaded":
                total_new += result["bytes"]
                entry.update(
                    {
                        "status": "downloaded-unverified",
                        "size": infos[shard]["size"],
                        "sha256": result["sha256"],
                        "downloaded_bytes": result["bytes"],
                        "resumed_bytes": result.get("resumed_bytes", 0),
                    }
                )
            elif result["status"] == "reused-verified":
                pass
            else:
                entry.update({"status": result["status"], "detail": str(result)[:300]})
                print(f"{shard}: {result['status']}")
            state["files"][shard] = entry
            atomic_write_json(state_path(root), state)
            done = sum(
                1
                for f in state["files"].values()
                if f.get("status") in ("downloaded-unverified", "verified")
            )
            print(f"[{done}/{len(shards)}] {shard}: {result['status']}", flush=True)

        pass_number = 0
        while not stop.is_set():
            pass_number += 1
            pending = [
                s
                for s in shards
                if state["files"].get(s, {}).get("status")
                not in ("downloaded-unverified", "verified")
            ]
            if not pending:
                break
            if pass_number > 1:
                print(
                    f"pass {pass_number}: {len(pending)} shards pending; backing off 60s",
                    flush=True,
                )
                for _ in range(60):
                    if stop.is_set():
                        break
                    time.sleep(1)
            if stop.is_set():
                break
            print(
                f"pass {pass_number}: downloading {len(pending)} shards",
                flush=True,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = {
                    pool.submit(
                        _download_shard,
                        root,
                        revision,
                        shard,
                        infos[shard]["size"],
                        infos[shard]["sha256"],
                        state,
                        stop,
                    ): shard
                    for shard in pending
                }
                for fut in concurrent.futures.as_completed(futures):
                    shard = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:  # noqa: BLE001 - keep going
                        result = {
                            "file": shard,
                            "status": "download-error",
                            "detail": f"{type(exc).__name__}: {exc}"[:300],
                        }
                    handle_result(shard, result)
                    if stop.is_set():
                        break
                # Retry failed shards (transient errors, bad ranges) up to
                # two more times; checksum failures restart fresh.
                for attempt in (2, 3):
                    if stop.is_set():
                        break
                    failed = [
                        s
                        for s in pending
                        if state["files"].get(s, {}).get("status")
                        not in (
                            "downloaded-unverified",
                            "verified",
                            "reused-verified",
                        )
                    ]
                    if not failed:
                        break
                    print(
                        f"retry pass {attempt}: {len(failed)} shards",
                        flush=True,
                    )
                    for shard in failed:
                        if stop.is_set():
                            break
                        try:
                            result = _download_shard(
                                root,
                                revision,
                                shard,
                                infos[shard]["size"],
                                infos[shard]["sha256"],
                                state,
                                stop,
                            )
                        except Exception as exc:  # noqa: BLE001 - keep going
                            result = {
                                "file": shard,
                                "status": "download-error",
                                "detail": (f"{type(exc).__name__}: {exc}")[:300],
                            }
                        handle_result(shard, result)
        for name in (
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        ):
            result = _download_small(root, revision, name, stop)
            if stop.is_set() and (root / name).stat().st_size == 0:
                (root / name).unlink(missing_ok=True)
                break
            print(f"{name}: {result['status']}", flush=True)
    except KeyboardInterrupt:
        stop.set()
        print("interrupted; partial state kept for resume")
        return 130
    elapsed = time.perf_counter() - started
    print(
        f"downloaded {total_new} new bytes in {elapsed:.0f}s "
        f"({total_new / elapsed / 1024 / 1024:.1f} MiB/s)"
    )
    if stop.is_set():
        print("stopped early by signal; partial .part state kept")
        print("rerun download to resume; completed shards are reused")
        return 130
    print("next: python scripts/manage_checkpoint.py verify")
    return 0


def cmd_verify(args) -> int:
    revision = pinned_revision()
    root = model_root(args)
    manifest = load_manifest()
    weight_map = {t["name"]: t["shard"] for t in manifest["tensors"]}
    shards = sorted(set(weight_map.values()))
    state = load_state(root, revision)
    from huggingface_hub import HfApi

    infos = {}
    for i in range(0, len(shards), 20):
        batch = HfApi().get_paths_info(
            REPO_ID, paths=shards[i : i + 20], revision=revision, expand=True
        )
        for info in batch:
            lfs = getattr(info, "lfs", None)
            infos[info.path] = {
                "size": info.size,
                "sha256": lfs.sha256 if lfs else None,
            }
    started = time.perf_counter()
    verified_bytes = 0
    failures = []
    for i, shard in enumerate(shards, 1):
        dest = root / shard
        entry = state["files"].get(shard, {})
        if not dest.exists():
            failures.append(f"{shard}: missing")
            continue
        size_ok = dest.stat().st_size == infos[shard]["size"]
        digest_ok = entry.get("sha256") == infos[shard]["sha256"]
        header = verify_shard_against_manifest(dest, shard, manifest)
        ok = (
            size_ok
            and digest_ok
            and header["ok"]
            and (entry.get("status") in ("downloaded-unverified", "verified"))
        )
        if ok:
            entry.update(
                {
                    "status": "verified",
                    "verified_bytes": dest.stat().st_size,
                    "verified_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                }
            )
            verified_bytes += dest.stat().st_size
        else:
            problems = (
                header["problems"]
                + ([] if size_ok else ["size mismatch"])
                + ([] if digest_ok else ["digest mismatch"])
            )
            failures.append(f"{shard}: {problems}")
            entry.update({"status": "failed", "problems": problems})
        state["files"][shard] = entry
        atomic_write_json(state_path(root), state)
        print(f"[{i}/{len(shards)}] {shard}: {'verified' if ok else 'FAILED'}", flush=True)
    elapsed = time.perf_counter() - started
    total_tensors = manifest["totals"]["tensor_bytes"]
    print(f"verified {verified_bytes} bytes in {elapsed:.0f}s")
    print(f"manifest tensor bytes: {total_tensors}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for failure in failures[:10]:
            print(f"  {failure}")
        return 1
    print("verify: 48/48 PASS")
    return 0


def cmd_status(args) -> int:
    revision = pinned_revision()
    root = model_root(args)
    manifest = load_manifest()
    state = load_state(root, revision)
    status = sanitized_status(state, manifest)
    print(json.dumps(status, indent=1))
    out = REPO_ROOT / "profiles" / "m1-max-64gb" / "checkpoint-status.json"
    if args.write_status:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(status, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out} (sanitized: no absolute paths)")
    return 0 if status["complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinned DeepSeek checkpoint manager.")
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument(
        "--write-status", action="store_true", help="write sanitized checkpoint-status.json"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("download")
    sub.add_parser("verify")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "preflight":
        return cmd_preflight(args)
    if args.command == "download":
        return cmd_download(args)
    if args.command == "verify":
        return cmd_verify(args)
    return cmd_status(args)


if __name__ == "__main__":
    raise SystemExit(main())
