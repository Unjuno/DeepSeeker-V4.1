#!/usr/bin/env python3
"""Issue #8: P1 routing-trace collection entry point.

    python scripts/collect_trace.py emit-synthetic --requests 4 --tokens 32 --out trace.jsonl
    python scripts/collect_trace.py validate --trace trace.jsonl
    python scripts/collect_trace.py stats --trace trace.jsonl
    python scripts/collect_trace.py readiness

Online collection (hooks into the reference runtime) unlocks when
`readiness` reports checkpoint + runtime deps present. Until then,
synthetic traces feed P2 simulator development.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import check_checkpoint_readiness, load_json, resolve_model_root
from deepseeker.trace import (
    arch_config,
    emit_synthetic,
    layer_histogram,
    make_header,
    read_trace,
    trace_stats,
    write_trace,
)

REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
ONLINE_MODULES = ("torch", "tilelang", "PIL", "transformers", "safetensors")


def pinned_revision() -> str:
    path = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    return load_json(path)["resolved_revision"]


def manifest() -> dict:
    revision = pinned_revision()
    path = REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    return load_json(path)


def cmd_emit(args) -> int:
    cfg = arch_config(manifest())
    header = make_header(REPO_ID, pinned_revision(), "synthetic", cfg, seed=args.seed)
    records = emit_synthetic(
        header,
        n_requests=args.requests,
        tokens_per_request=args.tokens,
        seed=args.seed,
        stickiness=args.stickiness,
    )
    count = write_trace(args.out, header, records)
    print(f"wrote {count} records ({args.requests}x{args.tokens}x{cfg['n_layers']}) to {args.out}")
    return 0


def cmd_validate(args) -> int:
    try:
        header, records = read_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"FAIL validate: {exc}")
        return 1
    print(f"PASS validate: {len(records)} records, schema {header['schema']}")
    return 0


def cmd_stats(args) -> int:
    try:
        header, records = read_trace(args.trace)
    except (OSError, ValueError) as exc:
        print(f"FAIL stats: {exc}")
        return 1
    stats = trace_stats(header, records)
    print(json.dumps(stats, indent=1))
    if args.top:
        hist = layer_histogram(records, header["n_layers"])
        for layer in range(min(args.top, header["n_layers"])):
            top = sorted(hist[layer].items(), key=lambda kv: -kv[1])[:5]
            print(f"layer {layer}: " + ", ".join(f"e{e}={c}" for e, c in top))
    return 0


def cmd_readiness(args) -> int:
    model_root = resolve_model_root(REPO_ROOT, args.model_root)
    state_path = model_root / ".deepseeker-checkpoint.json"
    state = load_json(state_path) if state_path.exists() else {}
    readiness = check_checkpoint_readiness(model_root, state)
    missing = [m for m in ONLINE_MODULES if importlib.util.find_spec(m) is None]
    shard_ok = readiness["verified"] == 48
    small_ok = all(readiness["small_files"].values())
    print(f"checkpoint shards verified: {readiness['verified']}/48")
    print(f"small files: {sum(readiness['small_files'].values())}/{len(readiness['small_files'])}")
    print(f"runtime modules missing: {missing or 'none'}")
    if shard_ok and small_ok and not missing:
        print("readiness: READY for online collection")
        return 0
    print("readiness: BLOCKED (synthetic traces only)")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 routing-trace collection.")
    parser.add_argument("--model-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    emit = sub.add_parser("emit-synthetic")
    emit.add_argument("--requests", type=int, default=4)
    emit.add_argument("--tokens", type=int, default=32)
    emit.add_argument("--seed", type=int, default=0)
    emit.add_argument("--stickiness", type=float, default=0.7)
    emit.add_argument("--out", type=Path, required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--trace", type=Path, required=True)

    stats = sub.add_parser("stats")
    stats.add_argument("--trace", type=Path, required=True)
    stats.add_argument("--top", type=int, default=3)

    sub.add_parser("readiness")
    args = parser.parse_args()
    if args.command == "emit-synthetic":
        return cmd_emit(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "stats":
        return cmd_stats(args)
    return cmd_readiness(args)


if __name__ == "__main__":
    raise SystemExit(main())
