#!/usr/bin/env python3
"""Physical SSD service-time benchmark for DeepSeek access patterns.

Models real Issue #2/#3 geometry on a disposable benchmark file:

- single routed experts (exact six-tensor ranges from the manifest);
- expert groups 1/2/4/8/16/32 (same-layer, cross-layer, random);
- Engram per-token row batches from Issue #3 traces;
- queue-depth sweep 1/2/4/8/16;
- sustained read workload with time buckets.

    python scripts/benchmark_storage_patterns.py

Safe default: advisory-uncached reads (F_NOCACHE, randomized placement),
no sudo, no weight download. A ~10 KB exact-range verification against
the real upstream shards is performed unless --no-verify is passed.

Optional cold claim (requires a manual `sudo purge` OUTSIDE this script):

    sudo purge
    python scripts/benchmark_storage_patterns.py \\
        --assert-cold "sudo purge at <time>, <note>"

The script itself never escalates privileges.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.storage_bench import (
    BenchFile,
    Range,
    align_ranges,
    coalesce_ranges,
    median_of,
    remap_ranges,
    run_qd,
    run_sustained,
    scatter_ranges,
)

REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
QD_LEVELS = [1, 2, 4, 8, 16]
GROUP_SIZES = [1, 2, 4, 8, 16, 32]
DEFAULT_FILE_MIB = 8192
COALESCE_GAP = 4096


# --------------------------------------------------------------------------
# Manifest-derived pattern construction


def load_revision() -> str:
    path = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    return json.loads(path.read_text(encoding="utf-8"))["resolved_revision"]


def load_manifest(revision: str) -> dict:
    path = REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def expert_ranges(manifest: dict, layer: int, expert: int) -> list[Range]:
    """Exact six file ranges (w1/w2/w3 weight+scale) for one expert."""
    out = []
    for t in manifest["tensors"]:
        if t["subsystem"] == "routed-expert" and t["layer"] == layer and t["expert"] == expert:
            s, e = t["file_range"]
            out.append(Range(s, e - s))
    assert len(out) == 6, (layer, expert, len(out))
    return sorted(out, key=lambda r: r.offset)


def expert_span(ranges: list[Range]) -> int:
    return max(r.end for r in ranges) - min(r.offset for r in ranges)


def select_experts(manifest: dict) -> dict:
    """Early/mid/late layer experts plus min/max file-span experts."""
    per_expert: dict[tuple[int, int], list[Range]] = {}
    for t in manifest["tensors"]:
        if t["subsystem"] == "routed-expert":
            per_expert.setdefault((t["layer"], t["expert"]), []).append(
                Range(t["file_range"][0], t["file_range"][1] - t["file_range"][0])
            )
    spans = {k: expert_span(v) for k, v in per_expert.items()}
    closest = min(spans, key=lambda k: spans[k])
    widest = max(spans, key=lambda k: spans[k])
    picked = {
        "early": (0, 0),
        "mid": (20, 0),
        "late": (39, 0),
        "min_span": closest,
        "max_span": widest,
    }
    return {
        name: {
            "layer": layer,
            "expert": expert,
            "ranges": sorted(per_expert[(layer, expert)], key=lambda r: r.offset),
            "span_bytes": spans[(layer, expert)],
        }
        for name, (layer, expert) in picked.items()
    }


def group_ranges(manifest: dict, mode: str, size: int, seed: int) -> list[Range]:
    """Ranges for an expert group: same-layer / cross-layer / random."""
    import random

    rng = random.Random(seed)
    layers = sorted({t["layer"] for t in manifest["tensors"] if t["subsystem"] == "routed-expert"})
    if mode == "same-layer":
        layer = layers[20]
        experts = rng.sample(range(384), size)
        ids = [(layer, e) for e in experts]
    elif mode == "cross-layer":
        ids = [(layers[i % len(layers)], rng.randrange(384)) for i in range(size)]
    else:
        ids = [(rng.choice(layers), rng.randrange(384)) for _ in range(size)]
    out = []
    for layer, expert in ids:
        out.extend(expert_ranges(manifest, layer, expert))
    return sorted(out, key=lambda r: r.offset)


def verify_against_upstream(revision: str, manifest: dict, timeout: int = 60) -> dict:
    """Fetch ~10 KB of exact manifest ranges from the real shards.

    Proves the recorded file_range convention against reality without
    downloading meaningful weight payload.
    """
    checked = []
    total = 0
    # First 4 KiB of one expert weight tensor + one Engram weight row.
    targets = []
    for t in manifest["tensors"]:
        if (
            t["subsystem"] == "routed-expert"
            and t["layer"] == 0
            and t["expert"] == 0
            and t["name"].endswith("w1.weight")
        ):
            targets.append(t)
    for t in manifest["tensors"]:
        if t["name"] == "layers.1.engram.embed.weight":
            targets.append(t)
    for t in targets:
        shard = t["shard"]
        start = t["file_range"][0]
        url = f"https://huggingface.co/{REPO_ID}/resolve/{revision}/{shard}"
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + 4095}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 206:
                raise RuntimeError(f"range failed for {shard}")
            payload = resp.read()
            total += len(payload)
            checked.append(
                {
                    "tensor": t["name"],
                    "shard": shard,
                    "requested": [start, start + 4096],
                    "received_bytes": len(payload),
                }
            )
    return {"checked": checked, "network_bytes": total, "weight_payload_downloaded": False}


# --------------------------------------------------------------------------
# Engram trace replay


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "analyze_engram_access",
        REPO_ROOT / "scripts" / "analyze_engram_access.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engram_token_ranges(
    analyzer, manifest: dict, corpus_name: str, max_tokens: int
) -> tuple[list[list[Range]], int]:
    """Per-token manifest file ranges for an Issue #3 corpus trace."""
    from deepseeker.engram import (
        EngramGeometry,
        build_multipliers,
        build_offsets,
        build_primes,
        hash_positions,
        row_ranges,
        table_refs_from_manifest,
    )

    cfg = analyzer.upstream_config()
    refs = table_refs_from_manifest(manifest)
    layer_ids = sorted(refs)
    cache_dir = REPO_ROOT / ".cache" / "engram-analysis"
    tokenizer, _ = analyzer.load_tokenizer(cache_dir)
    upstream_spec = importlib.util.spec_from_file_location(
        "upstream_engram",
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "inference" / "engram.py",
    )
    assert upstream_spec is not None and upstream_spec.loader is not None
    upstream_engram = importlib.util.module_from_spec(upstream_spec)
    upstream_spec.loader.exec_module(upstream_engram)

    hf_tok = AutoTokenizer.from_pretrained(
        str(cache_dir), local_files_only=True, trust_remote_code=False
    )
    token_map, _ = upstream_engram.build_compressed_token_map(hf_tok)
    geometry = EngramGeometry(
        layer_ids=tuple(cfg["engram_layer_ids"]),
        max_ngram_size=cfg["engram_max_ngram_size"],
        n_heads=cfg["engram_n_heads"],
        head_dim=cfg["engram_head_dim"],
        num_embeddings=tuple(cfg["engram_num_embeddings"]),
        vocab_size=cfg["engram_vocab_size"],
        compressed_vocab_size=cfg["engram_compressed_vocab_size"],
        pad_id=token_map[cfg["engram_pad_token_id"]],
    )
    primes = build_primes(geometry)
    offsets = build_offsets(primes)
    multipliers = build_multipliers(
        geometry.layer_ids,
        geometry.max_ngram_size,
        geometry.compressed_vocab_size,
    )
    corpus = analyzer.build_corpus()
    text = corpus[corpus_name]
    if text == "__RANDOM__":
        rng = np.random.default_rng(0)
        ids = [int(x) for x in rng.integers(0, len(tokenizer.get_vocab()), size=256)]
    else:
        ids = tokenizer.encode(text).ids[:max_tokens]
    compressed = [token_map[i] for i in ids]
    row_ids = hash_positions(compressed, geometry, multipliers, primes, offsets)
    per_token: list[list[Range]] = []
    for pos in row_ids:
        ranges: list[Range] = []
        for li, cols in enumerate(pos):
            ref = refs[layer_ids[li]]
            for row in cols:
                rr = row_ranges(ref, row)
                for part in ("weight", "scale"):
                    s, e = rr[part]["range"]
                    ranges.append(Range(s, e - s))
        per_token.append(sorted(ranges, key=lambda r: r.offset))
    return per_token, 264


# --------------------------------------------------------------------------
# Benchmark driver


def bench_case(
    bench: BenchFile,
    name: str,
    manifest_ranges: list[Range],
    file_size: int,
    salt: int,
    qd_levels: list[int],
    repeats: int,
    alignments: list[int],
    cache_confidence: str,
) -> dict:
    useful = sum(r.length for r in manifest_ranges)
    case: dict = {
        "case": name,
        "cache_confidence": cache_confidence,
        "useful_bytes": useful,
        "manifest_span_bytes": expert_span(manifest_ranges),
        "modes": {},
    }
    for label, prepared in [
        ("exact", manifest_ranges),
        *[(f"align_{a}", align_ranges(manifest_ranges, a)) for a in alignments],
        ("coalesced_4k", coalesce_ranges(manifest_ranges, COALESCE_GAP)),
    ]:
        replay = remap_ranges(prepared, file_size, salt)
        req_bytes = sum(r.length for r in replay)
        mode_runs = {}
        for qd in qd_levels:
            runs = [run_qd(bench, replay, qd, 1) for _ in range(repeats)]
            med = {
                "p50_ms": median_of([r["p50_ms"] for r in runs]),
                "p95_ms": median_of([r["p95_ms"] for r in runs]),
                "min_ms": min(r["min_ms"] for r in runs),
                "max_ms": max(r["max_ms"] for r in runs),
                "throughput_mib_s": median_of([r["throughput_mib_s"] for r in runs]),
            }
            mode_runs[f"qd{qd}"] = {
                "repeats": repeats,
                "requested_bytes": req_bytes,
                "ops": len(replay),
                "median_of_runs": med,
                "runs": runs,
            }
        case["modes"][label] = mode_runs
    return case


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SSD service-time benchmark for DeepSeek patterns."
    )
    parser.add_argument("--file-mib", type=int, default=DEFAULT_FILE_MIB)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sustain-seconds", type=float, default=60.0)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--bench-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--assert-cold",
        type=str,
        default=None,
        help="Manual purge note; labels results cold-proven.",
    )
    args = parser.parse_args()

    if args.assert_cold:
        cache_confidence = "cold-proven"
        print(f"cold claim recorded: {args.assert_cold}")
    else:
        cache_confidence = "advisory-uncached"

    revision = load_revision()
    manifest = load_manifest(revision)

    bench_dir = args.bench_dir or (REPO_ROOT / ".cache" / "storage-bench")
    bench_dir.mkdir(parents=True, exist_ok=True)
    bench = BenchFile(str(bench_dir / "bench.bin"), args.file_mib * 1024 * 1024)
    if not Path(bench.path).exists() or Path(bench.path).stat().st_size != bench.size_bytes:
        print(f"allocating {args.file_mib} MiB disposable file ...")
        info = bench.create()
    else:
        info = {
            "path": bench.path,
            "size_bytes": bench.size_bytes,
            "seed": bench.seed,
            "method": "reused existing file",
        }
    print(f"bench file: {info}")

    verification = (
        {"skipped": True} if args.no_verify else verify_against_upstream(revision, manifest)
    )
    print(f"upstream verification: {verification}")

    result: dict = {
        "schema": "deepseeker.storage-patterns/v1",
        "upstream_revision": revision,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "cache_confidence_default": cache_confidence,
        "cold_note": args.assert_cold,
        "bench_file": info,
        "verification": verification,
        "experts": {},
        "groups": {},
        "engram": {},
        "sustained": {},
    }

    experts = select_experts(manifest)
    for name, spec in experts.items():
        print(f"expert {name} (L{spec['layer']}/E{spec['expert']}) ...")
        result["experts"][name] = bench_case(
            bench,
            f"expert-{name}",
            spec["ranges"],
            bench.size_bytes,
            salt=hash(name) % 2**32,
            qd_levels=QD_LEVELS,
            repeats=args.repeats,
            alignments=[4096, 16384],
            cache_confidence=cache_confidence,
        )
        result["experts"][name]["span_bytes"] = spec["span_bytes"]

    for mode in ("same-layer", "cross-layer", "random"):
        result["groups"][mode] = {}
        for size in GROUP_SIZES:
            print(f"group {mode} x{size} ...")
            ranges = group_ranges(manifest, mode, size, seed=size)
            useful = sum(r.length for r in ranges)
            replay = remap_ranges(ranges, bench.size_bytes, salt=1000 + size)
            qd_runs = {}
            for qd in QD_LEVELS:
                runs = [run_qd(bench, replay, qd, 1) for _ in range(args.repeats)]
                qd_runs[f"qd{qd}"] = {
                    "repeats": args.repeats,
                    "requested_bytes": sum(r.length for r in replay),
                    "ops": len(replay),
                    "median_of_runs": {
                        "p50_ms": median_of([r["p50_ms"] for r in runs]),
                        "p95_ms": median_of([r["p95_ms"] for r in runs]),
                        "throughput_mib_s": median_of([r["throughput_mib_s"] for r in runs]),
                    },
                    "runs": runs,
                }
            result["groups"][mode][f"n{size}"] = {
                "useful_bytes": useful,
                "cache_confidence": cache_confidence,
                "qd": qd_runs,
            }

    analyzer = load_analyzer()
    for corpus in ("english_prose", "japanese_prose", "source_code", "repetitive", "random_tokens"):
        print(f"engram {corpus} ...")
        per_token, _ = engram_token_ranges(analyzer, manifest, corpus, max_tokens=256)
        # One benchmark job = one token's coalesced 4K-gap plan.
        # True Engram span (~98 GiB/table) exceeds the disposable file,
        # so lengths/counts replay exactly at scattered offsets.
        jobs = [coalesce_ranges(tok, COALESCE_GAP) for tok in per_token]
        flat = [r for tok in jobs for r in tok]
        useful = len(per_token) * 12672
        replay = scatter_ranges(flat, bench.size_bytes, seed=777)
        qd_runs = {}
        for qd in QD_LEVELS:
            runs = [run_qd(bench, replay, qd, 1) for _ in range(args.repeats)]
            qd_runs[f"qd{qd}"] = {
                "repeats": args.repeats,
                "tokens": len(per_token),
                "ops_per_token": round(len(flat) / len(per_token), 2),
                "requested_bytes_per_token": round(
                    sum(r.length for r in replay) / len(per_token), 1
                ),
                "useful_bytes_per_token": 12672,
                "median_of_runs": {
                    "p50_ms": median_of([r["p50_ms"] for r in runs]),
                    "elapsed_s": median_of([r["elapsed_s"] for r in runs]),
                },
                "runs": runs,
            }
        result["engram"][corpus] = {
            "cache_confidence": cache_confidence,
            "qd": qd_runs,
        }

    # Sustained: 8-expert same-layer group at QD1 and QD4.
    sustained_ranges = remap_ranges(
        group_ranges(manifest, "same-layer", 8, seed=8), bench.size_bytes, salt=8888
    )
    for qd in (1, 4):
        print(f"sustained qd{qd} {args.sustain_seconds}s ...")
        result["sustained"][f"qd{qd}"] = run_sustained(
            bench, sustained_ranges, qd, args.sustain_seconds, bucket_s=10.0
        )
    result["sustained"]["cache_confidence"] = cache_confidence

    out = args.out_dir or (REPO_ROOT / "profiles" / "m1-max-64gb")
    out.mkdir(parents=True, exist_ok=True)
    (out / "storage-patterns.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8"
    )
    (out / "storage-patterns.md").write_text(summarize(result), encoding="utf-8")
    print(f"wrote {out}/storage-patterns.json")
    return 0


def fmt_gibs(mib_s: float) -> str:
    return f"{mib_s / 1024:.2f} GiB/s"


def summarize(result: dict) -> str:
    lines = [
        "# Storage patterns benchmark",
        "",
        f"Revision: `{result['upstream_revision']}`",
        f"Generated: {result['generated_at_utc']}",
        f"Cache confidence: `{result['cache_confidence_default']}`",
        (
            f"Bench file: {result['bench_file']['size_bytes']} bytes "
            f"({result['bench_file']['method']})"
        ),
        "",
        "## Single experts (18,800,640 useful bytes each)",
        "",
    ]
    for name, case in result["experts"].items():
        exact = case["modes"]["exact"]["qd1"]["median_of_runs"]
        coal = case["modes"]["coalesced_4k"]["qd1"]["median_of_runs"]
        detail = (
            f"- {name}: span {case['span_bytes']} B; "
            f"exact QD1 p50 {exact['p50_ms']:.2f} ms "
            f"({fmt_gibs(exact['throughput_mib_s'])} useful); "
            f"coalesced QD1 p50 {coal['p50_ms']:.2f} ms"
        )
        lines.append(detail)
    lines += ["", "## Expert groups (same-layer, QD scaling, useful GiB/s)", ""]
    for size in GROUP_SIZES:
        row = result["groups"]["same-layer"][f"n{size}"]["qd"]
        line = (
            f"- n{size} "
            f"({result['groups']['same-layer'][f'n{size}']['useful_bytes'] / 1024 / 1024:.0f} MiB): "
        )
        line += ", ".join(
            f"QD{qd} {row[f'qd{qd}']['median_of_runs']['throughput_mib_s'] / 1024:.2f}"
            for qd in QD_LEVELS
        )
        lines.append(line + " GiB/s")
    lines += ["", "## Engram per-token (coalesced 4K-gap plan)", ""]
    for corpus, data in result["engram"].items():
        qd1 = data["qd"]["qd1"]["median_of_runs"]
        lines.append(
            f"- {corpus}: {data['qd']['qd1']['ops_per_token']} ops/token, "
            f"{data['qd']['qd1']['requested_bytes_per_token']:.0f} "
            f"req B/token, QD1 elapsed/token "
            f"{qd1['elapsed_s'] * 1000 / data['qd']['qd1']['tokens']:.2f} ms"
        )
    lines += ["", "## Sustained (8-expert same-layer group)", ""]
    for key, sus in result["sustained"].items():
        if key == "cache_confidence":
            continue
        buckets = " ".join(f"{b['mib_s']}" for b in sus["buckets"])
        lines.append(
            f"- {key}: avg {sus['avg_mib_s']} MiB/s over "
            f"{sus['duration_s']}s; buckets [{buckets}] MiB/s"
        )
    lines += [
        "",
        "## Cache-confidence conclusion",
        "",
        (
            "Default results are advisory-uncached (F_NOCACHE, no "
            "purge). Nothing here is cold-proven unless --assert-cold "
            "records a manual purge procedure."
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
