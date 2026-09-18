#!/usr/bin/env python3
"""Engram row-level access analysis (no model weights required).

Combines pinned upstream Engram hash semantics, the official tokenizer,
the Issue #2 tensor manifest, and working-set/cache simulation:

    python scripts/analyze_engram_access.py

Output (default ``profiles/deepseek-v4.1-flash/<revision>/``):

- ``engram-access.json`` — machine-readable geometry, traces stats,
  locality, cache sweeps, coalescing
- ``engram-access-summary.md`` — human-readable report

Only metadata is downloaded (tokenizer files, safetensors headers are
already in the manifest). No weight payload bytes are fetched.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.engram import (
    EngramGeometry,
    build_multipliers,
    build_offsets,
    build_primes,
    coalesce,
    hash_positions,
    lru_hit_rate,
    page_id,
    reuse_distances,
    row_ranges,
    table_refs_from_manifest,
)

REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
CACHE_BUDGETS = [16, 64, 256, 1024, 4096, 8192]  # MiB
WINDOWS = [8, 32, 128, 512, 2048]
PAGE_SIZES = [4096, 16384]
COALESCE_GAPS = [0, 4096, 65536]

ROW_PAIR_BUDGET_DIVISOR = 264  # recomputed from manifest, asserted


# --------------------------------------------------------------------------
# Deterministic multi-domain corpus (original synthetic text)


def build_corpus() -> dict[str, str]:
    en_para = (
        "The harbor lights flickered across the evening water while gulls "
        "circled the returning fishing boats. Mara counted the masts from "
        "the stone pier and wondered whether the tide would turn before "
        "midnight. Somewhere inland a train whistle faded into the hills."
    )
    ja_para = (
        "夕暮れの港では帰ってきた漁船の上でカモメが鳴いていた。"
        "石の埠頭からマストの数を数えながら、潮が変わる時刻を気にしていた。"
        "遠くでは列車の汽笛が山の方へ消えていった。"
        "波の音だけが規則正しく繰り返していた。"
    )
    code_para = (
        "def rolling_median(values, window):\n"
        "    result = []\n"
        "    for i in range(len(values)):\n"
        "        start = max(0, i - window + 1)\n"
        "        chunk = sorted(values[start:i + 1])\n"
        "        mid = len(chunk) // 2\n"
        "        if len(chunk) % 2:\n"
        "            result.append(chunk[mid])\n"
        "        else:\n"
        "            result.append((chunk[mid - 1] + chunk[mid]) / 2)\n"
        "    return result\n"
    )
    return {
        "english_prose": " ".join([en_para] * 4),
        "japanese_prose": " ".join([ja_para] * 4),
        "source_code": " ".join([code_para] * 3),
        "repetitive": " ".join(["lorem ipsum dolor sit amet"] * 40),
        "random_tokens": "__RANDOM__",  # filled from seeded vocab ids
        "short_prompt": "Hello, world!",
        "long_prompt": " ".join([en_para, ja_para, code_para] * 12),
    }


# --------------------------------------------------------------------------
# Loading helpers


def load_tokenizer(cache_dir: Path) -> tuple[object, int]:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    manifest_path = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    revision = json.loads(manifest_path.read_text(encoding="utf-8"))["resolved_revision"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    network_bytes = 0
    for name in ("tokenizer.json", "tokenizer_config.json"):
        dest = cache_dir / name
        if not dest.exists():
            src = hf_hub_download(REPO_ID, name, revision=revision)
            dest.write_bytes(Path(src).read_bytes())
        network_bytes += dest.stat().st_size
    return Tokenizer.from_file(str(cache_dir / "tokenizer.json")), (network_bytes)


def load_manifest() -> tuple[dict, str]:
    manifest_path = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    revision = json.loads(manifest_path.read_text(encoding="utf-8"))["resolved_revision"]
    path = REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    return json.loads(path.read_text(encoding="utf-8")), revision


def upstream_config() -> dict:
    path = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))["text_config"]


# --------------------------------------------------------------------------
# Analysis


def analyze_trace(
    row_ids: list[list[list[int]]],
    refs: dict,
    pair_bytes: int,
) -> dict:
    """row_ids: [position][layer][24 cols]. All stats derived here."""
    n_tokens = len(row_ids)
    n_layers = len(row_ids[0]) if n_tokens else 0
    flat: list[tuple[int, int]] = []  # (layer, row)
    per_token_uniques = []
    for pos in row_ids:
        seen = set()
        for li, cols in enumerate(pos):
            for row in cols:
                flat.append((li, row))
                seen.add((li, row))
        per_token_uniques.append(len(seen))
    requested = len(flat)
    unique_rows = len(set(flat))
    dupes = requested - sum(per_token_uniques)

    reuse = reuse_distances(flat)
    gaps = sorted(d for d in reuse if d is not None)
    reuse_stats = {
        "first_touch": sum(1 for d in reuse if d is None),
        "median_gap": gaps[len(gaps) // 2] if gaps else None,
        "max_gap": gaps[-1] if gaps else None,
    }

    overlap = []
    for a, b in itertools.pairwise(row_ids):
        sa = {(li, r) for li, cols in enumerate(a) for r in cols}
        sb = {(li, r) for li, cols in enumerate(b) for r in cols}
        overlap.append(len(sa & sb))
    overlap_stats = {
        "mean": sum(overlap) / len(overlap) if overlap else 0.0,
        "min": min(overlap) if overlap else 0,
        "max": max(overlap) if overlap else 0,
    }

    windows = {}
    for w in WINDOWS:
        prefix = flat[: w * n_layers * 24] if n_tokens else []
        windows[str(w)] = len({r for r in prefix})

    # Row-keyed LRU sweep (key = (layer, row)).
    row_sweep = {}
    for budget_mib in CACHE_BUDGETS:
        capacity = (budget_mib * 1024 * 1024) // pair_bytes
        row_sweep[str(budget_mib)] = round(lru_hit_rate(flat, capacity), 4)

    # Byte ranges for amplification/coalescing/page-cache sims.
    ranges: list[tuple[int, int, str]] = []  # (abs, abs, shard)
    layer_of = sorted(refs)
    for li, row in flat:
        ref = refs[layer_of[li]]
        rr = row_ranges(ref, row)
        for part in ("weight", "scale"):
            s, e = rr[part]["range"]
            ranges.append((s, e, rr[part]["shard"]))

    useful = requested * pair_bytes
    amplification = {}
    for page in PAGE_SIZES:
        pages = set()
        for s, e, shard in ranges:
            pages.update((shard, p) for p in range(page_id(s, page), page_id(e - 1, page) + 1))
        amplification[str(page)] = {
            "pages": len(pages),
            "bytes": len(pages) * page,
            "ratio": round(len(pages) * page / useful, 3) if useful else 0.0,
        }

    page_sweep = {}
    for page in PAGE_SIZES:
        keyed = [(shard, page_id(s, page)) for s, e, shard in ranges]
        per_budget = {}
        for budget_mib in CACHE_BUDGETS:
            capacity = (budget_mib * 1024 * 1024) // page
            per_budget[str(budget_mib)] = round(lru_hit_rate(keyed, capacity), 4)
        page_sweep[str(page)] = per_budget

    coalesced = {}
    by_shard: dict[str, list[tuple[int, int]]] = {}
    for s, e, shard in ranges:
        by_shard.setdefault(shard, []).append((s, e))
    for gap in COALESCE_GAPS:
        ops, merged_bytes = 0, 0
        for shard_ranges in by_shard.values():
            for s, e in coalesce(shard_ranges, gap):
                ops += 1
                merged_bytes += e - s
        coalesced[str(gap)] = {
            "read_ops": ops,
            "bytes": merged_bytes,
            "ops_per_token": round(ops / n_tokens, 2) if n_tokens else 0,
            "bytes_per_token": round(merged_bytes / n_tokens, 1) if n_tokens else 0,
        }

    return {
        "tokens": n_tokens,
        "requested_row_pairs": requested,
        "unique_row_pairs": unique_rows,
        "requested_per_token": requested / n_tokens if n_tokens else 0,
        "unique_per_token": round(sum(per_token_uniques) / n_tokens, 2) if n_tokens else 0,
        "duplicate_rows_within_token_total": dupes,
        "reuse": reuse_stats,
        "consecutive_overlap": overlap_stats,
        "window_unique_prefix": windows,
        "row_lru_hit_rate_by_mib": row_sweep,
        "page_lru_hit_rate_by_mib": page_sweep,
        "useful_bytes_total": useful,
        "useful_bytes_per_token": useful / n_tokens if n_tokens else 0,
        "amplification": amplification,
        "coalesced": coalesced,
    }


# --------------------------------------------------------------------------
# Report


def summarize(analysis: dict) -> str:
    geo = analysis["geometry"]
    lines = [
        "# Engram access summary",
        "",
        f"Revision: `{analysis['upstream_revision']}`",
        f"Generated: {analysis['generated_at_utc']}",
        "",
        "## Verified geometry (derived from pinned config + manifest)",
        "",
        f"- Engram layers: {geo['layer_ids']} ({geo['n_layers']} layers)",
        f"- Hash cols per layer/token: {geo['hash_cols']} (2/3/4-grams x {geo['n_heads']} heads)",
        (
            f"- Row pairs per token: {geo['row_pairs_per_token']} "
            f"({geo['n_layers']} layers x {geo['hash_cols']})"
        ),
        (
            f"- Weight row: {geo['weight_row_bytes']} B, "
            f"scale row: {geo['scale_row_bytes']} B "
            f"(pair {geo['pair_bytes']} B, from manifest shapes)"
        ),
        f"- Useful table bytes/token: {geo['useful_bytes_per_token']} B",
        (
            f"- Dense per-layer weights (wkv/q/k, resident candidates): "
            f"{geo['dense_bytes_per_layer']} B"
        ),
        f"- Weight and scale share a shard per layer: {geo['weight_scale_same_shard']}",
        "",
        "## Per-corpus results",
        "",
    ]
    for name, res in analysis["corpora"].items():
        amp4 = res["amplification"]["4096"]
        amp16 = res["amplification"]["16384"]
        coal = res["coalesced"]["4096"]
        lines += [
            f"### {name} ({res['tokens']} tokens)",
            "",
            f"- unique row pairs: {res['unique_row_pairs']} ({res['unique_per_token']}/token)",
            f"- consecutive overlap mean: {res['consecutive_overlap']['mean']:.1f} rows",
            (
                f"- 4 KiB: {amp4['bytes'] / res['tokens']:.0f} B/token "
                f"(x{amp4['ratio']}), 16 KiB: "
                f"{amp16['bytes'] / res['tokens']:.0f} B/token "
                f"(x{amp16['ratio']})"
            ),
            (
                f"- row-LRU 64 MiB hit rate: "
                f"{res['row_lru_hit_rate_by_mib']['64']}, "
                f"1 GiB: {res['row_lru_hit_rate_by_mib']['1024']}"
            ),
            (
                f"- coalesced (4 KiB gap): {coal['ops_per_token']} "
                f"ops/token, {coal['bytes_per_token']:.0f} B/token"
            ),
            "",
        ]
    lines += [
        "## Conclusion",
        "",
        analysis["conclusion"],
        "",
        "## Transport",
        "",
        f"- Tokenizer bytes downloaded: {analysis['tokenizer_bytes']}",
        "- Weight payload bytes downloaded: 0",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Engram row-level access analysis.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest, revision = load_manifest()
    cfg = upstream_config()
    assert cfg["engram_layer_ids"] == [1, 14]
    assert cfg["engram_max_ngram_size"] == 4
    assert cfg["engram_n_heads"] == 8
    assert cfg["engram_head_dim"] == 256

    refs = table_refs_from_manifest(manifest)
    layer_ids = sorted(refs)
    assert layer_ids == cfg["engram_layer_ids"], layer_ids
    pair_bytes = None
    dense_per_layer = {}
    for layer in layer_ids:
        ref = refs[layer]
        assert ref.weight_row_bytes == cfg["engram_head_dim"]
        assert ref.scale_row_bytes == cfg["engram_head_dim"] // 32
        pair_bytes = ref.weight_row_bytes + ref.scale_row_bytes
        dense = sum(
            t["bytes"]
            for t in manifest["tensors"]
            if t["subsystem"] == "engram" and t["layer"] == layer and ".embed." not in t["name"]
        )
        dense_per_layer[str(layer)] = dense
    assert pair_bytes == ROW_PAIR_BUDGET_DIVISOR, pair_bytes

    cache_dir = args.cache_dir or (REPO_ROOT / ".cache" / "engram-analysis")
    tokenizer, tokenizer_bytes = load_tokenizer(cache_dir)

    sys.path.insert(0, str(REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "inference"))
    from engram import build_compressed_token_map
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(
        str(cache_dir), local_files_only=True, trust_remote_code=False
    )
    token_map, compressed_vocab = build_compressed_token_map(hf_tok)
    assert compressed_vocab == cfg["engram_compressed_vocab_size"]
    assert cfg["engram_pad_token_id"] == 2  # -> ModelArgs.engram_pad_id
    pad_id = token_map[cfg["engram_pad_token_id"]]

    geometry = EngramGeometry(
        layer_ids=tuple(cfg["engram_layer_ids"]),
        max_ngram_size=cfg["engram_max_ngram_size"],
        n_heads=cfg["engram_n_heads"],
        head_dim=cfg["engram_head_dim"],
        num_embeddings=tuple(cfg["engram_num_embeddings"]),
        vocab_size=cfg["engram_vocab_size"],
        compressed_vocab_size=compressed_vocab,
        pad_id=pad_id,
    )
    primes = build_primes(geometry)
    for li, layer_primes in enumerate(primes):
        total = sum(p for ng in layer_primes for p in ng)
        assert total == geometry.num_embeddings[li], (li, total)
    offsets = build_offsets(primes)
    multipliers = build_multipliers(
        geometry.layer_ids,
        geometry.max_ngram_size,
        compressed_vocab,
    )

    corpus = build_corpus()
    corpora = {}
    for name, text in corpus.items():
        if text == "__RANDOM__":
            rng = np.random.default_rng(0)
            ids = [int(x) for x in rng.integers(0, len(tokenizer.get_vocab()), size=256)]
        else:
            ids = tokenizer.encode(text).ids
        compressed = [token_map[i] for i in ids]
        row_ids = hash_positions(compressed, geometry, multipliers, primes, offsets)
        corpora[name] = {
            "tokens": len(ids),
            **analyze_trace(row_ids, refs, pair_bytes),
        }

    useful_per_token = len(geometry.layer_ids) * geometry.n_hash_cols * pair_bytes
    analysis = {
        "schema": "deepseeker.engram-access/v1",
        "upstream_revision": revision,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "geometry": {
            "layer_ids": list(geometry.layer_ids),
            "n_layers": len(geometry.layer_ids),
            "n_heads": geometry.n_heads,
            "hash_cols": geometry.n_hash_cols,
            "row_pairs_per_token": len(geometry.layer_ids) * geometry.n_hash_cols,
            "weight_row_bytes": refs[layer_ids[0]].weight_row_bytes,
            "scale_row_bytes": refs[layer_ids[0]].scale_row_bytes,
            "pair_bytes": pair_bytes,
            "useful_bytes_per_token": useful_per_token,
            "dense_bytes_per_layer": dense_per_layer,
            "weight_scale_same_shard": all(r.weight_shard == r.scale_shard for r in refs.values()),
        },
        "corpora": corpora,
        "tokenizer_bytes": tokenizer_bytes,
        "weight_bytes_downloaded": 0,
        "conclusion": (
            f"Useful Engram table payload is {useful_per_token} B/token "
            f"({len(geometry.layer_ids)} layers x "
            f"{geometry.n_hash_cols} rows x {pair_bytes} B), versus "
            "189.13 GiB of tables. Large tables are conditional sparse "
            "memory: row-stream with a modest row/page cache, keep the "
            "~315 MB dense per-layer weights resident. Physical SSD "
            "latency remains unmeasured (simulation only)."
        ),
    }

    out = args.out_dir or (REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision)
    out.mkdir(parents=True, exist_ok=True)
    (out / "engram-access.json").write_text(json.dumps(analysis, indent=1) + "\n", encoding="utf-8")
    (out / "engram-access-summary.md").write_text(summarize(analysis), encoding="utf-8")
    print(f"corpora: {len(corpora)}, useful/token: {useful_per_token} B")
    print(f"tokenizer bytes: {tokenizer_bytes}, weight bytes: 0")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
