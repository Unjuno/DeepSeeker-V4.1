#!/usr/bin/env python3
"""Build a canonical DeepSeek-V4.1 tensor manifest and storage map.

Derives the full tensor inventory from the *pinned* upstream revision
without downloading weight payloads: the safetensors index gives the
name->shard map, and each shard's header is fetched with HTTP Range
requests (8-byte length prefix + JSON header only).

    python scripts/build_model_manifest.py

Output (default ``profiles/deepseek-v4.1-flash/<revision>/``):

- ``model-manifest.json`` — versioned manifest (see MANIFEST_SCHEMA)
- ``storage-summary.md`` — human-readable aggregate report

Offset convention (documented, tested): safetensors ``data_offsets``
are relative to the start of the data section. Absolute file offsets
are ``8 + header_len + relative_offset`` and are recorded as
``file_range`` alongside ``data_range``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import struct
import subprocess
import urllib.request
from pathlib import Path

MANIFEST_SCHEMA = "deepseeker.model-manifest/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash"

# Approximate bytes per element for sanity checks only. Encoded byte
# sizes always come from safetensors data_offsets (authoritative).
DTYPE_BYTES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "U8": 1,
    "I8": 1,
    "I32": 4,
    "I64": 8,
    "F64": 8,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def pinned_revision() -> str | None:
    manifest = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("resolved_revision")
    except (OSError, ValueError):
        return None


def deepseeker_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


# --------------------------------------------------------------------------
# Transport: HTTP Range header-only access


class RangeFetcher:
    """Fetch byte ranges; counts every network byte for reporting."""

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self.network_bytes = 0

    def get_range(self, url: str, start: int, end: int) -> tuple[bytes, int]:
        """Return (payload, full_file_size) for inclusive range."""
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if resp.status != 206:
                raise RuntimeError(f"range request not honored for {url}: status {resp.status}")
            content_range = resp.headers.get("Content-Range", "")
            m = re.search(r"/(\d+)$", content_range)
            if not m:
                raise RuntimeError(f"missing Content-Range size for {url}")
            payload = resp.read()
            self.network_bytes += len(payload)
            return payload, int(m.group(1))

    def fetch_shard_header(self, url: str) -> tuple[dict, int, int]:
        """Return (header_json, header_len, file_size)."""
        prefix, file_size = self.get_range(url, 0, 7)
        (header_len,) = struct.unpack("<Q", prefix[:8])
        if header_len <= 0 or header_len > 512 * 1024 * 1024:
            raise RuntimeError(f"implausible header length {header_len} for {url}")
        raw, _ = self.get_range(url, 8, 8 + header_len - 1)
        header = json.loads(raw.decode("utf-8"))
        return header, header_len, file_size


def parse_shard_tensors(header: dict) -> dict[str, dict]:
    """Parse a safetensors header into {name: {dtype, shape, data_range}}.

    ``data_range`` is [start, end) relative to the data-section start.
    """
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape = list(info["shape"])
        start, end = info["data_offsets"]
        tensors[name] = {
            "dtype": info["dtype"],
            "shape": shape,
            "data_range": [start, end],
        }
    return tensors


# --------------------------------------------------------------------------
# Deterministic tensor classification (ordered rules)


def classify(name: str) -> dict:
    """Return {subsystem, layer, expert, routed, detail}.

    Unknown names fall through to subsystem "unknown" and stay counted.
    """
    # hc_* auxiliaries: structural location known, semantics unconfirmed.
    # Kept visible as unknown rather than guessed into a subsystem.
    m = re.match(r"^((?:layers|mtp)\.(\d+)\.)?hc_(.+)$", name)
    if m:
        return {
            "subsystem": "unknown",
            "layer": int(m.group(2)) if m.group(2) is not None else None,
            "expert": None,
            "routed": None,
            "detail": name,
        }
    m = re.match(
        r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$",
        name,
    )
    if m:
        return {
            "subsystem": "routed-expert",
            "layer": int(m.group(1)),
            "expert": int(m.group(2)),
            "routed": True,
            "detail": (f"{m.group(3)}-{m.group(4)}"),
        }
    m = re.match(
        r"^layers\.(\d+)\.ffn\.shared_experts\.(w[123])\.(weight|scale)$",
        name,
    )
    if m:
        return {
            "subsystem": "shared-expert",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": f"{m.group(2)}",
        }
    m = re.match(r"^layers\.(\d+)\.ffn\.gate\.(.+)$", name)
    if m:
        return {
            "subsystem": "router",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": m.group(2),
        }
    m = re.match(r"^layers\.(\d+)\.ffn_norm\.weight$", name)
    if m:
        return {
            "subsystem": "norm",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": "ffn_norm",
        }
    m = re.match(r"^layers\.(\d+)\.attn_norm\.weight$", name)
    if m:
        return {
            "subsystem": "norm",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": "attn_norm",
        }
    m = re.match(r"^layers\.(\d+)\.attn\.(compressor|indexer)\.(.+)$", name)
    if m:
        return {
            "subsystem": "kv-index",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": f"{m.group(2)}.{m.group(3)}",
        }
    m = re.match(r"^layers\.(\d+)\.attn\.(.+)$", name)
    if m:
        return {
            "subsystem": "attention",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": m.group(2),
        }
    m = re.match(r"^layers\.(\d+)\.engram\.(.+)$", name)
    if m:
        return {
            "subsystem": "engram",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": m.group(2),
        }
    m = re.match(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(.+)$", name)
    if m:
        return {
            "subsystem": "mtp-expert",
            "layer": int(m.group(1)),
            "expert": int(m.group(2)),
            "routed": True,
            "detail": f"{m.group(3)}",
        }
    m = re.match(r"^mtp\.(\d+)\.ffn\.shared_experts\.(.+)$", name)
    if m:
        return {
            "subsystem": "mtp-shared",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": m.group(2),
        }
    m = re.match(r"^mtp\.(\d+)\.ffn\.gate\.(.+)$", name)
    if m:
        return {
            "subsystem": "mtp-router",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": m.group(2),
        }
    m = re.match(r"^mtp\.(\d+)\.(.+)$", name)
    if m:
        return {
            "subsystem": "mtp",
            "layer": int(m.group(1)),
            "expert": None,
            "routed": False,
            "detail": m.group(2),
        }
    if name == "embed.weight":
        return {
            "subsystem": "embedding",
            "layer": None,
            "expert": None,
            "routed": False,
            "detail": "embed",
        }
    if name == "head.weight":
        return {
            "subsystem": "lm-head",
            "layer": None,
            "expert": None,
            "routed": False,
            "detail": "head",
        }
    if name == "norm.weight":
        return {
            "subsystem": "norm",
            "layer": None,
            "expert": None,
            "routed": False,
            "detail": "final-norm",
        }
    if name in ("image_start", "image_end", "image_newline"):
        return {
            "subsystem": "embedding",
            "layer": None,
            "expert": None,
            "routed": False,
            "detail": f"special-token:{name}",
        }
    if name.startswith(("vision.", "aligner.")):
        return {
            "subsystem": "vision",
            "layer": None,
            "expert": None,
            "routed": False,
            "detail": name,
        }
    return {
        "subsystem": "unknown",
        "layer": None,
        "expert": None,
        "routed": None,
        "detail": name,
    }


# Residency rule (documented): only routed-expert tensors (backbone and
# MTP) are potentially streamable per token; everything else is treated
# as always-needed for planning. Vision/Engram conditionality is noted
# in the summary rather than encoded here.
STREAMABLE_SUBSYSTEMS = {"routed-expert", "mtp-expert"}


# --------------------------------------------------------------------------
# Manifest construction


def build_manifest(
    revision: str,
    weight_map: dict[str, str],
    shard_headers: dict[str, tuple[dict, int, int]],
    architecture: dict,
    network_bytes: int,
    index_bytes: int,
) -> dict:
    tensors = []
    shard_info: dict[str, dict] = {}
    for shard in sorted(shard_headers):
        header, header_len, file_size = shard_headers[shard]
        parsed = parse_shard_tensors(header)
        tensor_bytes = 0
        for name in sorted(parsed):
            info = parsed[name]
            start, end = info["data_range"]
            n_bytes = end - start
            tensor_bytes += n_bytes
            elements = 1
            for dim in info["shape"]:
                elements *= dim
            cls = classify(name)
            tensors.append(
                {
                    "name": name,
                    "shape": info["shape"],
                    "dtype": info["dtype"],
                    "elements": elements,
                    "bytes": n_bytes,
                    "shard": shard,
                    "data_range": [start, end],
                    "file_range": [
                        8 + header_len + start,
                        8 + header_len + end,
                    ],
                    **cls,
                }
            )
        shard_info[shard] = {
            "file_bytes": file_size,
            "header_bytes": 8 + header_len,
            "tensor_count": len(parsed),
            "tensor_bytes": tensor_bytes,
            "container_overhead_bytes": file_size - 8 - header_len - tensor_bytes,
        }

    # Cross-check: every weight_map entry must exist in its shard header.
    by_shard: dict[str, set[str]] = {}
    for t in tensors:
        by_shard.setdefault(t["shard"], set()).add(t["name"])
    missing = [n for n, s in sorted(weight_map.items()) if n not in by_shard.get(s, set())]
    extra = sorted({t["name"] for t in tensors} - set(weight_map.keys()))

    by_subsystem: dict[str, dict] = {}
    for t in tensors:
        bucket = by_subsystem.setdefault(t["subsystem"], {"bytes": 0, "tensors": 0})
        bucket["bytes"] += t["bytes"]
        bucket["tensors"] += 1

    per_layer: dict[str, dict] = {}
    for t in tensors:
        if t["layer"] is None or t["subsystem"] not in (
            "routed-expert",
            "shared-expert",
            "router",
            "attention",
            "kv-index",
            "norm",
            "engram",
        ):
            continue
        key = str(t["layer"])
        bucket = per_layer.setdefault(key, {"bytes": 0, "tensors": 0, "routed_experts": 0})
        bucket["bytes"] += t["bytes"]
        bucket["tensors"] += 1

    expert_bytes: dict[str, int] = {}
    for t in tensors:
        if t["subsystem"] == "routed-expert":
            key = f"{t['layer']}/{t['expert']}"
            expert_bytes[key] = expert_bytes.get(key, 0) + t["bytes"]
    for key, layer_info in per_layer.items():
        layer_experts = {e for k in expert_bytes for (layer, e) in [k.split("/")] if layer == key}
        layer_info["routed_experts"] = len(layer_experts)

    expert_sizes = sorted(expert_bytes.values())
    total_bytes = sum(t["bytes"] for t in tensors)
    streamable_bytes = sum(t["bytes"] for t in tensors if t["subsystem"] in STREAMABLE_SUBSYSTEMS)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "upstream": {
            "repo_id": REPO_ID,
            "revision": revision,
        },
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "deepseeker_commit": deepseeker_commit(),
        "architecture": architecture,
        "offset_convention": (
            "data_range is [start, end) relative to the safetensors "
            "data-section start; file_range is absolute from file byte 0 "
            "(8-byte header length + JSON header + relative offset)."
        ),
        "residency_rule": (
            "streamable = routed-expert and mtp-expert tensors only; "
            "everything else is always-needed for planning. "
            "Vision/Engram conditionality is a planning concern, "
            "not encoded here."
        ),
        "transport": {
            "method": "HTTP Range header-only fetch (no weight download)",
            "shard_count": len(shard_headers),
            "network_bytes": network_bytes,
            "index_bytes": index_bytes,
        },
        "totals": {
            "tensors": len(tensors),
            "tensor_bytes": total_bytes,
            "streamable_bytes": streamable_bytes,
            "always_needed_bytes": total_bytes - streamable_bytes,
            "by_subsystem": by_subsystem,
        },
        "expert_stats": {
            "experts": len(expert_bytes),
            "min_bytes": expert_sizes[0] if expert_sizes else 0,
            "median_bytes": expert_sizes[len(expert_sizes) // 2] if expert_sizes else 0,
            "max_bytes": expert_sizes[-1] if expert_sizes else 0,
        },
        "per_layer": per_layer,
        "per_expert_bytes": expert_bytes,
        "validation": {
            "weight_map_entries": len(weight_map),
            "header_tensor_entries": len(tensors),
            "missing_from_headers": missing,
            "extra_in_headers": extra,
            "accounting_reconciled": (
                sum(b["bytes"] for b in by_subsystem.values()) == total_bytes
            ),
        },
        "shards": shard_info,
        "tensors": tensors,
    }
    return manifest


def summarize(manifest: dict) -> str:
    totals = manifest["totals"]
    by_sub = totals["by_subsystem"]

    def gib(n: int) -> str:
        return f"{n / 1024**3:.2f} GiB"

    lines = [
        "# Storage summary — DeepSeek-V4.1-Flash",
        "",
        f"Revision: `{manifest['upstream']['revision']}`",
        f"Schema: `{manifest['schema']}`",
        f"Generated: {manifest['generated_at_utc']} (DeepSeeker {manifest['deepseeker_commit']})",
        "",
        "## Totals",
        "",
        f"- Tensors: {totals['tensors']}",
        f"- Tensor bytes: {totals['tensor_bytes']} ({gib(totals['tensor_bytes'])})",
        (
            f"- Streamable (routed experts): "
            f"{totals['streamable_bytes']} "
            f"({gib(totals['streamable_bytes'])})"
        ),
        (
            f"- Always-needed: {totals['always_needed_bytes']} "
            f"({gib(totals['always_needed_bytes'])})"
        ),
        "",
        "## By subsystem",
        "",
    ]
    for sub in sorted(by_sub):
        b = by_sub[sub]
        lines.append(f"- {sub}: {b['tensors']} tensors, {b['bytes']} bytes ({gib(b['bytes'])})")
    ex = manifest["expert_stats"]
    lines += [
        "",
        "## Routed experts",
        "",
        f"- Experts: {ex['experts']}",
        f"- Min/median/max bytes: {ex['min_bytes']} / {ex['median_bytes']} / {ex['max_bytes']}",
        "",
        "## Validation",
        "",
        f"- weight_map entries: {manifest['validation']['weight_map_entries']}",
        f"- header entries: {manifest['validation']['header_tensor_entries']}",
        f"- missing from headers: {len(manifest['validation']['missing_from_headers'])}",
        f"- extra in headers: {len(manifest['validation']['extra_in_headers'])}",
        f"- accounting reconciled: {manifest['validation']['accounting_reconciled']}",
        "",
        "## Transport",
        "",
        f"- Shards: {manifest['transport']['shard_count']}",
        (f"- Network bytes (headers+index only): {manifest['transport']['network_bytes']}"),
        "- Weights downloaded: none",
        "",
        "## Offset convention",
        "",
        manifest["offset_convention"],
        "",
    ]
    unknown = by_sub.get("unknown", {"tensors": 0, "bytes": 0})
    lines += [
        "## Unknown tensors",
        "",
        (
            f"- {unknown['tensors']} tensors, {unknown['bytes']} bytes. "
            "Enumerated in the manifest with full names; none dropped."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the canonical DeepSeek-V4.1 tensor manifest."
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Upstream revision (default: pinned revision).",
    )
    parser.add_argument(
        "--allow-other-revision",
        action="store_true",
        help="Permit a non-pinned revision (recorded explicitly).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory override.",
    )
    args = parser.parse_args()

    pinned = pinned_revision()
    revision = args.revision or pinned
    if revision is None:
        print("ERROR: no pinned revision found; run fetch_upstream first.")
        return 1
    if pinned is not None and revision != pinned:
        if not args.allow_other_revision:
            print(
                f"ERROR: requested {revision} != pinned {pinned}; "
                "pass --allow-other-revision to override."
            )
            return 1
        print(f"WARNING: using non-pinned revision {revision}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        return 1

    api = HfApi()
    index_path = api.hf_hub_download(REPO_ID, "model.safetensors.index.json", revision=revision)
    index_bytes = Path(index_path).stat().st_size
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    print(f"index: {len(weight_map)} tensors across shards")

    shards = sorted(set(weight_map.values()))
    fetcher = RangeFetcher()
    shard_headers = {}
    for i, shard in enumerate(shards, 1):
        url = f"https://huggingface.co/{REPO_ID}/resolve/{revision}/{shard}"
        header, header_len, file_size = fetcher.fetch_shard_header(url)
        shard_headers[shard] = (header, header_len, file_size)
        print(f"[{i}/{len(shards)}] {shard}: {len(header)} entries")

    # Architecture from the pinned local config (extracted, not hardcoded
    # model constants: every value below is read from config.json).
    config_path = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / ("config.json")
    architecture: dict = {"source": "upstream config.json"}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", {})
        architecture.update(
            {
                "dtype": cfg.get("dtype"),
                "hidden_size": text_cfg.get("hidden_size"),
                "num_hidden_layers": text_cfg.get("num_hidden_layers"),
                "n_routed_experts": text_cfg.get("n_routed_experts"),
                "num_experts_per_tok": text_cfg.get("num_experts_per_tok"),
                "n_shared_experts": text_cfg.get("n_shared_experts"),
                "moe_intermediate_size": text_cfg.get("moe_intermediate_size"),
                "num_nextn_predict_layers": text_cfg.get("num_nextn_predict_layers"),
                "quantization": text_cfg.get("quantization_config")
                or cfg.get("quantization_config"),
            }
        )
    else:
        architecture["warning"] = "local config.json not fetched"

    manifest = build_manifest(
        revision,
        weight_map,
        shard_headers,
        architecture,
        fetcher.network_bytes,
        index_bytes,
    )

    out = args.out_dir or (REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model-manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    (out / "storage-summary.md").write_text(summarize(manifest), encoding="utf-8")
    print(
        f"manifest: {manifest['totals']['tensors']} tensors, "
        f"{manifest['totals']['tensor_bytes']} bytes"
    )
    print(f"network bytes (headers+index): {fetcher.network_bytes}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
