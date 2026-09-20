"""Issue #15: KV/index cache memory model + unified budget (roadmap P5 input).

Element-exact cache geometry read off the reference runtime
(`upstream/.../inference/model.py`), replacing the HIGH-uncertainty
arch-estimates in the Issue #5 resident plan:

  window    every backbone layer: [B, window_size, head_dim] fp8 (1 B)
  compress  kv_source layers only: [B, S//ratio, head_dim] fp4 (0.5 B)
  index     kv_source layers only: [B, S//ratio, index_head_dim] fp4 (0.5 B)

Quant scales (+~3-6%) and MTP-layer state are excluded and noted, not
hidden. All sizes are resident-cache bytes for batch B, context S.
"""

from __future__ import annotations

SCHEMA = "deepseeker.kvmodel/v1"
FP8_BYTES = 1.0
FP4_BYTES = 0.5


def cache_geometry(config: dict) -> dict:
    """Element counts per (cache, layer) class from the runtime config."""
    n_layers = config["n_layers"]
    ratios = config["compress_ratios"][:n_layers]
    kv_sources = [layer for layer in config["kv_source_layers"] if layer < n_layers]
    return {
        "n_layers": n_layers,
        "window_size": config["window_size"],
        "head_dim": config["head_dim"],
        "index_head_dim": config["index_head_dim"],
        "kv_source_layers": kv_sources,
        "compress_ratios": {layer: ratios[layer] for layer in kv_sources},
        "window_bytes_per_elem": FP8_BYTES,
        "compress_bytes_per_elem": FP4_BYTES,
        "index_bytes_per_elem": FP4_BYTES,
    }


def cache_bytes(geometry: dict, batch: int, seq_len: int) -> dict:
    """Resident cache bytes split by cache class."""
    if batch < 1 or seq_len < 1:
        raise ValueError(f"batch/seq_len must be >= 1: {batch}, {seq_len}")
    head = geometry["head_dim"]
    window = (
        geometry["n_layers"] * batch * geometry["window_size"] * head
    ) * geometry["window_bytes_per_elem"]
    compress = sum(
        batch * (seq_len // geometry["compress_ratios"][layer]) * head
        for layer in geometry["kv_source_layers"]
    ) * geometry["compress_bytes_per_elem"]
    index = sum(
        batch * (seq_len // geometry["compress_ratios"][layer]) * geometry["index_head_dim"]
        for layer in geometry["kv_source_layers"]
    ) * geometry["index_bytes_per_elem"]
    total = window + compress + index
    return {
        "window_bytes": int(window),
        "compress_bytes": int(compress),
        "index_bytes": int(index),
        "total_bytes": int(total),
        "per_token_bytes": total / seq_len,
    }


def unified_table(
    common_model_bytes: int,
    dense_engram_bytes: int,
    expert_slots_per_layer: int,
    n_layers: int,
    expert_bytes: int,
    kv: dict,
    operating_bytes: int,
    engram_cache_bytes: int = 0,
    runtime_fixed_bytes: int = 1 * 1024**3,
) -> dict:
    """One P5 budget row: every resident against the operating ceiling."""
    expert_pool = expert_slots_per_layer * n_layers * expert_bytes
    components = {
        "common-model": common_model_bytes,
        "dense-engram": dense_engram_bytes,
        "expert-pool": expert_pool,
        "kv-cache": kv["total_bytes"],
        "engram-cache": engram_cache_bytes,
        "runtime-fixed": runtime_fixed_bytes,
    }
    total = sum(components.values())
    return {
        "schema": SCHEMA,
        "components": components,
        "total_bytes": total,
        "operating_bytes": operating_bytes,
        "headroom_bytes": operating_bytes - total,
        "over_budget": total > operating_bytes,
    }
