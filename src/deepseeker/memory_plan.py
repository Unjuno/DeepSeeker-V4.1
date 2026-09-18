"""Reusable resident-memory budget model (pure functions, no I/O).

All byte math reconciles to the Issue #2 manifest plus explicitly
marked estimates. Provenance travels with every component:
measured / manifest / arch-estimate / assumed.
"""

from __future__ import annotations

EXPERT_SLOT_BYTES = 18800640
N_BACKBONE_EXPERTS = 15360
N_BACKBONE_LAYERS = 40

# Manifest subsystems treated as common resident state (S1 baseline).
COMMON_SUBSYSTEMS = frozenset(
    {
        "attention",
        "embedding",
        "lm-head",
        "shared-expert",
        "router",
        "norm",
        "kv-index",
        "unknown",  # hc_* auxiliaries: small, per-layer, needed each step
    }
)
MTP_STATIC_SUBSYSTEMS = frozenset({"mtp", "mtp-router", "mtp-shared"})


def manifest_by_subsystem(manifest: dict) -> dict[str, dict]:
    totals: dict[str, dict] = {}
    for t in manifest["tensors"]:
        bucket = totals.setdefault(t["subsystem"], {"bytes": 0, "tensors": 0})
        bucket["bytes"] += t["bytes"]
        bucket["tensors"] += 1
    return totals


def common_bytes(manifest: dict, include_mtp_static: bool, include_vision: bool) -> dict:
    """Common resident candidate bytes + per-subsystem breakdown."""
    by_sub = manifest_by_subsystem(manifest)
    chosen = set(COMMON_SUBSYSTEMS)
    if include_mtp_static:
        chosen |= MTP_STATIC_SUBSYSTEMS
    if include_vision:
        chosen.add("vision")
    parts = {s: by_sub.get(s, {"bytes": 0})["bytes"] for s in sorted(chosen)}
    return {"bytes": sum(parts.values()), "parts": parts}


def dense_engram_bytes(manifest: dict) -> dict:
    """Non-lookup Engram weights (wkv/q/k): resident candidates."""
    total = sum(
        t["bytes"]
        for t in manifest["tensors"]
        if t["subsystem"] == "engram" and ".embed." not in t["name"]
    )
    return {"bytes": total, "provenance": "manifest"}


def lookup_table_bytes(manifest: dict) -> int:
    return sum(
        t["bytes"]
        for t in manifest["tensors"]
        if t["subsystem"] == "engram" and ".embed." in t["name"]
    )


def kv_per_token_estimate(manifest: dict, cfg: dict) -> dict:
    """MLA latent cache estimate with documented formula.

    kv_groups share one latent cache of (kv_latent + rope) BF16 values.
    kv_latent is read from the manifest wkv shape, not hardcoded.
    """
    latent = None
    for t in manifest["tensors"]:
        if t["name"] == "layers.0.attn.wkv.weight":
            latent = t["shape"][0]
    groups = len(cfg.get("kv_source_layer_ids", [2, 8, 14, 20]))
    rope = cfg.get("qk_rope_head_dim", 64)
    per_token = groups * (latent + rope) * 2
    return {
        "bytes_per_token": per_token,
        "formula": f"{groups} groups x ({latent}+{rope}) x 2B",
        "provenance": "arch-estimate",
    }


def index_per_token_estimate(cfg: dict) -> dict:
    """Sparse-index state: rough estimate, HIGH uncertainty."""
    layers = len(cfg.get("index_source_layer_ids", [])) or 8
    heads = cfg.get("index_n_heads", 32)
    dim = cfg.get("index_head_dim", 128)
    per_token = layers * heads * dim * 2
    return {
        "bytes_per_token": per_token,
        "formula": f"{layers} layers x {heads} x {dim} x 2B (rough)",
        "provenance": "arch-estimate",
        "uncertainty": "high",
    }


def build_scenario(
    name: str,
    operating_bytes: int,
    manifest: dict,
    cfg: dict,
    context_tokens: int,
    features: dict,
    engram_cache_mib: int,
    staging_experts: int,
    kv_multiplier: float = 1.5,
    runtime_fixed_bytes: int = 1024**3,
) -> dict:
    """Build one fully-reconciled memory plan."""
    common = common_bytes(
        manifest,
        include_mtp_static=features.get("mtp", False),
        include_vision=features.get("vision", False),
    )
    dense_engram = dense_engram_bytes(manifest)
    kv = kv_per_token_estimate(manifest, cfg)
    index = index_per_token_estimate(cfg)
    kv_bytes = int(kv["bytes_per_token"] * context_tokens * kv_multiplier)
    index_bytes = int(index["bytes_per_token"] * context_tokens * 2.0)
    token_ids = context_tokens * 8
    engram_cache = engram_cache_mib * 1024 * 1024
    staging = staging_experts * EXPERT_SLOT_BYTES
    mtp_expert_pool = 0
    if features.get("mtp", False):
        # MTP experts stream like backbone experts; reserve 1 slot/layer.
        mtp_expert_pool = 3 * EXPERT_SLOT_BYTES

    fixed = (
        common["bytes"]
        + dense_engram["bytes"]
        + kv_bytes
        + index_bytes
        + token_ids
        + runtime_fixed_bytes
        + engram_cache
        + staging
        + mtp_expert_pool
    )
    over_budget = fixed > operating_bytes
    pool_bytes = operating_bytes - fixed
    slots = pool_bytes // EXPERT_SLOT_BYTES if pool_bytes > 0 else 0
    used_pool = slots * EXPERT_SLOT_BYTES
    slack = operating_bytes - (fixed + used_pool)

    components = [
        {"name": "common-model", **_prov(common["bytes"], "manifest")},
        {"name": "dense-engram", **_prov(dense_engram["bytes"], "manifest")},
        {
            "name": "kv-cache",
            **_prov(kv_bytes, "arch-estimate", f"{kv['formula']} x ctx x {kv_multiplier}"),
        },
        {
            "name": "index-state",
            **_prov(
                index_bytes, "arch-estimate", f"{index['formula']} x ctx x 2.0, HIGH uncertainty"
            ),
        },
        {"name": "token-ids", **_prov(token_ids, "arch-estimate", "8B/token")},
        {
            "name": "runtime-fixed",
            **_prov(runtime_fixed_bytes, "assumed", "activations/scratch/allocator headroom"),
        },
        {
            "name": "engram-cache",
            **_prov(engram_cache, "measured-config", f"{engram_cache_mib} MiB cache option"),
        },
        {
            "name": "staging",
            **_prov(staging, "manifest", f"{staging_experts} expert slots in flight"),
        },
        {
            "name": "mtp-expert-reserve",
            **_prov(mtp_expert_pool, "manifest", "1 slot x 3 MTP layers"),
        },
        {"name": "expert-pool", **_prov(used_pool, "manifest", f"{slots} whole backbone slots")},
        {"name": "slack", **_prov(slack, "derived", "unallocated")},
    ]
    total = sum(c["bytes"] for c in components)
    assert total == operating_bytes, (total, operating_bytes)
    assert used_pool % EXPERT_SLOT_BYTES == 0
    return {
        "scenario": name,
        "context_tokens": context_tokens,
        "features": features,
        "operating_bytes": operating_bytes,
        "over_budget": over_budget,
        "shortfall_bytes": max(0, fixed - operating_bytes),
        "components": components,
        "expert_slots": slots,
        "slots_per_layer_even": round(slots / N_BACKBONE_LAYERS, 2),
        "pct_of_all_experts": round(100 * slots / N_BACKBONE_EXPERTS, 2),
        "pool_if_staging": {
            str(n): _pool_for_staging(operating_bytes, fixed - staging, n)
            for n in (1, 4, 8, 16, 32)
        },
    }


def _prov(n: int, provenance: str, note: str = "") -> dict:
    return {"bytes": n, "provenance": provenance, "note": note}


def _pool_for_staging(operating: int, fixed_without_staging: int, n: int) -> dict:
    pool = operating - fixed_without_staging - n * EXPERT_SLOT_BYTES
    slots = pool // EXPERT_SLOT_BYTES if pool > 0 else 0
    return {"slots": slots, "pct": round(100 * slots / N_BACKBONE_EXPERTS, 2)}
