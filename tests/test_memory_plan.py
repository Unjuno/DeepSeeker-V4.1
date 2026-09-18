"""Tests for the resident-memory budget model.

Byte accounting reconciles to the committed Issue #2 manifest;
no system access, no weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.memory_plan import (
    EXPERT_SLOT_BYTES,
    N_BACKBONE_EXPERTS,
    build_scenario,
    common_bytes,
    dense_engram_bytes,
    lookup_table_bytes,
    manifest_by_subsystem,
)

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"


def _manifest():
    path = REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / REVISION / "model-manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _cfg():
    path = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))["text_config"]
    cfg["kv_source_layer_ids"] = [2, 8, 14, 20]
    cfg["index_source_layer_ids"] = [2, 8, 14, 20, 24, 28, 32, 36]
    return cfg


def test_manifest_reconciliation():
    manifest = _manifest()
    by_sub = manifest_by_subsystem(manifest)
    assert sum(b["bytes"] for b in by_sub.values()) == manifest["totals"]["tensor_bytes"]
    common = common_bytes(manifest, False, False)
    rest = (
        by_sub["routed-expert"]["bytes"]
        + by_sub["mtp-expert"]["bytes"]
        + by_sub["mtp"]["bytes"]
        + by_sub["mtp-router"]["bytes"]
        + by_sub["mtp-shared"]["bytes"]
        + by_sub["vision"]["bytes"]
        + by_sub["engram"]["bytes"]
    )
    assert common["bytes"] + rest == manifest["totals"]["tensor_bytes"]
    assert dense_engram_bytes(manifest)["bytes"] < by_sub["engram"]["bytes"]
    assert lookup_table_bytes(manifest) > 150 * 1024**3  # ~189 GiB


def test_engram_tables_never_fully_resident():
    manifest = _manifest()
    tables = lookup_table_bytes(manifest)
    operating = 48 * 1024**3
    assert tables > operating  # cannot fit by construction
    scenario = build_scenario(
        "test", operating, manifest, _cfg(), 32768, {"mtp": False, "vision": False}, 64, 8
    )
    cache = next(c for c in scenario["components"] if c["name"] == "engram-cache")
    assert cache["bytes"] == 64 * 1024 * 1024
    assert tables not in [c["bytes"] for c in scenario["components"]]


def test_slot_arithmetic_and_budget():
    manifest = _manifest()
    operating = 48 * 1024**3
    scenario = build_scenario(
        "test", operating, manifest, _cfg(), 32768, {"mtp": False, "vision": False}, 64, 8
    )
    assert scenario["over_budget"] is False
    pool = next(c for c in scenario["components"] if c["name"] == "expert-pool")
    staging = next(c for c in scenario["components"] if c["name"] == "staging")
    assert pool["bytes"] % EXPERT_SLOT_BYTES == 0
    assert pool["bytes"] // EXPERT_SLOT_BYTES == scenario["expert_slots"]
    assert staging["bytes"] == 8 * EXPERT_SLOT_BYTES
    total = sum(c["bytes"] for c in scenario["components"])
    assert total == operating
    slack = next(c for c in scenario["components"] if c["name"] == "slack")
    assert slack["bytes"] >= 0
    assert scenario["pct_of_all_experts"] == round(
        100 * scenario["expert_slots"] / N_BACKBONE_EXPERTS, 2
    )
    for alt in scenario["pool_if_staging"].values():
        assert alt["slots"] * EXPERT_SLOT_BYTES <= operating


def test_context_growth_shrinks_pool():
    manifest = _manifest()
    operating = 48 * 1024**3
    slots = []
    flags = []
    for ctx in (4096, 32768, 131072, 524288, 1048576):
        sc = build_scenario(
            "test", operating, manifest, _cfg(), ctx, {"mtp": False, "vision": False}, 64, 8
        )
        slots.append(sc["expert_slots"])
        flags.append(sc["over_budget"])
    assert slots == sorted(slots, reverse=True)
    assert slots[0] > 0
    # Very long contexts may exceed budget under the high-uncertainty
    # index estimate; that must be flagged, never silently fitted.
    assert flags[-1] is True
    short = build_scenario(
        "test", operating, manifest, _cfg(), 1048576, {"mtp": False, "vision": False}, 64, 8
    )
    assert short["shortfall_bytes"] > 0


def test_scenario_determinism():
    manifest = _manifest()
    kwargs = {
        "operating_bytes": 48 * 1024**3,
        "manifest": manifest,
        "cfg": _cfg(),
        "context_tokens": 32768,
        "features": {"mtp": True, "vision": False},
        "engram_cache_mib": 64,
        "staging_experts": 8,
    }
    first = build_scenario("s", **kwargs)
    second = build_scenario("s", **kwargs)
    assert first == second


def test_kv_estimate_documents_formula():
    from deepseeker.memory_plan import kv_per_token_estimate

    manifest = _manifest()
    kv = kv_per_token_estimate(manifest, _cfg())
    assert kv["bytes_per_token"] == 4 * (512 + 64) * 2
    assert kv["provenance"] == "arch-estimate"
    assert "formula" in kv
