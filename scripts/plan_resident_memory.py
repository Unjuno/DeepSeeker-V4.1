#!/usr/bin/env python3
"""Resident-memory planner: manifest + capacity -> bounded S1-S4 plans.

    python scripts/plan_resident_memory.py

Reads profiles/m1-max-64gb/memory-capacity.json (Issue #5 probe),
the Issue #2 manifest, Issue #3 Engram analysis, and Issue #4 storage
results. Writes resident-plan.json (+ .md report) next to them.

Storage latency enters only as labeled parameters (optimistic /
advisory / unknown-cold); the planner never asserts unmeasured SSD
performance as fact.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.memory_plan import (
    N_BACKBONE_EXPERTS,
    build_scenario,
    common_bytes,
    dense_engram_bytes,
    lookup_table_bytes,
)

CONTEXTS = [4096, 32768, 131072, 524288, 1048576]
# Issue #3 measured these row-cache budgets; benefit is shown only
# where measured (no interpolation guesses).
ENGRAM_BENEFIT_BUDGETS = [16, 64, 256]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Resident-memory budget planner.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "profiles" / "m1-max-64gb")
    parser.add_argument("--engram-cache-mib", type=int, default=64)
    parser.add_argument("--staging-experts", type=int, default=8)
    parser.add_argument("--cold-ms-per-expert", type=float, default=None)
    args = parser.parse_args()

    revision = load_json(
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    cfg = load_json(REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "config.json")["text_config"]
    cfg["kv_source_layer_ids"] = [2, 8, 14, 20]
    cfg["index_source_layer_ids"] = [2, 8, 14, 20, 24, 28, 32, 36]
    capacity = load_json(args.out_dir / "memory-capacity.json")
    operating = capacity["recommendation"]["operating_gib"] * 1024**3
    engram_access = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "engram-access.json"
    )
    storage = load_json(args.out_dir / "storage-patterns.json")

    # Storage-confidence modes: labeled parameters, never asserted facts.
    single = storage["experts"]["early"]["modes"]["exact"]
    storage_modes = {
        "optimistic": {
            "label": "cache-served upper bound (measured QD8)",
            "expert_ms": single["qd8"]["median_of_runs"]["p50_ms"],
            "provenance": "measured-cache",
        },
        "advisory": {
            "label": "advisory-uncached QD1 (measured)",
            "expert_ms": single["qd1"]["median_of_runs"]["p50_ms"],
            "provenance": "measured-advisory",
        },
        "unknown-cold": {
            "label": "user-parameterized conservative cold guess",
            "expert_ms": args.cold_ms_per_expert
            or round(single["qd1"]["median_of_runs"]["p50_ms"] * 3.0, 2),
            "provenance": "assumed",
        },
    }

    engram_benefit = {
        str(mib): {
            corpus: data["row_lru_hit_rate_by_mib"][str(mib)]
            for corpus, data in engram_access["corpora"].items()
        }
        for mib in ENGRAM_BENEFIT_BUDGETS
    }

    scenarios: dict = {}
    plans = {
        "S1": {"mtp": False, "vision": False},
        "S2": {"mtp": True, "vision": False},
        "S3": {"mtp": False, "vision": True},
        "S4": {"mtp": False, "vision": False},
    }
    for name, features in plans.items():
        scenarios[name] = {}
        contexts = CONTEXTS if name in ("S1", "S2", "S4") else [4096, 32768]
        if name == "S3":
            contexts = [4096, 32768]
        for ctx in contexts:
            scenarios[name][str(ctx)] = build_scenario(
                f"{name}@{ctx}",
                operating,
                manifest,
                cfg,
                ctx,
                features,
                args.engram_cache_mib,
                args.staging_experts,
            )

    common_s1 = common_bytes(manifest, False, False)
    plan = {
        "schema": "deepseeker.resident-plan/v1",
        "upstream_revision": revision,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "operating_bytes": operating,
        "capacity_source": "memory-capacity.json",
        "common_model_bytes": common_s1,
        "dense_engram_bytes": dense_engram_bytes(manifest),
        "lookup_table_bytes_total": lookup_table_bytes(manifest),
        "lookup_table_resident_bytes": 0,
        "lookup_table_policy": ("conditional row-stream + cache (Issue #3); never fully resident"),
        "engram_cache_benefit_hit_rate": engram_benefit,
        "storage_modes": storage_modes,
        "scenarios": scenarios,
        "n_backbone_experts": N_BACKBONE_EXPERTS,
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "resident-plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    (out / "resident-plan.md").write_text(summarize(plan), encoding="utf-8")
    s1 = scenarios["S1"]["32768"]
    print(f"operating: {operating / 1024**3:.1f} GiB")
    print(
        f"S1@32K: {s1['expert_slots']} slots "
        f"({s1['pct_of_all_experts']}%), "
        f"{s1['slots_per_layer_even']}/layer"
    )
    print(f"wrote {out}/resident-plan.json")
    return 0


def _gib(n: int) -> str:
    return f"{n / 1024**3:.2f}"


def summarize(plan: dict) -> str:
    lines = [
        "# Resident-memory plan",
        "",
        f"Revision: `{plan['upstream_revision']}`",
        f"Generated: {plan['generated_at_utc']}",
        (f"Operating budget: {_gib(plan['operating_bytes'])} GiB (from safe-capacity probe)"),
        (
            f"Lookup tables resident: "
            f"{plan['lookup_table_resident_bytes']} B "
            f"({plan['lookup_table_policy']})"
        ),
        "",
        "## Scenarios (expert slots / % of 15,360)",
        "",
    ]
    for name, contexts in plan["scenarios"].items():
        for ctx, sc in contexts.items():
            kb = int(ctx) // 1024
            text = (
                f"- {name}@{kb}K: {sc['expert_slots']} slots "
                f"({sc['pct_of_all_experts']}%, "
                f"{sc['slots_per_layer_even']}/layer even)"
            )
            lines.append(text)
    lines += ["", "## Storage modes (expert service p50)", ""]
    for mode, spec in plan["storage_modes"].items():
        lines.append(f"- {mode}: {spec['expert_ms']} ms ({spec['provenance']})")
    lines += ["", "## Engram cache benefit (row-LRU hit rate, 64 MiB)", ""]
    for corpus, rates in plan["engram_cache_benefit_hit_rate"]["64"].items():
        lines.append(f"- {corpus}: {rates}")
    lines += [
        "",
        (
            "Provenance travels per component in resident-plan.json: "
            "measured / manifest / arch-estimate / assumed."
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
