"""Tests for storage benchmark primitives and pattern generation.

Hermetic except manifest reads (metadata only). The real benchmark run
is validated separately by its own output checks.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.storage_bench import (
    BenchFile,
    Range,
    align_ranges,
    coalesce_ranges,
    describe_case,
    remap_ranges,
    run_qd,
    run_sustained,
    scatter_ranges,
)

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"


def load_benchmark_script():
    spec = importlib.util.spec_from_file_location(
        "benchmark_storage_patterns",
        REPO_ROOT / "scripts" / "benchmark_storage_patterns.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest():
    path = REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / REVISION / "model-manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_expert_ranges_match_manifest():
    script = load_benchmark_script()
    manifest = _manifest()
    ranges = script.expert_ranges(manifest, 12, 305)
    assert len(ranges) == 6
    assert sum(r.length for r in ranges) == 18800640
    expected = sorted(
        (t["file_range"][0], t["file_range"][1] - t["file_range"][0])
        for t in manifest["tensors"]
        if t["subsystem"] == "routed-expert" and t["layer"] == 12 and t["expert"] == 305
    )
    assert [(r.offset, r.length) for r in ranges] == expected


def test_align_and_coalesce_accounting():
    ranges = [Range(100, 264), Range(5000, 264), Range(9000, 264)]
    aligned = align_ranges(ranges, 4096)
    assert aligned[0] == Range(0, 4096)
    assert all(r.offset % 4096 == 0 and r.length % 4096 == 0 for r in aligned)
    merged = coalesce_ranges(ranges, 4096)
    assert merged == [Range(100, 264), Range(5000, 4264)]
    useful = sum(r.length for r in ranges)
    requested = sum(r.length for r in merged)
    assert useful <= requested <= sum(r.length for r in aligned)


def test_remap_preserves_geometry_inside_file():
    ranges = [Range(1_000_000, 264), Range(5_000_000, 4096), Range(9_000_000, 100)]
    replay = remap_ranges(ranges, 512 * 1024 * 1024, salt=3)
    assert [r.length for r in replay] == [264, 4096, 100]
    gaps_before = [b.offset - a.end for a, b in itertools.pairwise(ranges)]
    gaps_after = [b.offset - a.end for a, b in itertools.pairwise(replay)]
    assert gaps_before == gaps_after
    assert all(0 <= r.offset and r.end <= 512 * 1024 * 1024 for r in replay)
    again = remap_ranges(ranges, 512 * 1024 * 1024, salt=3)
    assert replay == again


def test_scatter_preserves_lengths_and_stays_inside():
    ranges = [Range(10**9, 264), Range(50 * 10**9, 8), Range(90 * 10**9, 4096)]
    replay = scatter_ranges(ranges, 512 * 1024 * 1024, seed=11)
    assert sorted(r.length for r in replay) == [8, 264, 4096]
    assert all(0 <= r.offset and r.end <= 512 * 1024 * 1024 for r in replay)
    assert scatter_ranges(ranges, 512 * 1024 * 1024, seed=11) == replay


def test_describe_case_labels_and_invariant():
    ranges = [Range(0, 264), Range(4096, 264)]
    aligned = align_ranges(ranges, 4096)
    stats = {"elapsed_s": 0.01, "repeats": 1, "p50_ms": 1.0}
    case = describe_case("demo", 528, ranges, aligned, stats, "advisory-uncached")
    assert case["cache_confidence"] == "advisory-uncached"
    assert case["useful_bytes"] == 528
    assert case["requested_bytes"] == 528
    assert case["aligned_bytes"] == 8192


def _tiny_bench(tmp_path: Path) -> BenchFile:
    bench = BenchFile(str(tmp_path / "tiny.bin"), 8 * 1024 * 1024)
    bench.create()
    return bench


def test_run_qd_depths_on_disposable_file(tmp_path):
    bench = _tiny_bench(tmp_path)
    ranges = [Range(i * 65536, 4096) for i in range(16)]
    for qd in (1, 2, 4):
        stats = run_qd(bench, ranges, qd, repeats=1)
        assert stats["samples"] == 16
        assert stats["total_bytes"] == 16 * 4096
        assert stats["min_ms"] <= stats["p50_ms"] <= stats["max_ms"]


def test_run_sustained_buckets(tmp_path):
    bench = _tiny_bench(tmp_path)
    ranges = [Range(i * 65536, 16384) for i in range(8)]
    result = run_sustained(bench, ranges, queue_depth=2, duration_s=2.0, bucket_s=1.0)
    assert result["total_bytes"] > 0
    assert len(result["buckets"]) >= 1
    assert result["avg_mib_s"] > 0


def test_group_and_expert_selection_shapes():
    script = load_benchmark_script()
    manifest = _manifest()
    for mode in ("same-layer", "cross-layer", "random"):
        ranges = script.group_ranges(manifest, mode, 4, seed=4)
        assert len(ranges) == 4 * 6
        assert sum(r.length for r in ranges) == 4 * 18800640
    experts = script.select_experts(manifest)
    assert set(experts) == {"early", "mid", "late", "min_span", "max_span"}
    assert experts["early"]["layer"] == 0
    assert experts["late"]["layer"] == 39
