"""Tests for Issue #24 staging engine (fake reader; no checkpoint)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.pool import ResidentPool
from deepseeker.stage import StagingEngine, StagingError

FAKE_MANIFEST = {
    "tensors": [
        {"name": f"w{i}", "subsystem": "routed-expert", "layer": 0, "expert": e,
         "shard": "s.safetensors", "file_range": [i * 10 + e * 1000, i * 10 + e * 1000 + 10],
         "bytes": 10}
        for e in (0, 1) for i in range(6)
    ]
}
READS: list[tuple[int, int]] = []
READS_LOCK = threading.Lock()


def fake_reader(path, start, size):
    with READS_LOCK:
        READS.append((start, size))
    time.sleep(0.01)
    return bytes([(start + i) % 256 for i in range(size)])


def engine(**kwargs):
    kwargs.setdefault("reader", fake_reader)
    return StagingEngine(Path("/nonexistent"), FAKE_MANIFEST, **kwargs)


def test_plan_has_six_ranges():
    eng = engine()
    plan = eng.expert_plan(0, 0)
    assert len(plan) == 6
    assert sum(expected for _, _, _, expected in plan) == 60
    eng.shutdown()


def test_no_span_coalescing():
    READS.clear()
    eng = engine(max_workers=2)
    blob = eng.wait(eng.submit(0, 0))
    assert len(blob) == 60
    with READS_LOCK:
        assert len(READS) == 6  # one pread per range, never [start0, endN]
    eng.shutdown()


def test_dedup_and_cancel():
    eng = engine(max_workers=1)
    t1 = eng.submit(0, 0)
    t2 = eng.submit(0, 0)
    assert t2 is t1
    assert eng.metrics()["dedup_hits"] == 1
    assert len(eng.wait(t1)) == 60
    t3 = eng.submit(0, 1)
    assert eng.cancel(t3) in (True, False)  # racy by design; must not raise
    eng.shutdown()


def test_validation_catches_short_bytes():
    calls = {"n": 0}

    def flaky_reader(path, start, size):
        calls["n"] += 1
        if calls["n"] == 1:
            return b"\x00" * (size - 1)
        return bytes([(start + i) % 256 for i in range(size)])

    eng = StagingEngine(Path("/x"), FAKE_MANIFEST, reader=flaky_reader)
    with pytest.raises(StagingError):
        eng.wait(eng.submit(0, 0))
    assert eng.metrics()["failures"] == 1
    assert len(eng.wait(eng.submit(0, 1))) == 60  # engine stays usable
    eng.shutdown()


def test_staging_bound_enforced():
    eng = engine(max_workers=1, max_staging_bytes=60)
    t1 = eng.submit(0, 0)
    with pytest.raises(RuntimeError):
        eng.submit(0, 1)  # 60B reserved + 60B would exceed
    assert len(eng.wait(t1)) == 60
    assert len(eng.wait(eng.submit(0, 1))) == 60
    eng.shutdown()


def test_pool_integration_byte_exact():
    eng = engine()
    pool = ResidentPool(2, 60)
    assert eng.stage_into_pool(pool, 0, 0) is None
    assert eng.stage_into_pool(pool, 0, 1) is None
    assert pool.bytes_resident == 120
    assert pool.lookup(0, 0) is not None
    eng.shutdown()
