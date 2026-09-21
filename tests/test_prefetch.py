"""Tests for Issue #25 gated prefetch (fake engine/pool; no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.pool import ResidentPool
from deepseeker.prefetch import PrefetchController


class FakeEngine:
    def __init__(self, store):
        self._store = store
        self.submits = 0

    def submit(self, layer, expert):
        self.submits += 1
        return (layer, expert)

    def wait(self, ticket, timeout=None):
        return self._store[ticket].copy()

    def shutdown(self):
        pass


def sticky_tokens(n_tokens=6, layers=2):
    groups = []
    for token in range(n_tokens):
        groups.append([
            {"request_id": "r", "token_pos": token, "layer": layer, "experts": [1, 2]}
            for layer in range(layers)
        ])
    return groups


def test_disabled_submits_nothing():
    pool = ResidentPool(2, 10)
    engine = FakeEngine({})
    controller = PrefetchController(pool, engine, enabled=False)
    for records in sticky_tokens():
        assert controller.prefetch_ahead() == 0
        controller.observe_token(records)
    assert engine.submits == 0


def test_enabled_prefetches_and_serves_identical():
    store = {(layer, e): np.full(10, layer * 10 + e, dtype=np.uint8)
             for layer in range(2) for e in (1, 2)}
    pool = ResidentPool(10, 10)
    engine = FakeEngine(store)
    controller = PrefetchController(pool, engine, enabled=True, budget=2)
    tok0 = [{"request_id": "r", "token_pos": 0, "layer": layer, "experts": [1, 2]}
            for layer in range(2)]
    controller.observe_token(tok0)  # history only, nothing served yet
    assert controller.prefetch_ahead() == 4  # pool empty: all predicted fetched
    assert controller.drain() == 4
    served = []
    for record in tok0:
        for expert in record["experts"]:
            served.append(controller.serve(
                record["layer"], expert,
                lambda layer, expert: store[(layer, expert)].copy()))
    for (layer, expert), blob in zip(
        [(0, 1), (0, 2), (1, 1), (1, 2)], served
    ):
        assert bytes(blob) == bytes(store[(layer, expert)])
    assert controller.report()["served_prefetched"] == 4


def test_shift_disables():
    pool = ResidentPool(4, 10)
    store = {(0, e): np.full(10, e, dtype=np.uint8) for e in range(8)}
    engine = FakeEngine(store)
    controller = PrefetchController(pool, engine, enabled=True, budget=2,
                                    min_agreement=0.9, window=2)
    for token in range(6):
        experts = [0, 1] if token < 2 else [4 + token, 5 + token]
        records = [{"request_id": "r", "token_pos": token, "layer": 0, "experts": experts}]
        controller.observe_token(records)
    assert controller.auto_disabled is True
    assert controller.enabled is False
    assert controller.prefetch_ahead() == 0


def test_serve_fallback_on_engine_failure():
    class DeadEngine(FakeEngine):
        def submit(self, layer, expert):
            raise RuntimeError("staging full")

    store = {(0, 1): np.full(10, 7, dtype=np.uint8)}
    pool = ResidentPool(2, 10)
    controller = PrefetchController(pool, DeadEngine(store), enabled=True)
    out = controller.serve(0, 1, lambda layer, expert: store[(layer, expert)].copy())
    assert bytes(out) == bytes(store[(0, 1)])
