"""Lockstep batch decode: gather/shift/topk correctness and cache stacking.

Pure-function tests plus a state-only Runner harness (no model weights), so
they stay in the fast suite. The engram case needs torch/transformers and the
pinned upstream reference and is skipped when those are missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.mlx_runner import Runner


def test_sparse_attn_batched_matches_row_loop():
    """3-D kv means each row owns its slot table; must equal per-row calls."""
    rng = np.random.default_rng(0)
    b, h, d, k = 3, 4, 8, 6
    q = mx.array(rng.normal(size=(b, h, d)).astype(np.float32))
    kv = mx.array(rng.normal(size=(b, k, d)).astype(np.float32))
    sink = mx.array(rng.normal(size=(h,)).astype(np.float32))
    idx = mx.array(np.array([[0, 2, 4, -1, -1, 3],
                             [1, 3, 5, 0, -1, -1],
                             [5, 4, 3, 2, 1, 0]], dtype=np.int32))
    scale = d**-0.5
    got = Runner._sparse_attn(None, q, kv, idx, sink, scale)
    for i in range(b):
        one = Runner._sparse_attn(None, q[i:i + 1], kv[i:i + 1],
                                  idx[i:i + 1], sink, scale)
        np.testing.assert_allclose(np.asarray(got[i]), np.asarray(one[0]),
                                   rtol=1e-5, atol=1e-5)


def test_sparse_attn_batched_matches_shared_2d_kv():
    """Rank-2 kv (the non-batch ring) still routes through the old gather."""
    rng = np.random.default_rng(1)
    b, h, d, k = 2, 3, 8, 5
    q = mx.array(rng.normal(size=(b, h, d)).astype(np.float32))
    kv = mx.array(rng.normal(size=(k, d)).astype(np.float32))
    sink = mx.array(np.zeros(h, dtype=np.float32))
    idx = mx.array(np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=np.int32))
    scale = d**-0.5
    got = Runner._sparse_attn(None, q, kv, idx, sink, scale)
    for i in range(b):
        one = Runner._sparse_attn(None, q[i:i + 1], kv, idx[i:i + 1],
                                  sink, scale)
        np.testing.assert_allclose(np.asarray(got[i]), np.asarray(one[0]),
                                   rtol=1e-5, atol=1e-5)


def test_window_topk_batch_broadcast_matches_rows():
    win, pos = 8, 19
    single = Runner._window_topk(win, 1, pos)
    batched = mx.broadcast_to(single, (4, win))
    assert batched.shape == (4, win)
    for i in range(4):
        np.testing.assert_array_equal(np.asarray(batched[i]),
                                      np.asarray(single[0]))


def test_shift_compress_idxs_batch():
    """Window ids stay put; compress ids are shifted by the window width."""
    win = 4
    idxs = mx.array(np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32))
    cidx = mx.array(np.array([[0, 2, -1], [1, -1, 3]], dtype=np.int32))
    got = np.asarray(Runner._shift_compress_idxs(idxs, cidx, win))
    ref = np.array([[0, 1, 2, 3, 4, 6, -1],
                    [4, 5, 6, 7, 5, -1, 7]], dtype=np.int32)
    np.testing.assert_array_equal(got, ref)


def test_select_candidates_batch_matches_rows():
    rng = np.random.default_rng(2)
    rows, width = 3, 17
    score = rng.normal(size=(rows, width))
    cl = np.array([[5], [9], [12]])
    kwargs = {"topk_blocks": 3, "block_size": 5}
    batched = Runner._select_candidates(None, mx.array(score), cl, **kwargs)
    for i in range(rows):
        one = Runner._select_candidates(
            None, mx.array(score[i:i + 1]), cl[i:i + 1], **kwargs)
        np.testing.assert_array_equal(np.asarray(batched)[i],
                                      np.asarray(one)[0])


def _snap(layers=(0, 1)) -> dict:
    d, win = 4, 8
    return {
        "window": {ly: mx.zeros((win, d)) for ly in layers},
        "compress": {ly: mx.zeros((2, d)) for ly in layers},
        "indexk": {ly: mx.zeros((3, 6)) for ly in layers},
        "kv_state": {ly: mx.zeros((2, d)) for ly in layers},
        "score_state": {ly: mx.zeros((2, d)) for ly in layers},
        "shared_indexk": mx.zeros((3, 6)),
        "shared_compress": mx.zeros((2, d)),
        "shared_topk": mx.zeros((5, 2), dtype=mx.int32),
        "candidates": mx.zeros((5, 7), dtype=mx.bool_),
        "mtp_window": {},
    }


def _bare(max_batch: int = 4) -> Runner:
    r = Runner.__new__(Runner)
    r._hash_bsz = max_batch
    r._batch = 0
    r._hash_row = 0
    r._batch_prefetch = False
    r._last_route = {}
    r._mtp_window = {}
    r._expert_protected = set()
    r._expert_used_now = set()
    return r


def test_begin_batch_stacks_every_row_cache():
    b = 3
    r = _bare()
    r.begin_batch([_snap(), _snap(), _snap()])
    assert r._batch == b
    assert r._window[0].shape == (b, 8, 4)
    assert r._compress[1].shape == (b, 2, 4)
    assert r._indexk[0].shape == (b, 3, 6)
    assert r._kv_state[1].shape == (b, 2, 4)
    assert r._score_state[1].shape == (b, 2, 4)
    # Per-row caches stack; row-shaped shared state is dropped for rebuild.
    assert r._shared_indexk.shape == (b, 3, 6)
    assert r._shared_compress.shape == (b, 2, 4)
    assert r._shared_topk is None
    assert r._candidates is None
    r.end_batch()
    assert r._batch == 0
    assert r._window == {}


def test_begin_batch_rejects_empty_and_mismatched_layers():
    r = _bare()
    with pytest.raises(ValueError, match="empty batch"):
        r.begin_batch([])
    with pytest.raises(ValueError, match="exceeds max_batch"):
        r.begin_batch([_snap()] * 5)
    short = _snap(layers=(0,))
    with pytest.raises(ValueError, match="inconsistent window"):
        r.begin_batch([_snap(), short])


def test_generate_batch_rejects_bad_prompt_sets():
    with pytest.raises(ValueError, match="empty prompts"):
        Runner.generate_batch(None, [], 4)
    with pytest.raises(ValueError, match="prompt lengths must match"):
        Runner.generate_batch(None, [[1, 2], [1, 2, 3]], 4)


def test_engram_batched_hashes_match_per_row_queries():
    """Batched lookback must read each sequence's own ngram cache row."""
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("transformers", reason="transformers not installed")
    upstream = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "inference"
    if not (upstream / "engram.py").exists():
        pytest.skip("pinned upstream reference code not fetched")
    cfg = json.loads((REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
                      / "inference" / "config.json").read_text())
    nrows = 3
    r = Runner.__new__(Runner)
    r._config = cfg
    r._root = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    r._hash_bsz = nrows
    r._hash_row = 0

    histories = [[101, 102, 103], [201, 202, 203], [301, 302, 303]]
    probes = [401, 402, 403]

    def prime() -> None:
        for i, seq in enumerate(histories):
            r._engram_hash_ids(seq, 0, row=i)

    prime()
    batched = r._engram_hash_ids(probes, 3, batched=True)
    assert batched.shape == (nrows, len(cfg["engram_layer_ids"]), r._hash_ncols)

    for i, tok in enumerate(probes):
        prime()  # identical cache contents for every reference query
        ref = r._engram_hash_ids([tok], 3, row=i)
        np.testing.assert_array_equal(batched[i], ref[0])


def test_cap_union_is_exact_when_the_union_fits():
    from deepseeker.mlx_forward import cap_union
    idx = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.int64)
    w = np.array([[0.7, 0.5, 0.3], [0.6, 0.5, 0.4]], dtype=np.float32)
    out_i, out_w, coverage = cap_union(idx, w, 8, 5)  # union is exactly 5
    np.testing.assert_array_equal(out_i, idx)
    np.testing.assert_allclose(out_w, w, rtol=1e-6)
    assert coverage == 1.0


def test_cap_union_shrinks_union_and_preserves_each_row_weight_sum():
    from deepseeker.mlx_forward import cap_union
    rng = np.random.default_rng(0)
    idx = np.array([[0, 1, 2], [0, 3, 4], [0, 5, 6],
                    [1, 3, 5], [2, 4, 6], [7, 8, 9]], dtype=np.int64)
    w = rng.random((6, 3)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)
    w *= 1.5
    out_i, out_w, coverage = cap_union(idx, w, 12, 4)

    live = out_i[out_i >= 0]
    # budget may grow by at most one forced expert per row
    assert np.unique(live).size <= 4 + idx.shape[0]
    np.testing.assert_allclose(out_w.sum(axis=1), w.sum(axis=1), rtol=1e-5)
    # kept slots keep their weight *ratio* within a row (rows are renormalised
    # back to their original sum, so only a common per-row factor may differ)
    kept = out_i >= 0
    for r in range(idx.shape[0]):
        vals = out_w[r][kept[r]] / w[r][kept[r]]
        np.testing.assert_allclose(vals, vals[0], rtol=1e-5)
        assert vals[0] >= 1.0 - 1e-6
    assert 0.0 < coverage <= 1.0


def test_cap_union_row_never_loses_every_expert():
    from deepseeker.mlx_forward import cap_union
    # row 0 routes to experts nobody else wants; a naive global top-3 drops it
    idx = np.array([[90, 91, 92],
                    [1, 2, 3],
                    [1, 2, 4]], dtype=np.int64)
    w = np.array([[0.5, 0.5, 0.5],
                  [5.0, 4.0, 3.0],
                  [5.0, 4.0, 3.0]], dtype=np.float32)
    out_i, out_w, coverage = cap_union(idx, w, 93, 3)
    assert (out_i[0] >= 0).sum() == 1
    np.testing.assert_allclose(out_w[0].sum(), w[0].sum(), rtol=1e-5)
    assert coverage < 1.0


def test_union_budget_only_applies_to_lockstep_decode():
    """A prefill's rows are one sequence's tokens: capping them corrupts it."""
    import deepseeker.mlx_forward as F

    r = Runner.__new__(Runner)
    r._dense = {"layers.0.ffn.gate.weight": mx.zeros((8, 4)),
                "layers.0.ffn.gate.bias": mx.zeros(8)}
    r._union_budget = 3
    r._route_stats = {"coverage": 0.0, "calls": 0.0}
    r._trace = None
    r._last_route = {}
    r._expert_fn = lambda layer, ns, **kw: None
    r._shared_fn = lambda layer, ns: None
    r.prefetch_experts = lambda *a, **k: None

    seen: dict = {}
    real = F.moe_layer

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return mx.zeros((2, 4))

    F.moe_layer = spy
    try:
        r._batch = 0
        r.moe(0, mx.zeros((2, 4)), 0)
        assert seen["union_budget"] == 0
        r._batch = 4
        r.moe(0, mx.zeros((2, 4)), 0)
        assert seen["union_budget"] == 3
    finally:
        F.moe_layer = real


def test_top_k_override_reaches_moe_layer():
    """Runner(top_k=N) must be what moe() hands the MoE layer, not config's 6."""
    import deepseeker.mlx_forward as F

    r = Runner.__new__(Runner)
    r._dense = {"layers.0.ffn.gate.weight": mx.zeros((8, 4)),
                "layers.0.ffn.gate.bias": mx.zeros(8)}
    r._union_budget = 0
    r._route_stats = {"coverage": 0.0, "calls": 0.0}
    r._trace = None
    r._last_route = {}
    r._batch = 0
    r._expert_fn = lambda layer, ns, **kw: None
    r._shared_fn = lambda layer, ns: None
    r.prefetch_experts = lambda *a, **k: None

    seen: dict = {}
    real = F.moe_layer

    def spy(*args, **kwargs):
        seen["top_k"] = args[5]
        return mx.zeros((2, 4))

    F.moe_layer = spy
    try:
        r._top_k = 2
        r.moe(0, mx.zeros((2, 4)), 0)
        assert seen["top_k"] == 2
        r._top_k = 6
        r.moe(0, mx.zeros((2, 4)), 0)
        assert seen["top_k"] == 6
    finally:
        F.moe_layer = real


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_route_topk_narrow_k_keeps_per_row_sum(k):
    """A narrower route must still give every row the full route_scale mass."""
    from deepseeker.mlx_forward import route_topk

    mx.random.seed(1)
    scores = mx.random.normal((5, 384)) * 0.05 + 0.4
    bias = mx.zeros((384,))
    mx.eval(scores, bias)
    w, idx = route_topk(scores, bias, k)
    mx.eval(w, idx)
    idx_np = np.asarray(idx)
    assert idx_np.shape == (5, k)
    assert idx_np.min() >= 0 and idx_np.max() < 384
    for row in range(5):
        assert len(set(idx_np[row].tolist())) == k
        assert abs(float(mx.sum(w[row]).item()) - 1.5) < 1e-4
