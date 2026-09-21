"""Tests for runner forward components (shapes/math, no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.mlx_forward import (
    apply_rotary,
    hc_mixes,
    hc_post,
    hc_pre,
    hc_split_sinkhorn,
    moe_layer,
    rmsnorm,
    route_topk,
    sqrtsoftplus_scores,
)


def test_rmsnorm():
    x = mx.array([[3.0, 4.0]])
    w = mx.array([1.0, 1.0])
    out = rmsnorm(x, w, 0.0)
    mx.eval(out)
    assert np.allclose(np.asarray(out), [[0.84852815, 1.1313709]], atol=1e-5)


def test_router_topk_bias_steers():
    mx.random.seed(0)
    scores = mx.random.normal((2, 8)) * 0.01 + 0.5
    mx.eval(scores)
    bias = mx.array([0, 0, 0, 0, 0, 0, 0, 10.0])
    weights, indices = route_topk(scores, bias, 2)
    mx.eval(weights, indices)
    assert 7 in np.asarray(indices)[0]
    assert abs(float(mx.sum(weights[0]).item()) - 1.5) < 1e-4


def test_sqrtsoftplus_positive():
    x = mx.array([[-100.0, 0.0, 100.0]])
    w = mx.eye(3)
    s = sqrtsoftplus_scores(x, w)
    mx.eval(s)
    assert bool(((np.asarray(s) >= 0)).all())


def test_rotary_identity_at_zero():
    x = mx.random.normal((1, 4, 8))
    mx.eval(x)
    freqs = mx.zeros((1, 4))
    out = apply_rotary(x, freqs)
    mx.eval(out)
    assert np.allclose(np.asarray(out), np.asarray(x), atol=1e-5)


def test_rotary_preserves_norm():
    x = mx.random.normal((1, 4, 8))
    mx.eval(x)
    freqs = mx.random.normal((1, 4)) * 0.1
    out = apply_rotary(x, freqs)
    mx.eval(out)
    assert np.allclose(float(mx.sum(out * out).item()),
                       float(mx.sum(x * x).item()), rtol=1e-4)


def test_hc_mixes_shapes():
    mx.random.seed(5)
    hc, dim = 4, 8
    mix_hc = (2 + hc) * hc
    x = mx.random.normal((2, hc, dim))
    fn = mx.random.normal((mix_hc, hc * dim))
    scale = mx.ones((3,))
    base = mx.zeros((mix_hc,))
    mx.eval(x, fn, scale, base)
    pre, post, comb = hc_mixes(x, fn, scale, base, hc, 1e-6)
    mx.eval(pre, post, comb)
    assert pre.shape == (2, hc)
    assert post.shape == (2, hc)
    assert comb.shape == (2, hc, hc)


def test_hc_roundtrip_shapes():
    mx.random.seed(1)
    x = mx.random.normal((2, 4, 8))
    pre_mix = mx.ones((2, 4)) / 4
    mx.eval(x, pre_mix)
    collapsed = hc_pre(x, pre_mix)
    assert collapsed.shape == (2, 8)
    post = mx.ones((2, 4)) / 4
    comb = mx.ones((2, 4, 4)) / 4
    back = hc_post(collapsed, x, post, comb)
    assert back.shape == (2, 4, 8)


def test_sinkhorn_doubly_stochastic():
    mx.random.seed(2)
    mixes = mx.random.normal((3, (2 + 4) * 4))
    scale = mx.ones((3,))
    base = mx.zeros(((2 + 4) * 4,))
    mx.eval(mixes, scale, base)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, 4, iters=20)
    mx.eval(pre, post, comb)
    c = np.asarray(comb)
    assert np.allclose(c.sum(-2), 1.0, atol=0.05)  # ends on column step
    assert bool(((np.asarray(pre) > 0)).all())


def test_moe_layer_shapes_and_trace():
    mx.random.seed(3)
    dim, n_exp, top_k = 32, 8, 2
    x = mx.random.normal((3, dim))
    gate_w = mx.random.normal((n_exp, dim)) * 0.1
    gate_b = mx.zeros((n_exp,))
    bank = {e: (mx.random.normal((16, dim)) * 0.1,
                mx.random.normal((16, dim)) * 0.1,
                mx.random.normal((dim, 16)) * 0.1) for e in range(n_exp)}
    mx.eval(x, gate_w, gate_b, *[a for t in bank.values() for a in t])
    seen = []

    def expert_fn(e):
        return bank[e]

    def shared_fn(x):
        return mx.zeros_like(x)

    out = moe_layer(x, gate_w, gate_b, expert_fn, shared_fn, top_k,
                    trace_hook=lambda layer, idx, w: seen.append((layer, idx, w)),
                    layer_id=5)
    mx.eval(out)
    assert out.shape == (3, dim)
    assert len(seen) == 1 and seen[0][0] == 5
    assert np.asarray(seen[0][1]).shape == (3, top_k)
