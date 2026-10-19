"""Tests for runner forward components (shapes/math, no checkpoint)."""

from __future__ import annotations

import sys
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.mlx_forward import (
    apply_rotary,
    gather_pinned_contributions,
    hc_mixes,
    hc_post,
    hc_pre,
    hc_split_sinkhorn,
    moe_layer,
    qmm,
    rmsnorm,
    route_topk,
    split_gate_up_expert,
    sqrtsoftplus_scores,
    stack_gate_up_expert,
)
from deepseeker.mlx_runner import Runner


def test_rmsnorm():
    x = mx.array([[3.0, 4.0]])
    w = mx.array([1.0, 1.0])
    out = rmsnorm(x, w, 0.0)
    mx.eval(out)
    assert np.allclose(np.asarray(out), [[0.84852815, 1.1313709]], atol=1e-5)


def test_gate_up_stack_roundtrip_preserves_single_row_qmm_exactly():
    """Cached w13 must reproduce the independent w1/w3 single-row outputs."""
    mx.random.seed(73)
    x = mx.random.normal((1, 64))
    w1 = mx.quantize(mx.random.normal((32, 64)), group_size=32,
                     bits=4, mode="mxfp4")[:2]
    w3 = mx.quantize(mx.random.normal((32, 64)), group_size=32,
                     bits=4, mode="mxfp4")[:2]
    expert = stack_gate_up_expert({"w1": w1, "w3": w3, "w2": w1})
    split = split_gate_up_expert(expert)
    expected = (qmm(x, w1), qmm(x, w3))
    combined = qmm(x, expert["w13"])
    actual = mx.split(combined, 2, axis=-1)
    mx.eval([*expected, *actual, *split["w1"], *split["w3"]])
    assert np.array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
    assert np.array_equal(np.asarray(actual[1]), np.asarray(expected[1]))
    assert np.array_equal(np.asarray(qmm(x, split["w1"])), np.asarray(expected[0]))
    assert np.array_equal(np.asarray(qmm(x, split["w3"])), np.asarray(expected[1]))


def test_stacked_gate_up_cache_preserves_single_and_multirow_moe_outputs():
    mx.random.seed(91)
    dim, mid, n_exp, top_k = 64, 32, 8, 2

    def pack(a):
        return mx.quantize(a, group_size=32, bits=4, mode="mxfp4")[:2]

    expert_weights = {
        e: {"w1": pack(mx.random.normal((mid, dim))),
            "w3": pack(mx.random.normal((mid, dim))),
            "w2": pack(mx.random.normal((dim, mid)))}
        for e in range(n_exp)
    }
    stacked_weights = {e: stack_gate_up_expert(w)
                       for e, w in expert_weights.items()}
    gate_w = mx.random.normal((n_exp, dim))
    gate_b = mx.zeros((n_exp,))

    for rows in (1, 3):
        x = mx.random.normal((rows, dim))
        expected = moe_layer(x, gate_w, gate_b,
                             lambda e: tuple(expert_weights[e][k]
                                              for k in ("w1", "w3", "w2")),
                             None, top_k)
        actual = moe_layer(x, gate_w, gate_b,
                           lambda e: stacked_weights[e], None, top_k,
                           stack_gateup=True)
        mx.eval(expected, actual)
        assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_runner_cache_stores_only_stacked_gate_up_weights_when_enabled():
    mx.random.seed(101)
    dim, mid = 64, 32

    def pack(a):
        return mx.quantize(a, group_size=32, bits=4, mode="mxfp4")[:2]

    weights = {"w1": pack(mx.random.normal((mid, dim))),
               "w3": pack(mx.random.normal((mid, dim))),
               "w2": pack(mx.random.normal((dim, mid)))}
    runner = Runner.__new__(Runner)
    runner._expert_gateup_stack = True
    runner._expert_cache = OrderedDict()
    runner._expert_freq = {}
    runner._expert_pinned = set()
    runner._expert_used_now = set()
    runner._expert_protected = set()
    runner._expert_policy = "lru"
    runner._fake_experts = False
    runner._expert_pin_gather = False
    runner._expert_cache_cap = 4
    runner._pin_resident = 0
    runner.expert_evictions = 0
    runner.expert_cache_hits = 0
    runner.expert_loads = 0
    stored = runner._expert_touch(("layers", 0, 3), weights,
                                  stack_gateup=True)
    assert set(stored) == {"w13", "w2"}
    assert stored is runner._expert_cache[("layers", 0, 3)]
    fetched = runner._expert_fn(0)(3)
    assert fetched is stored
    fetched_for_batch = runner._expert_fn(0, stack_gateup=False)(3)
    assert fetched_for_batch is stored


def test_async_moe_uses_stacked_gate_up_for_one_row():
    mx.random.seed(121)
    dim, mid, n_exp, top_k = 64, 32, 8, 2

    def pack(a):
        return mx.quantize(a, group_size=32, bits=4, mode="mxfp4")[:2]

    bank = {
        e: (pack(mx.random.normal((mid, dim))),
            pack(mx.random.normal((mid, dim))),
            pack(mx.random.normal((dim, mid))))
        for e in range(n_exp)
    }
    x = mx.random.normal((1, dim))
    gate_w = mx.random.normal((n_exp, dim))
    gate_b = mx.zeros((n_exp,))
    expected = moe_layer(x, gate_w, gate_b, lambda e: bank[e], None, top_k)

    def submit(_layer, expert):
        future = Future()
        w1, w3, w2 = bank[expert]
        future.set_result(({"w1": w1, "w3": w3, "w2": w2},
                           0, False, 0.0, 0.0))
        return future

    actual = moe_layer(
        x, gate_w, gate_b, lambda e: bank[e], None, top_k,
        pipeline=True, async_submit_fn=submit, async_finalize_fn=lambda *_: None,
        stack_gateup=True)
    mx.eval(expected, actual)
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


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
    dim, mid, n_exp, top_k = 64, 32, 8, 2
    x = mx.random.normal((3, dim))
    gate_w = mx.random.normal((n_exp, dim)) * 0.1
    gate_b = mx.zeros((n_exp,))

    def pack(a):
        w, s = mx.quantize(a, group_size=32, bits=4, mode="mxfp4")
        return (w, s)

    bank = {e: (pack(mx.random.normal((mid, dim)) * 0.1),
                pack(mx.random.normal((mid, dim)) * 0.1),
                pack(mx.random.normal((dim, mid)) * 0.1)) for e in range(n_exp)}
    mx.eval(x, gate_w, gate_b)
    for pair in bank.values():
        for w, s in pair:
            mx.eval(w, s)
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


def test_moe_layer_banked_pins_match_serial_expert_path():
    """Banked MXFP4 gather must preserve the normal routed output exactly."""
    mx.random.seed(31)
    dim, mid, n_exp, top_k = 64, 32, 8, 3
    x = mx.random.normal((4, dim))
    gate_w = mx.random.normal((n_exp, dim)) * 0.1
    gate_b = mx.random.normal((n_exp,)) * 0.01

    def pack(a):
        return mx.quantize(a, group_size=32, bits=4, mode="mxfp4")

    experts = {
        e: (pack(mx.random.normal((mid, dim)) * 0.1),
            pack(mx.random.normal((mid, dim)) * 0.1),
            pack(mx.random.normal((dim, mid)) * 0.1))
        for e in range(n_exp)
    }
    mx.eval(x, gate_w, gate_b)
    for roles in experts.values():
        for w, s in roles:
            mx.eval(w, s)

    pins = [0, 2, 3, 6]
    slots = {expert: slot for slot, expert in enumerate(pins)}
    banks = {
        role: (mx.stack([experts[e][role_i][0] for e in pins]),
               mx.stack([experts[e][role_i][1] for e in pins]))
        for role, role_i in (("w1", 0), ("w2", 2), ("w3", 1))
    }
    mx.eval([a for pair in banks.values() for a in pair])

    def expert_fn(e):
        return experts[e]

    plain = moe_layer(x, gate_w, gate_b, expert_fn, None, top_k)

    def pinned_fn(layer, activations, indices, weights):
        del layer
        return gather_pinned_contributions(
            activations, indices, weights, banks, slots)

    banked = moe_layer(x, gate_w, gate_b, expert_fn, None, top_k,
                       pinned_gather_fn=pinned_fn)
    mx.eval(plain, banked)
    delta = np.max(np.abs(np.asarray(plain) - np.asarray(banked)))
    assert np.allclose(np.asarray(plain), np.asarray(banked), atol=1e-6, rtol=1e-5), delta


def test_no_shared_skips_the_shared_expert():
    """shared_fn=None is the speed mode: routed experts only."""
    mx.random.seed(7)
    dim, mid, n_exp, top_k = 64, 32, 8, 2
    x = mx.random.normal((3, dim))
    gate_w = mx.random.normal((n_exp, dim)) * 0.1
    gate_b = mx.zeros((n_exp,))

    def pack(a):
        w, s = mx.quantize(a, group_size=32, bits=4, mode="mxfp4")
        return (w, s)

    bank = {e: (pack(mx.random.normal((mid, dim)) * 0.1),
                pack(mx.random.normal((mid, dim)) * 0.1),
                pack(mx.random.normal((dim, mid)) * 0.1)) for e in range(n_exp)}
    mx.eval(x, gate_w, gate_b)
    for pair in bank.values():
        for w, s in pair:
            mx.eval(w, s)

    def expert_fn(e):
        return bank[e]

    out_no = moe_layer(x, gate_w, gate_b, expert_fn, None, top_k)
    out_yes = moe_layer(x, gate_w, gate_b, expert_fn,
                        lambda t: mx.ones_like(t) * 0.25, top_k)
    mx.eval(out_no, out_yes)
    assert out_no.shape == (3, dim)
    diff = float(mx.max(mx.abs(out_yes - out_no)).item())
    assert diff > 1e-3, "shared expert must actually change the output"
