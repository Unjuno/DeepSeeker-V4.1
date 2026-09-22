"""Differential tests: MLX forward math vs direct torch translations.

Catches formula-level port bugs (axes, scales, missing terms) that shape
tests miss. Small random shapes; tight tolerances.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker import mlx_forward as F


def test_sparse_attn_matches_torch():
    torch.manual_seed(0)
    b, m, h, d = 1, 3, 4, 16
    q = torch.randn(b, m, h, d)
    kv = torch.randn(b, 9, d)
    sink = torch.randn(h)
    idx = torch.tensor([[[0, 2, 4, -1, -1]] * 1 + [[1, 3, 5, 6, 7]] + [[0, 1, 2, 3, 4]]])
    scale = d**-0.5
    # torch reference: gather + sink-augmented softmax
    outs = []
    for qi in range(m):
        rows = idx[0, qi]
        valid = rows[rows >= 0]
        kk = kv[0, valid]
        s = (q[0, qi] @ kk.T) * scale
        mrow = s.max(-1, keepdim=True).values
        e = torch.exp(s - mrow)
        w = e / (e.sum(-1, keepdim=True) + torch.exp(sink - mrow.squeeze(-1)).unsqueeze(-1))
        outs.append(w @ kk)
    ref = torch.stack(outs, dim=0).unsqueeze(0)
    from deepseeker.mlx_runner import Runner

    got = Runner._sparse_attn(
        None, mx.array(q[0].numpy()), mx.array(kv[0].numpy()),
        mx.array(idx[0].numpy()), mx.array(sink.numpy()), scale)
    np.testing.assert_allclose(np.asarray(got), ref[0].numpy(), rtol=1e-4, atol=1e-4)


def test_hc_mixes_matches_torch():
    torch.manual_seed(1)
    hc, dim = 4, 8
    mix_hc = (2 + hc) * hc
    x = torch.randn(2, hc, dim)
    fn = torch.randn(mix_hc, hc * dim)
    scale = torch.tensor([0.5, 1.5, 2.5])
    base = torch.randn(mix_hc) * 0.1
    flat = x.reshape(2, hc * dim).float()
    rstd = torch.rsqrt((flat * flat).mean(-1, keepdim=True) + 1e-20)
    mixes = (flat @ fn.T) * rstd
    pre_ref = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + 1e-6
    pre, post, comb = F.hc_mixes(
        mx.array(x.numpy()), mx.array(fn.numpy()),
        mx.array(scale.numpy()), mx.array(base.numpy()), hc, 1e-20, 20, 1e-6)
    mx.eval(pre, post, comb)
    np.testing.assert_allclose(np.asarray(pre), pre_ref.numpy(), rtol=1e-4, atol=1e-4)


def test_router_matches_torch_gate():
    torch.manual_seed(2)
    n, dim, n_exp, top_k = 3, 16, 8, 2
    x = torch.randn(n, dim)
    w = torch.randn(n_exp, dim) * 0.1
    b = torch.randn(n_exp)
    scores = torch.sqrt(torch.nn.functional.softplus(x.float() @ w.float().T))
    indices = (scores + b).topk(top_k, dim=-1)[1]
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 1.5
    gw, gidx = F.route_topk(
        F.sqrtsoftplus_scores(mx.array(x.numpy()), mx.array(w.numpy())),
        mx.array(b.numpy()), top_k)
    mx.eval(gw, gidx)
    # order may differ (ascending tail vs descending topk); pairs must match
    got_pairs = sorted(zip(np.asarray(gidx).ravel(),
                           np.asarray(gw).ravel()))
    want_pairs = sorted(zip(indices.numpy().ravel(), weights.numpy().ravel()))
    assert [p[0] for p in got_pairs] == [p[0] for p in want_pairs]
    np.testing.assert_allclose([p[1] for p in got_pairs], [p[1] for p in want_pairs],
                               rtol=1e-4, atol=1e-4)


def test_compressor_pool_matches_torch():
    torch.manual_seed(3)
    ratio, dim = 2, 16
    kv = torch.randn(5, dim)
    score = torch.randn(5, dim)
    cut = 4
    ref = (kv[:cut].reshape(-1, ratio, dim)
           * (score[:cut].reshape(-1, ratio, dim).softmax(dim=1))).sum(dim=1)
    s_reshaped = mx.array(score[:cut].numpy()).reshape(-1, ratio, dim)
    got = (mx.array(kv[:cut].numpy()).reshape(-1, ratio, dim)
           * mx.softmax(s_reshaped, axis=1)).sum(axis=1)
    mx.eval(got)
    np.testing.assert_allclose(np.asarray(got), ref.numpy(), rtol=1e-4, atol=1e-4)
