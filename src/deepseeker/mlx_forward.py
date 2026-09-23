"""MLX-native DeepSeek-V4.1-Flash forward, decode-first (roadmap: runner epic).

Exact port of inference/model.py math to MLX, batch 1, text-only:
RMSNorm, sqrtsoftplus router, SwiGLU MoE (streamed fp4 experts),
shared expert, Hyper-Connection mixes/pre/post, Engram gather+gate,
parallel head, YaRN RoPE, sparse attention with sink, compressor and
indexer state machines, window ring + compress caches.

Known deviations from the reference (documented, bounded):
- caches held bf16, not fp8-quantized (no MLX fp8 matmul);
- tilelang kernels replaced by MLX ops (same math);
- single rank (world_size 1): no all-reduce/all-gather;
- vision/DSpark/MTP paths not implemented (text decode only).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def rmsnorm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """y = x / sqrt(mean(x^2) + eps) * weight, float32 accumulate."""
    xf = x.astype(mx.float32)
    rstd = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * rstd * weight.astype(mx.float32)).astype(x.dtype)


def sqrtsoftplus_scores(x: mx.array, gate_w: mx.array) -> mx.array:
    """Router scores: sqrt(softplus(x @ W.T)) in float32."""
    scores = (x.astype(mx.float32) @ gate_w.astype(mx.float32).T)
    return mx.sqrt(mx.logaddexp(scores, mx.zeros_like(scores)))


def route_topk(scores: mx.array, bias: mx.array, top_k: int,
               route_scale: float = 1.5) -> tuple[mx.array, mx.array]:
    """Authoritative routing: topk(scores + bias), weights from raw scores.

    Returns (weights [n, top_k] float32, indices [n, top_k] int). The
    bias steers selection only. norm_topk_prob with 1e-20 (not norm_eps).
    """
    biased = scores + bias.astype(mx.float32)
    order = mx.argsort(biased, axis=-1)
    indices = order[:, -top_k:]
    rows = mx.arange(scores.shape[0])[:, None]
    weights = scores[rows, indices]
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return weights * route_scale, indices


def moe_layer(x: mx.array, gate_w: mx.array, gate_b: mx.array,
              expert_fn, shared_fn, top_k: int = 6, limit: float = 10.0,
              trace_hook=None, layer_id: int = 0, prefetch_fn=None) -> mx.array:
    """MoE: route (exact), per-expert grouped eval, shared expert add.

    expert_fn(expert) -> (w1, w3, w2) bf16 arrays (streamed).
    trace_hook(layer, indices, weights) records ground truth (#20).
    prefetch_fn(layer, unique_experts) warms cache in parallel before eval.
    """
    scores = sqrtsoftplus_scores(x, gate_w)
    weights, indices = route_topk(scores, gate_b, top_k)
    out = mx.zeros_like(x, dtype=mx.float32)
    idx_np = np.asarray(indices, dtype=np.int64)
    w_np = np.asarray(weights, dtype=np.float32)
    unique = np.unique(idx_np)
    if prefetch_fn is not None:
        prefetch_fn(layer_id, [int(e) for e in unique])
    for expert in unique:
        mask = idx_np == expert
        rows = np.where(mask.any(axis=1))[0]
        cols = [int(np.where(mask[r])[0][0]) for r in rows]
        w1, w3, w2down = expert_fn(int(expert))
        xr = x[mx.array(rows)]
        g = xr.astype(mx.float32) @ w1.astype(mx.float32).T
        u = xr.astype(mx.float32) @ w3.astype(mx.float32).T
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
        h = (g / (1.0 + mx.exp(-g))) * u
        rw = mx.array(w_np[rows, cols], dtype=mx.float32).reshape(-1, 1)
        out = out.at[mx.array(rows)].add((h * rw) @ w2down.astype(mx.float32).T)
        del xr, g, u, h, rw
    if trace_hook is not None:
        trace_hook(layer_id, idx_np.tolist(), w_np.tolist())
    out = out + shared_fn(x).astype(mx.float32)
    return out.astype(x.dtype)


def hc_split_sinkhorn(mixes: mx.array, hc_scale: mx.array, hc_base: mx.array,
                      hc_mult: int, iters: int = 20, eps: float = 1e-6):
    """Split mixes into pre/post/comb (doubly-stochastic comb via Sinkhorn).

    Exact port of the reference kernel: row-softmax with +eps, then
    alternating column/row normalizations, ending on a column step
    (comb is column-stochastic, not row-stochastic).
    """
    pre = mx.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(mixes[:, hc_mult:2 * hc_mult] * hc_scale[1]
                          + hc_base[hc_mult:2 * hc_mult])
    frag = (mixes[:, 2 * hc_mult:].reshape(-1, hc_mult, hc_mult) * hc_scale[2]
            + hc_base[2 * hc_mult:].reshape(hc_mult, hc_mult))
    row_max = mx.max(frag, axis=-1, keepdims=True)
    comb = mx.exp(frag - row_max)
    comb = comb / mx.sum(comb, axis=-1, keepdims=True) + eps
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(int(iters) - 1):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    return pre, post, comb


def hc_mixes(x: mx.array, hc_fn: mx.array, hc_scale: mx.array, hc_base: mx.array,
             hc_mult: int, norm_eps: float, iters: int = 20, eps: float = 1e-6):
    """Coefficients from the flattened stream (RMS over hc*dim jointly)."""
    flat = x.reshape(x.shape[0], x.shape[1], -1).astype(mx.float32)
    n, hc, d = flat.shape
    flat2d = flat.reshape(n, hc * d)
    rstd = mx.rsqrt(mx.mean(flat2d * flat2d, axis=-1, keepdims=True) + norm_eps)
    mixes = (flat2d @ hc_fn.astype(mx.float32).T) * rstd
    return hc_split_sinkhorn(mixes, hc_scale.astype(mx.float32),
                             hc_base.astype(mx.float32), hc_mult, iters, eps)


def hc_pre(x: mx.array, pre_mix: mx.array) -> mx.array:
    # x [S,hc,d], pre_mix [S,hc]: collapse over the hc axis (axis 1).
    y = mx.sum(pre_mix[..., None].astype(mx.float32) * x.astype(mx.float32), axis=1)
    return y.astype(x.dtype)


def hc_post(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    y = (post[..., None].astype(mx.float32) * x[..., None, :].astype(mx.float32)
         + mx.sum(comb[..., None].astype(mx.float32)
                  * residual[..., None, :].astype(mx.float32), axis=2))
    return y.astype(x.dtype)


def rope_freqs(dim: int, seqlen: int, original_seq_len: int, base: float,
               factor: float, beta_fast: float, beta_slow: float) -> mx.array:
    """YaRN rotary frequencies [seqlen, dim//2] complex (matches precompute)."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    outer = np.outer(np.arange(seqlen), freqs)
    return mx.array(outer)


def apply_rotary(x: mx.array, freqs: mx.array) -> mx.array:
    """Rotate last-dim pairs: x [..., S?, H?, rd] by freqs [S, rd//2].

    The seqlen axis broadcasts over any middle (heads) axes.
    """
    d = x.shape[-1]
    xr = x.astype(mx.float32).reshape(*x.shape[:-1], d // 2, 2)
    s = freqs.shape[0]
    mid = [1] * (xr.ndim - 3)
    cos = mx.cos(freqs).reshape(s, *mid, d // 2)
    sin = mx.sin(freqs).reshape(s, *mid, d // 2)
    re, im = xr[..., 0], xr[..., 1]
    out = mx.stack([re * cos - im * sin, re * sin + im * cos], axis=-1)
    return out.reshape(x.shape).astype(x.dtype)
