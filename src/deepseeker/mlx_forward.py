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

import functools
import math
import os
import time
from concurrent.futures import as_completed

import mlx.core as mx
import numpy as np

from deepseeker.mlx_weights import Q3_BITS, Q3_GROUP_SIZE, ExpertLoader

PROF = os.environ.get("DEEPSEEKER_PROFILE_SYNC") == "1"
SYNC: dict[str, float] = {"gpu_wait": 0.0, "cpu": 0.0, "n": 0.0,
                          "prefetch": 0.0, "expert_gpu": 0.0}

if PROF:
    import atexit

    @atexit.register
    def _sync_report() -> None:
        g, c, n = SYNC["gpu_wait"], SYNC["cpu"], SYNC["n"]
        tot = g + c
        print(f"\nSYNC n={int(n)} moe_total={tot:.2f}s gpu_wait={g:.2f}s "
              f"cpu={c:.2f}s (prefetch={SYNC['prefetch']:.2f}s) "
              f"expert_gpu={SYNC['expert_gpu']:.2f}s", flush=True)


def rmsnorm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """y = x / sqrt(mean(x^2) + eps) * weight, float32 accumulate.

    For float32 inputs this goes through ``mx.fast.rms_norm``, a single fused
    kernel. It matches the explicit form to ~3e-7 relative but costs far less:
    the hand-written form is a reduction plus a handful of elementwise kernels,
    and on a decode step those launches dominate the step (measured ~600us
    versus ~245us for the fused form on [32, 5120]). Lower precision inputs keep
    the explicit path so their accumulate-and-cast behaviour is unchanged.
    """
    if x.dtype == mx.float32:
        return mx.fast.rms_norm(x, weight, eps)
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


def qmm(x: mx.array, w: tuple) -> mx.array:
    """x @ W.T with W held as compressed mxfp4 or q3-affine operands.

    mxfp4: (uint32 words, uint8 ue8m0 scales) -- MLX dequantizes into
    registers and fuses it into the GEMM, so the weight never materializes
    as bf16: 3.76x less memory traffic and no dequant pass. Input stays
    float32 (accumulate); output float32, max rel diff ~3e-7 against the
    dequant-then-bf16-GEMM staging path.

    q3: (uint32 words, float16 scales, float16 biases) -- MLX-native affine
    quantization at group_size=64, bits=3, produced by requant_experts_3bit.
    Same fused-dequant property; ~17% fewer bytes per token than mxfp4.
    """
    packed, scales, *rest = w
    if rest:
        return mx.quantized_matmul(x, packed, scales=scales, biases=rest[0],
                                   transpose=True, group_size=Q3_GROUP_SIZE,
                                   bits=Q3_BITS)
    return mx.quantized_matmul(x, packed, scales=scales, transpose=True,
                               group_size=32, bits=4, mode="mxfp4")


def stack_gate_up_expert(weights: dict) -> dict:
    """Replace separate packed w1/w3 tensors with one output-row-concatenated w13."""
    if "w13" in weights:
        return weights
    w1, w3 = weights["w1"], weights["w3"]
    if len(w1) != len(w3) or len(w1) not in (2, 3):
        raise ValueError("w1/w3 operand arity must be 2 (mxfp4) or 3 (q3)")
    for a, b in zip(w1, w3):
        if a.shape[1:] != b.shape[1:] or a.shape[0] != b.shape[0]:
            raise ValueError("gate/up packed shapes must agree")
    return {"w13": (tuple(mx.concatenate((a, b), axis=0)
                         for a, b in zip(w1, w3))),
            "w2": weights["w2"]}


def split_gate_up_expert(weights: dict) -> dict:
    """Return w1/w3 views from a cached w13 representation without copying."""
    if "w13" not in weights:
        return weights
    parts = weights["w13"]
    packed = parts[0]
    if packed.shape[0] % 2:
        raise ValueError("stacked gate/up rows must divide evenly")
    out = {}
    for part in parts:
        scale_half = part.shape[0] // 2
        if part.shape[0] % 2:
            raise ValueError("stacked gate/up operand rows must divide evenly")
        out.setdefault("w1", []).append(part[:scale_half])
        out.setdefault("w3", []).append(part[scale_half:])
    return {"w1": tuple(out["w1"]), "w3": tuple(out["w3"]),
            "w2": weights["w2"]}


def gate_up_qmm(x: mx.array, weights, *, stack_single_row: bool = False
                ) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array]]:
    """Project gate/up, fusing their qmm only for a single routed row."""
    if isinstance(weights, dict):
        expert = weights
    else:
        w1, w3, w2 = weights
        expert = {"w1": w1, "w3": w3, "w2": w2}
    if "w13" not in expert and stack_single_row and x.shape[0] == 1:
        expert = stack_gate_up_expert(expert)
    if "w13" in expert and stack_single_row and x.shape[0] == 1:
        gate, up = mx.split(qmm(x, expert["w13"]), 2, axis=-1)
    else:
        separate = split_gate_up_expert(expert)
        gate = qmm(x, separate["w1"])
        up = qmm(x, separate["w3"])
    return gate, up, expert["w2"]


def gather_pinned_contributions(
        x: mx.array, indices: np.ndarray, weights: np.ndarray,
        banks: dict[str, tuple], slots: dict[int, int], limit: float = 10.0
        ) -> dict[int, tuple[np.ndarray, mx.array]]:
    """Compute selected pinned experts with three banked gather_qmm calls.

    Returns each expert's routed row indices and weighted output rows. Dynamic
    experts remain on the ordinary per-expert path. Pair order is sorted by
    expert then row so the caller can preserve the reference scatter-add order.
    Banks carry mxfp4 pairs or q3-affine triples; both use fused-dequant
    gather_qmm.
    """
    pairs = [(int(expert), row, col)
             for expert in sorted(slots)
             for row in range(indices.shape[0])
             for col in range(indices.shape[1])
             if int(indices[row, col]) == int(expert)]
    if not pairs:
        return {}
    experts = [p[0] for p in pairs]
    rows = np.asarray([p[1] for p in pairs], dtype=np.int32)
    positions: dict[int, list[int]] = {}
    for pos, expert in enumerate(experts):
        positions.setdefault(expert, []).append(pos)
    rhs = mx.array([slots[expert] for expert in experts], dtype=mx.int32)
    lhs = mx.arange(len(pairs), dtype=mx.int32)
    routed_x = x[mx.array(rows)].astype(mx.float32)[:, None, :]

    def gather(role: str, activations: mx.array) -> mx.array:
        parts = banks[role]
        if len(parts) == 3:
            return mx.gather_qmm(
                activations, parts[0], scales=parts[1], biases=parts[2],
                lhs_indices=lhs, rhs_indices=rhs, transpose=True,
                group_size=Q3_GROUP_SIZE, bits=Q3_BITS
            ).reshape(len(pairs), -1)
        return mx.gather_qmm(
            activations, parts[0], scales=parts[1], lhs_indices=lhs,
            rhs_indices=rhs, transpose=True, group_size=32, bits=4,
            mode="mxfp4").reshape(len(pairs), -1)

    gate = mx.minimum(gather("w1", routed_x), limit)
    up = mx.clip(gather("w3", routed_x), -limit, limit)
    hidden = gate / (1.0 + mx.exp(-gate)) * up
    route_w = mx.array(np.asarray(
        [weights[row, col] for _, row, col in pairs], dtype=np.float32))[:, None]
    contrib = gather("w2", (hidden * route_w)[:, None, :])
    return {
        expert: (np.asarray([pairs[pos][1] for pos in pos_list], dtype=np.int32),
                 contrib[mx.array(pos_list)])
        for expert, pos_list in positions.items()
    }


def cap_union(idx: np.ndarray, w: np.ndarray, n_experts: int,
              budget: int, *, scores: np.ndarray | None = None,
              strict: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Cap the batch's expert union at `budget` experts per layer.

    Keeps the experts with the largest summed gate weight across the whole
    step (optimal retention for a byte budget), zeroes the rest and
    renormalises each row back to its original weight sum so surviving
    contributions keep their scale. Dropped slots become index -1 and are
    skipped by the expert loop, so the union - and with it the expert reads -
    actually shrinks. A row that would lose everything keeps its own best
    expert, so the union may exceed the budget by at most `rows`.
    `strict=True` drops that rescue: the union is then exactly `budget`, which
    is what actually cuts the SSD reads. `scores` supplies the raw gate logits
    used for ranking - at top_k=1 `w` is a constant after normalisation and
    cannot order the experts.

    Returns (indices, weights, weight_coverage) where coverage is the share of
    total gate weight retained *before* renormalisation - the quality metric.
    """
    if np.unique(idx).size <= budget:
        return idx, w, 1.0
    score = np.zeros(n_experts, dtype=np.float32)
    if scores is not None:
        raw = scores[np.arange(idx.shape[0])[:, None], idx]
        np.add.at(score, idx.ravel(), raw.ravel())
    else:
        np.add.at(score, idx.ravel(), w.ravel())
    keep = np.argpartition(score, -budget)[-budget:]
    keep_mask = np.zeros(n_experts, dtype=bool)
    keep_mask[keep] = True
    kept = keep_mask[idx]
    empty = ~kept.any(axis=1)
    if empty.any() and not strict:
        kept[np.where(empty)[0], w[empty].argmax(axis=1)] = True
    kept_w = np.where(kept, w, 0.0)
    total = float(w.sum())
    kept_sum = float(kept_w.sum())
    scale = w.sum(axis=1, keepdims=True) / (kept_w.sum(axis=1, keepdims=True) + 1e-20)
    out_w = (kept_w * scale).astype(np.float32)
    out_idx = np.where(kept, idx, -1).astype(np.int64)
    return out_idx, out_w, (kept_sum / total if total > 0 else 1.0)


def moe_layer(x: mx.array, gate_w: mx.array, gate_b: mx.array,
              expert_fn, shared_fn, top_k: int = 6, limit: float = 10.0,
              trace_hook=None, layer_id: int = 0, prefetch_fn=None,
              union_budget: int = 0, route_stats: dict | None = None,
              union_strict: bool = False, pipeline: bool = False,
              async_submit_fn=None, async_finalize_fn=None,
              pinned_gather_fn=None, stack_gateup: bool = False) -> mx.array:
    """MoE: route (exact), per-expert grouped eval, shared expert add.

    shared_fn=None skips the shared expert entirely (speed mode).

    expert_fn(expert) -> (w1, w3, w2) mxfp4 operand pairs (streamed packed).
    trace_hook(layer, indices, weights) records ground truth (#20).
    prefetch_fn(layer, unique_experts) warms cache in parallel before eval.
    union_budget > 0 with a multi-row batch drops globally-unpopular experts
    (see cap_union); route_stats accumulates retained gate weight.

    pipeline=True selects the opt-in async pipeline (experimental): experts
    are submitted to parallel I/O+pack+upload upfront without waiting, and
    each expert's qmm is mx.async_eval'd as soon as its Future completes,
    overlapping GPU execution with waiting for the next Future's SSD load.
    Requires async_submit_fn(layer, expert) -> Future[(packed, nbytes, was_hit)]
    and async_finalize_fn(layer, sorted_experts, saved), which replays the
    baseline prefetch and expert lookup cache operations after async work.
    Routing, weighting, trace_hook, route_stats, hook, shared_out are
    identical to baseline. No cache ops during async phase (deterministic, no
    intermediate evictions); cache operations happen at end in baseline order.
    Scatter-add in sorted
    order at end (bit-identical). prefetch_fn skipped (async replaces it).
    On ANY exception: cancel pending (best-effort), propagate (fail fast,
    safe: no admits yet, cache unchanged, no partial results).
    """
    _tc = time.perf_counter() if PROF else 0.0
    scores = sqrtsoftplus_scores(x, gate_w)
    weights, indices = route_topk(scores, gate_b, top_k)
    out = mx.zeros(x.shape, dtype=mx.float32)
    _tr = time.perf_counter() if PROF else 0.0
    idx_np = np.asarray(indices, dtype=np.int64)
    w_np = np.asarray(weights, dtype=np.float32)
    if union_budget > 0 and idx_np.shape[0] > 1:
        s_np = np.asarray(scores, dtype=np.float32)  # free: synced above
        idx_np, w_np, coverage = cap_union(idx_np, w_np, scores.shape[1],
                                           union_budget, scores=s_np,
                                           strict=union_strict)
        if route_stats is not None:
            rows = float(idx_np.shape[0])
            route_stats["coverage"] += coverage * rows
            route_stats["calls"] += rows
    _tw = time.perf_counter() if PROF else 0.0
    # The shared expert only needs x, not the routing, so it is enqueued
    # before the blocking prefetch: the GPU works through it while the SSD
    # streams the routed experts.
    shared_out = (shared_fn(x).astype(mx.float32)
                  if shared_fn is not None else None)
    unique = np.unique(idx_np)
    unique = unique[unique >= 0]
    use_pipe = (pipeline and async_submit_fn is not None
                and async_finalize_fn is not None and not PROF
                and pinned_gather_fn is None)
    pinned_results: dict[int, tuple[np.ndarray, mx.array]] = {}
    if pinned_gather_fn is not None:
        # Schedule banked pin matmuls before blocking on dynamic expert reads.
        # The pin bank is already resident; its GPU work can overlap prefetch.
        pinned_results = pinned_gather_fn(layer_id, x, idx_np, w_np)
        if pinned_results:
            mx.async_eval([value for _, value in pinned_results.values()])
    if use_pipe:
        # Opt-in async pipeline (experimental, reversible). Submit all expert
        # loads upfront (non-blocking; hits return completed Futures, misses
        # return pending Futures for background SSD+pack with no cache touch).
        # Async-eval shared_out immediately so GPU starts while SSD loads in
        # the background. Then process Futures in completion order (max
        # overlap: GPU executes early qmms while waiting for later SSD loads),
        # building + async_eval each qmm as its weights arrive, SAVING outputs
        # (no scatter-add yet, no cache mutation). After ALL complete, the
        # finalize callback replays baseline prefetch and lookup operations in
        # order; then scatter-add saved contributions in sorted order.
        # prefetch_fn skipped (async replaces it). No cache ops during async
        # phase (deterministic, no intermediate evictions). On ANY exception:
        # cancel pending (best-effort), propagate (fail fast, safe: no admits
        # yet, cache unchanged, no partial results).
        if shared_out is not None:
            mx.async_eval(shared_out)
        futs: dict[int, object] = {}
        for e in unique:
            ee = int(e)
            futs[ee] = async_submit_fn(layer_id, ee)
        saved: dict[int, tuple] = {}
        fut_to_expert = {f: e for e, f in futs.items()}
        try:
            for fut in as_completed(futs.values()):
                e = fut_to_expert[fut]
                d, nbytes, was_hit, read_worker_s, pack_worker_s = fut.result()
                # Workers return NumPy operands; convert on this thread so
                # no lazy node is bound to a worker stream.
                d = {role: ExpertLoader.to_mx_operands(ops)
                     for role, ops in d.items()}
                mask = idx_np == e
                rows = np.where(mask.any(axis=1))[0]
                cols = [int(np.where(mask[r])[0][0]) for r in rows]
                xr = x[mx.array(rows)].astype(mx.float32)
                g, u, w2down = gate_up_qmm(
                    xr, d, stack_single_row=stack_gateup)
                g = mx.minimum(g, limit)
                u = mx.clip(u, -limit, limit)
                h = (g / (1.0 + mx.exp(-g))) * u
                rw = mx.array(w_np[rows, cols],
                              dtype=mx.float32).reshape(-1, 1)
                contrib = qmm(h * rw, w2down)
                mx.async_eval(contrib)
                saved[e] = (rows, contrib, d, nbytes, was_hit,
                            read_worker_s, pack_worker_s)
                del xr, g, u, h, rw
        except Exception:
            for f in futs.values():
                if not f.done():
                    f.cancel()
            raise
        async_finalize_fn(layer_id, [int(e) for e in unique], saved)
        for expert in unique:
            ee = int(expert)
            rows, contrib = saved[ee][:2]
            out = out.at[mx.array(rows)].add(contrib)
        if trace_hook is not None:
            trace_hook(layer_id, idx_np.tolist(), w_np.tolist())
        if shared_out is not None:
            out = out + shared_out
        return out.astype(x.dtype)
    if prefetch_fn is not None:
        prefetch_fn(layer_id, [int(e) for e in unique])
    _tpf = time.perf_counter() if PROF else 0.0
    for expert in unique:
        pinned = pinned_results.get(int(expert))
        if pinned is not None:
            rows, contrib = pinned
            out = out.at[mx.array(rows)].add(contrib)
            continue
        mask = idx_np == expert
        rows = np.where(mask.any(axis=1))[0]
        cols = [int(np.where(mask[r])[0][0]) for r in rows]
        expert_weights = expert_fn(int(expert))
        xr = x[mx.array(rows)].astype(mx.float32)
        g, u, w2down = gate_up_qmm(
            xr, expert_weights, stack_single_row=stack_gateup)
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
        h = (g / (1.0 + mx.exp(-g))) * u
        rw = mx.array(w_np[rows, cols], dtype=mx.float32).reshape(-1, 1)
        out = out.at[mx.array(rows)].add(qmm(h * rw, w2down))
        del xr, g, u, h, rw
    if trace_hook is not None:
        trace_hook(layer_id, idx_np.tolist(), w_np.tolist())
    if shared_out is not None:
        out = out + shared_out
    if PROF:
        _te = time.perf_counter()
        mx.eval(out)
        SYNC["prefetch"] += (_tpf - _tw)
        SYNC["expert_gpu"] += (time.perf_counter() - _te)
        SYNC["gpu_wait"] += (_tw - _tr)
        SYNC["cpu"] += (_tr - _tc) + (_te - _tw)
        SYNC["n"] += 1
    return out.astype(x.dtype)


@functools.lru_cache(maxsize=8)
def _sinkhorn_loop(iters: int, eps: float):
    """Compiled Sinkhorn normalizer: 38 launches collapse into one graph.

    The comb tensor is (n, hc, hc) = (8, 4, 4), so the alternating
    row/column normalizations are pure launch overhead (~78us/call) unless
    fused; hc_mixes runs twice per layer (80x per decode step).
    """

    def loop(comb: mx.array) -> mx.array:
        for _ in range(int(iters) - 1):
            comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
        return comb

    return mx.compile(loop)


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
    comb = _sinkhorn_loop(int(iters), float(eps))(comb)
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
