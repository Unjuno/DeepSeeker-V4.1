"""MLX-native DeepSeek-V4.1-Flash runner (roadmap: runner epic).

Batch 1, text-only decode (+prefill), exact port of inference/model.py
math (components in mlx_forward). All tensors [S, ...] (batch squeezed).

Weight flow: dense bf16 resident (~15GB) via DenseLoader; routed
experts streamed per token via ExpertLoader (#33 dequant); engram
embed rows streamed via EngramRowCache (#26).

Trace hooks record authoritative routing/KV/Engram events (#20) with
zero effect on numerics. Deviations: bf16 caches (not fp8-quantized),
no distributed, no vision. DSpark/MTP draft path implemented (#29).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path

import mlx.core as mx
import numpy as np

from deepseeker import mlx_forward as F
from deepseeker.engram_cache import EngramRowCache
from deepseeker.mlx_weights import DenseLoader, ExpertLoader


class Runner:
    """Owns weights, caches, and streaming loaders for one session."""

    def __init__(self, model_root: Path | str, manifest: dict, config: dict,
                 trace_hook=None, expert_cache_cap: int = 256,
                 expert_prefetch_workers: int = 8) -> None:
        self._root = Path(model_root)
        self._manifest = manifest
        self._config = config
        self._trace = trace_hook
        self._expert_loader = ExpertLoader(model_root, manifest)
        self._expert_prefetch_workers = expert_prefetch_workers
        print("loading dense weights...", flush=True)
        self._dense = DenseLoader(model_root, manifest).load_all()
        print(f"dense resident: {sum(a.nbytes for a in self._dense.values()) / 2**30:.1f} GiB",
              flush=True)
        from deepseeker.engram import table_refs_from_manifest

        self._refs = table_refs_from_manifest(manifest)
        self._engram_cache = EngramRowCache(model_root, self._refs, 256 * 1024**2, 4096)
        self._freqs = None
        self._window: dict[int, mx.array] = {}
        self._compress: dict[int, mx.array] = {}
        self._indexk: dict[int, mx.array] = {}
        self._kv_state: dict[int, mx.array] = {}
        self._score_state: dict[int, mx.array] = {}
        self._shared_compress = None
        self._shared_topk = None
        self._shared_indexk = None
        self._candidates = None
        # OrderedDict = O(1) LRU. Separate protected set pins the previous
        # decode step's working set so temporal prefetch can actually hit
        # (cap must hold ~2 steps or pin is useless).
        self._expert_cache: OrderedDict[tuple[str, int, int], dict] = OrderedDict()
        self._expert_protected: set[tuple[str, int, int]] = set()
        self._expert_used_now: set[tuple[str, int, int]] = set()
        self._mtp_window: dict[int, mx.array] = {}
        # 40 layers x 6 = 240/step; protect prev step + hold current => ~480.
        self._expert_cache_cap = max(240, int(expert_cache_cap))
        self.expert_loads = 0
        self.expert_cache_hits = 0
        self.expert_evictions = 0
        # Last-step routing per layer: temporal locality for decode prefetch.
        self._last_route: dict[tuple[str, int], list[int]] = {}
        self._prefetch_pool = None
        try:
            # Dense ~18GiB + experts (cap*70.8MB) + head f32; leave KV/engram slack.
            mx.set_cache_limit(4 * 1024**3)
            mx.set_memory_limit(56 * 1024**3)
        except (RuntimeError, AttributeError):
            pass  # non-Metal backends ignore memory governance

    def W(self, name: str) -> mx.array:
        return self._dense[name]

    def shutdown(self) -> None:
        if self._prefetch_pool is not None:
            self._prefetch_pool.shutdown(wait=False)
            self._prefetch_pool = None
        self._engram_cache.shutdown()

    # -- embedding / head ------------------------------------------------
    def embed(self, ids: list[int]) -> mx.array:
        table = self._dense["embed.weight"]
        return table[mx.array(ids)]

    def head_logits(self, h: mx.array) -> mx.array:
        cfg = self._config
        normed = F.rmsnorm(h, self._dense["norm.weight"], cfg.get("norm_eps", 1e-20))
        wt = getattr(self, "_head_wt", None)
        if wt is None:
            # 1.26GiB bf16 -> f32.T once (~2.5GiB); avoids ~700ms/token cast.
            wt = self._dense["head.weight"].astype(mx.float32).T
            mx.eval(wt)
            self._head_wt = wt
            del self._dense["head.weight"]  # drop bf16 copy
            mx.clear_cache()
        return normed.astype(mx.float32) @ wt

    # -- MoE --------------------------------------------------------------
    def _expert_touch(self, key: tuple[str, int, int], weights: dict | None = None) -> dict:
        """LRU touch; optionally insert. Never evicts used_now keys."""
        cache = self._expert_cache
        if weights is not None:
            cache[key] = weights
            self._expert_used_now.add(key)
        elif key in cache:
            cache.move_to_end(key)
            self._expert_used_now.add(key)
        while len(cache) > self._expert_cache_cap:
            victim = None
            # Pass 1: unprotected, not used this step.
            for old in cache:
                if old not in self._expert_used_now and old not in self._expert_protected:
                    victim = old
                    break
            # Pass 2: protected but idle (prev-step history).
            if victim is None:
                for old in cache:
                    if old not in self._expert_used_now:
                        victim = old
                        break
            # Pass 3: everything in use — drop oldest (should be rare).
            if victim is None and cache:
                victim = next(iter(cache))
            if victim is None:
                break
            del cache[victim]
            self.expert_evictions += 1
        return cache[key]

    def _expert_begin_step(self, *, new_sequence: bool = False) -> None:
        """Start of forward: pin prev-step keys; clear pin on new sequence."""
        if new_sequence:
            self._expert_protected = set()
            self._expert_used_now = set()
            return
        if self._expert_used_now:
            self._expert_protected = set(self._expert_used_now)
        self._expert_used_now = set()

    def shrink_expert_cache(self, new_cap: int) -> int:
        """Lower the expert cache cap, evicting LRU excess. Returns evicted count."""
        new_cap = max(1, int(new_cap))
        self._expert_cache_cap = new_cap
        evicted = 0
        while len(self._expert_cache) > new_cap:
            victim = None
            for k in self._expert_cache:
                if k not in self._expert_protected:
                    victim = k
                    break
            if victim is None:
                victim = next(iter(self._expert_cache))
            del self._expert_cache[victim]
            self.expert_evictions += 1
            evicted += 1
        return evicted

    def _expert_fn(self, layer: int, ns: str = "layers"):
        def fetch(expert: int):
            key = (ns, layer, expert)
            hit = self._expert_cache.get(key)
            if hit is not None:
                self.expert_cache_hits += 1
                self._expert_touch(key)
                return hit["w1"], hit["w3"], hit["w2"]
            self.expert_loads += 1
            d = self._expert_loader.load_expert(layer, expert, prefix=ns)
            self._expert_touch(key, d)
            return d["w1"], d["w3"], d["w2"]
        return fetch

    def prefetch_experts(self, layer: int, experts, ns: str = "layers") -> None:
        """Warm the expert cache for one layer's activated experts in parallel."""
        if not experts:
            return
        missing = []
        for e in experts:
            key = (ns, layer, int(e))
            if key not in self._expert_cache:
                missing.append((layer, int(e), ns))
            else:
                self._expert_touch(key)
        if not missing:
            return
        loaded = self._expert_loader.load_experts_parallel(
            missing, workers=self._expert_prefetch_workers)
        for (prefix, ly, ex), d in loaded.items():
            key = (prefix, ly, ex)
            if key in self._expert_cache:
                self._expert_touch(key)
                continue
            self.expert_loads += 1
            self._expert_touch(key, d)

    def _shared_fn(self, layer: int, ns: str = "layers"):
        if ns == "mtp":
            prefix = f"mtp.{layer}.ffn.shared_experts"
        else:
            prefix = f"layers.{layer}.ffn.shared_experts"
        w1 = self._dense[prefix + ".w1.weight"]
        w3 = self._dense[prefix + ".w3.weight"]
        w2 = self._dense[prefix + ".w2.weight"]

        def run(x: mx.array) -> mx.array:
            # f32 x @ bf16 W: mlx upcasts W, matches explicit f32 cast (maxdiff 0)
            # and skips the 3x bf16->f32 materialization per token.
            g = mx.minimum(x.astype(mx.float32) @ w1.T, 10.0)
            u = mx.clip(x.astype(mx.float32) @ w3.T, -10.0, 10.0)
            h = (g / (1.0 + mx.exp(-g))) * u
            return (h @ w2.T).astype(x.dtype)

        return run

    def moe(self, layer: int, x: mx.array, token_start: int = 0,
            ns: str = "layers", top_k: int = 6) -> mx.array:
        base = f"mtp.{layer}" if ns == "mtp" else f"layers.{layer}"
        gate_w = self._dense[f"{base}.ffn.gate.weight"]
        gate_b = self._dense[f"{base}.ffn.gate.bias"]

        def hook(layer_id, idx, w):
            # Temporal prefetch is for decode (S=1). Prefill unions across
            # tokens can exceed the cache and thrash LRU — skip those.
            n_tok = len(idx)
            if n_tok == 1:
                self._last_route[(ns, layer_id)] = sorted(
                    {int(e) for e in idx[0]})
            # Backbone schema only records layers [0, n_layers); skip MTP drafts.
            if self._trace is not None and ns == "layers":
                self._trace.router(layer_id, token_start, idx, w)

        def prefetch(layer_id, experts):
            self.prefetch_experts(layer_id, experts, ns=ns)

        return F.moe_layer(x, gate_w, gate_b, self._expert_fn(layer, ns),
                           self._shared_fn(layer, ns), top_k, 10.0, hook, layer,
                           prefetch_fn=prefetch)

    def prefetch_last_routes(self) -> Future | None:
        """Async warm of expert cache from previous step's routing (decode).

        Returns a Future so the caller can overlap loads with embed/attention.
        Only misses are loaded; LRU already holds the previous working set.
        Multi-step unions are intentionally NOT used: they over-fetch and
        thrash the 256–400 key budget (measured loads 4896 vs 2771).
        """
        if not self._last_route:
            return None
        specs: list[tuple[int, int, str]] = []
        for (ns, layer), experts in self._last_route.items():
            for e in experts:
                key = (ns, layer, e)
                if key not in self._expert_cache:
                    specs.append((layer, e, ns))
        if not specs:
            return None
        if self._prefetch_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._prefetch_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="expert-warm")

        def _work():
            loaded = self._expert_loader.load_experts_parallel(
                specs, workers=self._expert_prefetch_workers)
            for (prefix, ly, ex), d in loaded.items():
                key = (prefix, ly, ex)
                if key in self._expert_cache:
                    self._expert_touch(key)
                    continue
                self.expert_loads += 1
                self._expert_touch(key, d)

        return self._prefetch_pool.submit(_work)

    # -- rotary -------------------------------------------------------------
    # Per-layer tables (reference Attention.__init__): layers WITH
    # compress_ratio use YaRN (original 65536, base compress_rope_theta);
    # ratio-0 layers disable YaRN (original 0, base rope_theta).
    def _freqs_for(self, layer: int, ns: str = "layers") -> mx.array:
        cfg = self._config
        key = layer if ns == "layers" else (ns, layer)
        cached = self._freqs.get(key) if isinstance(self._freqs, dict) else None
        if cached is not None:
            return cached
        if self._freqs is None or not isinstance(self._freqs, dict):
            self._freqs = {}
        # DSpark stages are always compress_ratio 0 (asserted upstream).
        if ns == "mtp":
            ratio = 0
        else:
            ratio = cfg["compress_ratios"][layer]
        if ratio:
            original, base = cfg.get("original_seq_len", 0), cfg["compress_rope_theta"]
        else:
            original, base = 0, cfg["rope_theta"]
        table = F.rope_freqs(
            cfg["rope_head_dim"], cfg.get("max_seq_len") or 65536,
            original, base,
            cfg.get("rope_factor", 40.0), cfg.get("beta_fast", 32),
            cfg.get("beta_slow", 1))
        mx.eval(table)
        self._freqs[key] = table
        return table

    # -- attention ------------------------------------------------------------
    @staticmethod
    def _window_topk(win: int, seqlen: int, start_pos: int) -> mx.array:
        """Ring slots each query attends to; -1 marks an empty/invalid slot.

        start_pos==0: causal window inside the prefill chunk (seqlen rows).
        start_pos>0, seqlen==1: whole ring, oldest first (reference decode).
        start_pos>0, seqlen>1: multi-token verify — rows index into
        concat(old_ring[win], chunk[seqlen]); history uses ring slots,
        chunk positions use win + (q - start_pos). Requires the ring to
        still hold pre-forward history (call before writing the chunk).
        """
        if start_pos == 0:
            end = np.arange(seqlen)[:, None]
            idxs = np.clip(end - win + 1, 0, None) + np.arange(min(seqlen, win))[None, :]
            return mx.array(np.where(idxs > end, -1, idxs).astype(np.int32))
        if seqlen == 1:
            oldest = start_pos % win + 1
            idxs = np.concatenate([np.arange(oldest, win), np.arange(oldest)])[None, :]
            idxs = np.where(idxs > start_pos, -1, idxs)
            return mx.array(idxs.astype(np.int32))
        # Multi-token decode: width = win (history ring) + seqlen (current chunk).
        idxs = np.full((seqlen, win + seqlen), -1, dtype=np.int32)
        for i in range(seqlen):
            p = start_pos + i
            lo = max(0, p - win + 1)
            for col, q in enumerate(range(lo, p + 1)):
                if q < start_pos:
                    idxs[i, col] = q % win
                else:
                    idxs[i, col] = win + (q - start_pos)
        return mx.array(idxs)

    def _sparse_attn(self, q: mx.array, kv: mx.array, topk_idxs: mx.array,
                     sink: mx.array, scale: float) -> mx.array:
        """Gather + online softmax + sink (sink feeds denominator only)."""
        # Stay on device: previous numpy path forced GPU->CPU every layer.
        qf = q.astype(mx.float32)
        kvf = kv.astype(mx.float32)
        sink_f = sink.astype(mx.float32)
        m, _, d = qf.shape
        idx = topk_idxs.reshape(m, -1)
        mask = idx >= 0
        safe = mx.maximum(idx, 0).astype(mx.int32)
        kg = kvf[safe.reshape(-1)].reshape(m, -1, d)  # [m, k, d]
        scores = mx.einsum("mhd,mkd->mhk", qf, kg) * scale
        scores = mx.where(mask[:, None, :], scores,
                          mx.full(scores.shape, -1e30, dtype=mx.float32))
        mrow = mx.max(scores, axis=-1, keepdims=True)
        exp_s = mx.exp(scores - mrow)
        exp_s = mx.where(mask[:, None, :], exp_s, mx.zeros_like(exp_s))
        denom = mx.sum(exp_s, axis=-1, keepdims=True) + mx.exp(
            sink_f[None, :, None] - mrow)
        out = mx.einsum("mhk,mkd->mhd", exp_s / denom, kg)
        return out.astype(q.dtype)

    def _unrotary(self, x: mx.array, freqs: mx.array) -> mx.array:
        d = x.shape[-1]
        xr = x.astype(mx.float32).reshape(*x.shape[:-1], d // 2, 2)
        s = freqs.shape[0]
        mid = [1] * (xr.ndim - 3)
        cos = mx.cos(freqs).reshape(s, *mid, d // 2)
        sin = mx.sin(freqs).reshape(s, *mid, d // 2)
        re, im = xr[..., 0], xr[..., 1]
        out = mx.stack([re * cos + im * sin, -re * sin + im * cos], axis=-1)
        return out.reshape(x.shape).astype(x.dtype)

    def _run_compressor(self, layer: int, x: mx.array, start_pos: int):
        """Decode/prefill compressor state machine; latent or None."""
        cfg = self._config
        p = f"layers.{layer}.attn."
        ratio = cfg["compress_ratios"][layer]
        head_dim = cfg["head_dim"]
        seqlen = x.shape[0]
        if ratio == 1:
            kv = x.astype(mx.float32) @ self._dense[p + "compressor.wkv.weight"].T \
                if p + "compressor.wkv.weight" in self._dense else \
                x.astype(mx.float32) @ self._dense[p + "wkv.weight"].T
            norm_w = (self._dense[p + "compressor.norm.weight"]
                      if p + "compressor.norm.weight" in self._dense
                      else self._dense[p + "kv_norm.weight"])
            return F.rmsnorm(kv, norm_w, 1e-20)
        wkv = self._dense[p + "compressor.wkv.weight"]
        wgate = self._dense[p + "compressor.wgate.weight"]
        xf = x.astype(mx.float32)
        kv, score = xf @ wkv.T, xf @ wgate.T
        st = self._kv_state.get(layer)
        sc = self._score_state.get(layer)
        if st is None:
            st = mx.zeros((ratio, head_dim))
            sc = mx.full((ratio, head_dim), float("-inf"))
            mx.eval(st, sc)
            self._kv_state[layer], self._score_state[layer] = st, sc
        if start_pos == 0:
            if seqlen >= ratio:
                rem = seqlen % ratio
                cut = seqlen - rem
                if rem:
                    st = mx.concatenate([kv[cut:], st[rem:]], axis=0)
                    sc = mx.concatenate([score[cut:], sc[rem:]], axis=0)
                    self._kv_state[layer], self._score_state[layer] = st, sc
                kv = kv[:cut].reshape(-1, ratio, head_dim)
                score = score[:cut].reshape(-1, ratio, head_dim)
                kv = (kv * mx.softmax(score, axis=1)).sum(axis=1)
            else:
                st = mx.concatenate([kv, st[seqlen:]], axis=0)
                sc = mx.concatenate([score, sc[seqlen:]], axis=0)
                self._kv_state[layer], self._score_state[layer] = st, sc
                return None
        elif seqlen == 1:
            slot = start_pos % ratio
            st = mx.concatenate([st[:slot], kv[:1], st[slot + 1:]], axis=0)
            sc = mx.concatenate([sc[:slot], score[:1], sc[slot + 1:]], axis=0)
            self._kv_state[layer], self._score_state[layer] = st, sc
            if (start_pos + 1) % ratio != 0:
                return None
            kv = (st * mx.softmax(sc, axis=0)).sum(axis=0, keepdims=True)
        else:
            # Multi-token decode: walk each absolute position through the ring state.
            latents = []
            for i in range(seqlen):
                pos = start_pos + i
                slot = pos % ratio
                st = mx.concatenate([st[:slot], kv[i:i + 1], st[slot + 1:]], axis=0)
                sc = mx.concatenate([sc[:slot], score[i:i + 1], sc[slot + 1:]], axis=0)
                if (pos + 1) % ratio == 0:
                    latents.append((st * mx.softmax(sc, axis=0)).sum(axis=0, keepdims=True))
            self._kv_state[layer], self._score_state[layer] = st, sc
            if not latents:
                return None
            kv = mx.concatenate(latents, axis=0)
        return F.rmsnorm(kv, self._dense[p + "compressor.norm.weight"], 1e-20)

    def _select_candidates(self, index_score: mx.array, compress_lens,
                           topk_blocks: int, block_size: int) -> mx.array:
        """Level-one block mask (exact port of select_candidate_blocks)."""
        logits = np.asarray(index_score)
        width = logits.shape[-1]
        pad = (-width) % block_size
        padded = np.pad(logits, ((0, 0), (0, pad)), constant_values=-np.inf) if pad else logits
        scores = padded.reshape(logits.shape[0], -1, block_size).max(axis=-1)
        num_blocks = scores.shape[-1]
        keep = np.zeros_like(scores, dtype=bool)
        # pin newest block per query + top scoring reachable blocks
        if np.ndim(compress_lens) == 0:
            last = (int(compress_lens) - 1) // block_size
            scores[:, last] = np.inf  # consume one topk slot, like the reference
            keep[:, last] = True
        else:
            for qi, cl in enumerate(np.asarray(compress_lens).ravel()):
                keep[qi, (int(cl) - 1) // block_size] = True
        k = min(topk_blocks, num_blocks)
        for qi in range(scores.shape[0]):
            order = np.argsort(-scores[qi], kind="stable")[:k]
            for b in order:
                if scores[qi, b] > -np.inf:
                    keep[qi, b] = True
        mask = np.repeat(keep, block_size, axis=-1)[:, :width]
        return mx.array(mask)

    def _run_indexer(self, layer: int, x: mx.array, qr: mx.array, latent,
                     start_pos: int, seqlen: int) -> mx.array:
        """Indexer topk for index_source layers; others read shared topk.

        Non-owning indexers score against the shared index keys published
        by the last owning layer (exact reference behavior).
        """
        cfg = self._config
        p = f"layers.{layer}.attn."
        ratio = cfg["compress_ratios"][layer]
        n_heads, index_dim = cfg["index_n_heads"], cfg["index_head_dim"]
        rd = cfg["rope_head_dim"]
        owns_k = layer in cfg["kv_source_layers"]
        freqs_all = self._freqs_for(layer)
        if owns_k and latent is not None:
            k = F.rmsnorm(
                latent.astype(mx.float32)
                @ self._dense[p + "indexer.wk.weight"].astype(mx.float32).T,
                self._dense[p + "indexer.k_norm.weight"], 1e-20)
            n_lat = k.shape[0]
            if start_pos == 0:
                kfreq = freqs_all[: n_lat * ratio:ratio][:n_lat]
            elif ratio == 1:
                kfreq = freqs_all[start_pos:start_pos + n_lat]
            else:
                rows = [pos + 1 - ratio for pos in range(start_pos, start_pos + seqlen)
                        if (pos + 1) % ratio == 0]
                if len(rows) != n_lat:
                    raise RuntimeError(
                        f"index k rows {n_lat} vs completions {len(rows)}")
                kfreq = freqs_all[rows]
            k = mx.concatenate([k[..., :-rd], F.apply_rotary(k[..., -rd:], kfreq)], axis=-1)
            k_cache = self._indexk.get(layer)
            pos = start_pos // max(ratio, 1)
            need = pos + n_lat
            if k_cache is None:
                k_cache = mx.zeros((need + 64, index_dim))
                mx.eval(k_cache)
            elif k_cache.shape[0] < need:
                k_cache = mx.concatenate(
                    [k_cache, mx.zeros((need - k_cache.shape[0] + 64, index_dim))], axis=0)
            k_cache = k_cache.at[pos:pos + n_lat].add(k - k_cache[pos:pos + n_lat])
            self._indexk[layer] = k_cache
            self._shared_indexk = k_cache
        q = (qr @ self._dense[p + "indexer.wq_b.weight"].astype(mx.float32).T).reshape(
            seqlen, n_heads, index_dim)
        q = mx.concatenate([q[..., :-rd], F.apply_rotary(
            q[..., -rd:], self._freqs_for(layer)[start_pos:start_pos + seqlen])], axis=-1)
        end_pos = start_pos + seqlen
        index_k = self._shared_indexk
        if index_k is None:
            empty = mx.full((seqlen, 0), -1, dtype=mx.int32)
            self._shared_topk = empty
            return empty
        index_k = index_k[: end_pos // max(ratio, 1)]
        weights = (x.astype(mx.float32)
                   @ self._dense[p + "indexer.weights_proj.weight"].astype(mx.float32).T)
        weights = weights * (index_dim**-0.5 * n_heads**-0.5)
        # GPU score: avoid per-layer GPU->CPU sync of full [S,H,T] tensor.
        score_mx = mx.einsum("shd,td->sht", q.astype(mx.float32),
                             index_k.astype(mx.float32))
        score_mx = mx.maximum(score_mx, 0) * weights[:, :, None]
        score = np.einsum("sht->st", np.asarray(score_mx))
        if start_pos == 0:
            cl = (np.arange(1, seqlen + 1) // ratio)[:, None]
            score = np.where(np.arange(score.shape[1]) >= cl, -np.inf, score)
            compress_lens = cl
        else:
            # Per-query: query i at absolute pos start_pos+i sees (pos+1)//ratio groups.
            cl = ((np.arange(start_pos, start_pos + seqlen) + 1)
                  // max(ratio, 1))[:, None]
            score = np.where(np.arange(score.shape[1]) >= cl, -np.inf, score)
            compress_lens = cl
        if layer == cfg.get("candidate_source_layer", -1):
            self._candidates = self._select_candidates(
                mx.array(score), compress_lens, cfg["candidate_topk_blocks"],
                cfg["candidate_block_size"])
        elif cfg.get("candidate_source_layer", -1) >= 0 and layer > cfg["candidate_source_layer"]:
            mask = np.asarray(self._candidates)
            if mask.shape != score.shape:
                raise RuntimeError(
                    f"candidate mask {mask.shape} vs score {score.shape} "
                    f"at layer {layer} start_pos {start_pos} seqlen {seqlen}")
            score = np.where(mask, score, -np.inf)
        topk = min(cfg["index_topk"], end_pos // max(ratio, 1))
        if topk <= 0:
            empty = mx.full((seqlen, 0), -1, dtype=mx.int32)
            self._shared_topk = empty
            return empty
        idx = np.argsort(-score, axis=-1, kind="stable")[:, :topk]
        idx = np.sort(idx, axis=-1)
        if start_pos == 0:
            cl = (np.arange(1, seqlen + 1) // ratio)
            out = np.where(idx < cl[:, None], idx, -1)
        else:
            out = np.where(idx < compress_lens, idx, -1)
        out_mx = mx.array(out.astype(np.int32))
        self._shared_topk = out_mx
        return out_mx

    def indexer_topk(self, layer: int, x: mx.array, qr: mx.array, latent,
                     start_pos: int, seqlen: int) -> mx.array:
        """Shared-topk dispatch: only index sources compute; rest reuse."""
        if layer not in self._config["index_source_layers"]:
            if self._shared_topk is None:
                raise RuntimeError("no shared topk published yet")
            return self._shared_topk
        return self._run_indexer(layer, x, qr, latent, start_pos, seqlen)

    def attention(self, layer: int, x: mx.array, start_pos: int) -> mx.array:
        """Exact sparse path: window ring + compress + indexer topk + sink."""
        cfg = self._config
        p = f"layers.{layer}.attn."
        n_heads, head_dim = cfg["n_heads"], cfg["head_dim"]
        rd = cfg["rope_head_dim"]
        win = cfg["window_size"]
        ratio = cfg["compress_ratios"][layer]
        seqlen = x.shape[0]
        freqs = self._freqs_for(layer)[start_pos:start_pos + seqlen]

        qr = F.rmsnorm(x.astype(mx.float32) @ self._dense[p + "wq_a.weight"].astype(mx.float32).T,
                       self._dense[p + "q_norm.weight"], 1e-20)
        q = (qr @ self._dense[p + "wq_b.weight"].astype(mx.float32).T).reshape(
            seqlen, n_heads, head_dim)
        q = mx.concatenate([q[..., :-rd], F.apply_rotary(q[..., -rd:], freqs)], axis=-1)

        kv_new = x.astype(mx.float32) @ self._dense[p + "wkv.weight"].astype(mx.float32).T
        kv_new = mx.concatenate(
            [kv_new[..., :-rd], F.apply_rotary(kv_new[..., -rd:], freqs)], axis=-1)
        ring = self._window.get(layer)
        if ring is None:
            ring = mx.zeros((win, head_dim), dtype=mx.bfloat16)
            mx.eval(ring)
        if start_pos == 0:
            if seqlen <= win:
                ring = ring.at[:seqlen].add(kv_new.astype(mx.bfloat16) - ring[:seqlen])
            else:
                cutoff = seqlen % win
                tail_new = kv_new[-win:].astype(mx.bfloat16)
                ring = (mx.concatenate([tail_new[win - cutoff:], tail_new[:win - cutoff]], axis=0)
                        if cutoff else tail_new)
            window_kv = kv_new
            idxs = self._window_topk(win, seqlen, 0)
        elif seqlen == 1:
            ring = ring.at[start_pos % win].add(
                kv_new[0].astype(mx.bfloat16) - ring[start_pos % win])
            window_kv = ring
            idxs = self._window_topk(win, 1, start_pos)
        else:
            # Multi-token verify: attend over pre-forward ring + chunk (causal),
            # then commit the chunk into the ring for subsequent steps.
            kv_bf = kv_new.astype(mx.bfloat16)
            idxs = self._window_topk(win, seqlen, start_pos)
            window_kv = mx.concatenate([ring, kv_bf], axis=0)
            for i in range(seqlen):
                slot = (start_pos + i) % win
                ring = ring.at[slot].add(kv_bf[i] - ring[slot])
        self._window[layer] = ring

        topk_all = idxs
        ck = None
        if ratio:
            compress_len = (start_pos + seqlen) // ratio
            latent = None
            if layer in cfg["kv_source_layers"]:
                latent = self._run_compressor(layer, x, start_pos)
                if latent is not None:
                    ck = self._register_compress(layer, self._rotate_latent(
                        layer, latent, ratio, start_pos, seqlen), ratio)
                    self._shared_compress = ck
                own = self._compress.get(layer)
                if own is None:
                    own = mx.zeros((0, cfg["head_dim"]))
                    mx.eval(own)
                self._shared_compress = own
            ck = self._shared_compress
            cidx = self.indexer_topk(layer, x, qr, latent, start_pos, seqlen)
            if ck is not None:
                ck = ck[:compress_len]
                window_kv = mx.concatenate([window_kv, ck], axis=0)
                topk_all = self._shift_compress_idxs(
                    idxs, cidx, int(window_kv.shape[0]) - int(ck.shape[0]))
        sink = self._dense[p + "attn_sink"].astype(mx.float32)
        n_compress = int(ck.shape[0]) if ratio and ck is not None else 0
        out = self._sparse_attn(q, window_kv, topk_all, sink, head_dim**-0.5)
        if self._trace is not None:
            self._trace.kv(layer, start_pos, n_compress,
                           int(np.asarray(topk_all).shape[-1]))
        out = mx.concatenate([out[..., :-rd], self._unrotary(out[..., -rd:], freqs)], axis=-1)
        o_groups = cfg.get("o_groups", 8)
        o_lora = cfg.get("o_lora_rank", 1024)
        wo_a = self._dense[p + "wo_a.weight"].reshape(o_groups, o_lora, -1)
        og = out.reshape(seqlen, o_groups, -1)
        og = mx.einsum("sgd,grd->sgr", og.astype(mx.float32), wo_a.astype(mx.float32))
        return (og.reshape(seqlen, -1).astype(mx.float32)
                @ self._dense[p + "wo_b.weight"].astype(mx.float32).T)

    @staticmethod
    def _shift_compress_idxs(idxs: mx.array, cidx: mx.array, base: int) -> mx.array:
        left = np.asarray(idxs, dtype=np.int64)
        right = np.asarray(cidx, dtype=np.int64)
        right = np.where(right >= 0, right + base, -1)
        return mx.array(np.concatenate([left, right], axis=-1).astype(np.int32))

    def _rotate_latent(self, layer: int, latent: mx.array, ratio: int, start_pos: int, seqlen: int) -> mx.array:
        """Rotate each latent row with its group-start position frequencies."""
        rd = self._config["rope_head_dim"]
        freqs_all = self._freqs_for(layer)
        n = latent.shape[0]
        if start_pos == 0:
            rows = np.minimum(np.arange(n) * ratio, freqs_all.shape[0] - 1)
            freqs = mx.array(np.asarray(freqs_all)[rows])
        elif ratio == 1:
            freqs = freqs_all[start_pos:start_pos + n]
        else:
            rows = [pos + 1 - ratio for pos in range(start_pos, start_pos + seqlen)
                    if (pos + 1) % ratio == 0]
            if len(rows) != n:
                raise RuntimeError(
                    f"latent rows {n} vs completions {len(rows)} "
                    f"start_pos={start_pos} seqlen={seqlen} ratio={ratio}")
            freqs = mx.array(np.asarray(freqs_all)[rows])
        tail = F.apply_rotary(latent[..., -rd:], freqs)
        return mx.concatenate([latent[..., :-rd], tail], axis=-1)

    def _register_compress(self, layer: int, latent: mx.array, ratio: int) -> mx.array:
        cache = self._compress.get(layer)
        new = latent.astype(mx.bfloat16)
        if cache is None:
            cache = new
        else:
            cache = mx.concatenate([cache, new], axis=0)
        self._compress[layer] = cache
        mx.eval(cache)
        return cache

    # -- engram -----------------------------------------------------------------
    def _engram_hash_ids(self, token_ids: list[int], start_pos: int):
        """Exact upstream hashes (torch CPU): [L, n_engram_layers, n_cols]."""
        import sys as _sys

        snap = str(Path(__file__).resolve().parents[2] / "upstream"
                   / "DeepSeek-V4.1-Flash" / "inference")
        if snap not in _sys.path:
            _sys.path.insert(0, snap)
        import engram as _up
        import torch as _torch

        if not hasattr(self, "_hash_state"):
            from types import SimpleNamespace

            cfg = self._config
            args = SimpleNamespace(
                engram_layer_ids=cfg["engram_layer_ids"],
                engram_max_ngram_size=cfg["engram_max_ngram_size"],
                engram_n_heads=cfg["engram_n_heads"],
                engram_head_dim=cfg["engram_head_dim"],
                engram_vocab_size=cfg["engram_vocab_size"],
                engram_num_embeddings=cfg["engram_num_embeddings"],
                engram_compressed_vocab_size=cfg["engram_compressed_vocab_size"],
                engram_pad_id=cfg["engram_pad_id"],
                max_batch_size=1,
                max_seq_len=cfg.get("max_seq_len", 65536),
            )
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(self._root))
            layout = _up.EngramLayout.from_args(args)
            self._hash_state = _up.NgramHashState(args, layout, tok)
            self._hash_ncols = (cfg["engram_max_ngram_size"] - 1) * cfg["engram_n_heads"]
        ids = _torch.tensor([token_ids])
        with _torch.inference_mode():
            hashes = self._hash_state(ids, start_pos)
        return np.asarray(hashes[0]).reshape(len(token_ids), -1, self._hash_ncols)

    def _engram_row(self, layer: int, row_id: int) -> np.ndarray:
        """Dequantized embed row (256 fp8 + 8 ue8m0 scales) as float32."""
        from deepseeker.mlx_weights import E4M3_TABLE, decode_u8m0

        rows = self._engram_cache.fetch_rows(layer, [int(row_id)])
        w_raw = np.frombuffer(rows[0][0], dtype=np.uint8)
        s_raw = np.frombuffer(rows[0][1], dtype=np.uint8)
        return (E4M3_TABLE[w_raw.astype(np.int64)] * np.repeat(
            decode_u8m0(s_raw), 32)).astype(np.float32)

    def engram_forward(self, layer: int, start_pos: int, x: mx.array, hash_ids: np.ndarray) -> mx.array:
        """Exact Engram math: embed rows -> wkv -> gated add into stream."""
        cfg = self._config
        dim, hc = cfg["dim"], cfg["hc_mult"]
        rows = np.stack([[self._engram_row(layer, r) for r in pos] for pos in hash_ids])
        kv = rows.reshape(rows.shape[0], -1).astype(np.float32)
        kv = kv @ np.asarray(self._dense[f"layers.{layer}.engram.wkv.weight"].astype(mx.float32),
                             dtype=np.float32).T
        key, value = kv[..., : hc * dim], kv[..., hc * dim:]
        key = key.reshape(key.shape[0], hc, dim)
        q_w = np.asarray(self._dense[f"layers.{layer}.engram.q_weight"].astype(mx.float32))
        k_w = np.asarray(self._dense[f"layers.{layer}.engram.k_weight"].astype(mx.float32))
        weight = q_w * k_w
        h = np.asarray(x.astype(mx.float32))
        eps = cfg.get("norm_eps", 1e-20)
        rstd = 1 / np.sqrt((h ** 2).mean(-1) + eps) * \
            1 / np.sqrt((key ** 2).mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
        gate = 1 / (1 + np.exp(-np.copysign(np.clip(np.abs(dot), 1e-6, None) ** 0.5, dot)))
        out = h + gate[..., None] * value.reshape(value.shape[0], 1, dim)
        if self._trace is not None:
            self._trace.engram(layer, start_pos, hash_ids.tolist())
        return mx.array(out.astype(np.float32)).astype(x.dtype)

    # -- block + transformer ------------------------------------------------------
    def block(self, layer: int, x: mx.array, start_pos: int, pre_mix: mx.array,
              ns: str = "layers", main_x: mx.array | None = None,
              top_k: int = 6) -> tuple[mx.array, mx.array]:
        """One Block: hc mixes -> attn -> hc_post -> hc mixes -> MoE -> hc_post.

        ns='mtp' uses mtp.{layer}. weights; main_x routes to DSparkAttention.
        """
        cfg = self._config
        p = (f"mtp.{layer}." if ns == "mtp" else f"layers.{layer}.")
        hc = cfg["hc_mult"]
        eps = cfg.get("norm_eps", 1e-20)
        iters = int(cfg.get("hc_sinkhorn_iters", 20))
        hc_eps = cfg.get("hc_eps", 1e-6)
        attn_pre, attn_post, attn_comb = F.hc_mixes(
            x, self._dense[p + "hc_attn_fn"],
            self._dense[p + "hc_attn_scale"], self._dense[p + "hc_attn_base"],
            hc, eps, iters, hc_eps)
        xin = F.hc_pre(x, pre_mix)
        xin = F.rmsnorm(xin, self._dense[p + "attn_norm.weight"], cfg.get("norm_eps", 1e-20))
        if ns == "mtp":
            xa = self.dspark_attention(layer, xin, start_pos, main_x)
        else:
            xa = self.attention(layer, xin, start_pos)
        x = F.hc_post(xa, x, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = F.hc_mixes(
            x, self._dense[p + "hc_ffn_fn"],
            self._dense[p + "hc_ffn_scale"], self._dense[p + "hc_ffn_base"],
            hc, eps, iters, hc_eps)
        xin = F.hc_pre(x, attn_pre)
        xin = F.rmsnorm(xin, self._dense[p + "ffn_norm.weight"], cfg.get("norm_eps", 1e-20))
        xf = self.moe(layer, xin, start_pos, ns=ns, top_k=top_k)
        x = F.hc_post(xf, x, ffn_post, ffn_comb)
        return x, ffn_pre

    def forward(self, token_ids: list[int], start_pos: int,
                want_main: bool = False) -> mx.array | tuple[mx.array, mx.array]:
        """Text path: embed -> hc expand -> [engram] 40 blocks -> norm -> head.

        want_main also returns cat of target-layer (37,38,39) mean-pooled hiddens
        for the DSpark MTP draft head (#29).
        """
        cfg = self._config
        hc = cfg["hc_mult"]
        self._expert_begin_step(new_sequence=(start_pos == 0))
        warm_future = None
        if start_pos == 0:
            # New sequence: drop temporal map so we don't prefetch stale keys.
            self._last_route.clear()
        elif self._last_route:
            warm_future = self.prefetch_last_routes()
        h = self.embed(token_ids)
        h = mx.broadcast_to(h[:, None, :], (h.shape[0], hc, h.shape[1]))
        pre_mix = mx.zeros((h.shape[0], hc))
        pre_mix = pre_mix.at[:, 0].add(1.0)  # identity pre-mix, not uniform
        mx.eval(h, pre_mix)
        hashes = None
        if cfg.get("engram_layer_ids"):
            hashes = self._engram_hash_ids(token_ids, start_pos)
        targets = set(cfg.get("dspark_target_layer_ids") or ())
        if warm_future is not None:
            warm_future.result()  # block only once, after embed/engram setup
        main_parts: list[mx.array] = []
        for layer in range(cfg["n_layers"]):
            if hashes is not None and layer in cfg["engram_layer_ids"]:
                li = cfg["engram_layer_ids"].index(layer)
                h = self.engram_forward(layer, start_pos, h, hashes[:, li, :])
            if want_main and layer in targets:
                main_parts.append(mx.mean(h, axis=1))  # [S, dim]
            h, pre_mix = self.block(layer, h, start_pos, pre_mix)
        final = F.hc_pre(h, pre_mix)
        logits = self.head_logits(final)
        if want_main:
            main_hidden = mx.concatenate(main_parts, axis=-1)  # [S, 3*dim]
            return logits, main_hidden
        return logits

    def dspark_attention(self, stage: int, x: mx.array, start_pos: int,
                         main_x: mx.array) -> mx.array:
        """DSparkAttention: window ring seeded by main_kv; draft queries only.

        compress_ratio == 0 (window-only). start_pos==0 seeds the ring from
        main_x and returns x unchanged (prefill path).
        """
        cfg = self._config
        p = f"mtp.{stage}.attn."
        n_heads, head_dim = cfg["n_heads"], cfg["head_dim"]
        rd = cfg["rope_head_dim"]
        win = cfg["window_size"]
        freqs_m = self._freqs_for(stage, "mtp")[start_pos:start_pos + main_x.shape[0]]
        # main_kv from main_x (projected target hidden), roped at main positions
        mkv = main_x.astype(mx.float32) @ self._dense[p + "wkv.weight"].astype(mx.float32).T
        mkv = mx.concatenate(
            [mkv[..., :-rd], F.apply_rotary(mkv[..., -rd:], freqs_m)], axis=-1)
        ring = self._mtp_window.get(stage)
        if ring is None:
            ring = mx.zeros((win, head_dim), dtype=mx.bfloat16)
            mx.eval(ring)
        if start_pos == 0:
            seqlen = int(main_x.shape[0])
            if seqlen <= win:
                ring = ring.at[:seqlen].add(mkv.astype(mx.bfloat16) - ring[:seqlen])
            else:
                cutoff = seqlen % win
                tail = mkv[-win:].astype(mx.bfloat16)
                ring = (mx.concatenate([tail[win - cutoff:], tail[:win - cutoff]], axis=0)
                        if cutoff else tail)
            self._mtp_window[stage] = ring
            return x
        block = int(x.shape[0])
        seqlen = int(main_x.shape[0])
        freqs_d = self._freqs_for(stage, "mtp")[start_pos + seqlen:start_pos + seqlen + block]
        qr = F.rmsnorm(x.astype(mx.float32) @ self._dense[p + "wq_a.weight"].astype(mx.float32).T,
                       self._dense[p + "q_norm.weight"], 1e-20)
        q = (qr @ self._dense[p + "wq_b.weight"].astype(mx.float32).T).reshape(
            block, n_heads, head_dim)
        q = mx.concatenate([q[..., :-rd], F.apply_rotary(q[..., -rd:], freqs_d)], axis=-1)
        kv_new = x.astype(mx.float32) @ self._dense[p + "wkv.weight"].astype(mx.float32).T
        kv_new = mx.concatenate(
            [kv_new[..., :-rd], F.apply_rotary(kv_new[..., -rd:], freqs_d)], axis=-1)
        # write single main_kv row (decode seqlen==1) into ring at start_pos
        ring = ring.at[start_pos % win].add(
            mkv[0].astype(mx.bfloat16) - ring[start_pos % win])
        self._mtp_window[stage] = ring
        hist = min(win, start_pos + 1)
        kv_all = mx.concatenate([ring, kv_new.astype(mx.bfloat16)], axis=0)
        base = np.concatenate([np.arange(hist), win + np.arange(block)])
        idxs = np.tile(base.astype(np.int32), (block, 1))
        sink = self._dense[p + "attn_sink"].astype(mx.float32)
        out = self._sparse_attn(q, kv_all, mx.array(idxs), sink, head_dim**-0.5)
        out = mx.concatenate([out[..., :-rd], self._unrotary(out[..., -rd:], freqs_d)], axis=-1)
        o_groups = cfg.get("o_groups", 8)
        o_lora = cfg.get("o_lora_rank", 1024)
        wo_a = self._dense[p + "wo_a.weight"].reshape(o_groups, o_lora, -1)
        og = out.reshape(block, o_groups, -1)
        og = mx.einsum("sgd,grd->sgr", og.astype(mx.float32), wo_a.astype(mx.float32))
        return (og.reshape(block, -1).astype(mx.float32)
                @ self._dense[p + "wo_b.weight"].astype(mx.float32).T)

    def _snapshot_caches(self) -> dict:
        return {
            "window": dict(self._window),
            "compress": dict(self._compress),
            "indexk": dict(self._indexk),
            "kv_state": dict(self._kv_state),
            "score_state": dict(self._score_state),
            "shared_compress": self._shared_compress,
            "shared_topk": self._shared_topk,
            "shared_indexk": self._shared_indexk,
            "candidates": self._candidates,
            "mtp_window": dict(self._mtp_window),
        }

    def _restore_caches(self, snap: dict) -> None:
        self._window = dict(snap["window"])
        self._compress = dict(snap["compress"])
        self._indexk = dict(snap["indexk"])
        self._kv_state = dict(snap["kv_state"])
        self._score_state = dict(snap["score_state"])
        self._shared_compress = snap["shared_compress"]
        self._shared_topk = snap["shared_topk"]
        self._shared_indexk = snap["shared_indexk"]
        self._candidates = snap["candidates"]
        self._mtp_window = dict(snap["mtp_window"])

    @staticmethod
    def _sample(last: np.ndarray, temperature: float) -> int:
        if temperature == 0:
            return int(np.argmax(last))
        probs = np.exp(last / max(temperature, 1e-5))
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def mtp_seed(self, main_hidden: mx.array) -> None:
        """Prefill: seed each MTP stage window from main_hidden (start_pos==0)."""
        cfg = self._config
        w_proj = self._dense["mtp.0.main_proj.weight"]
        main_x = main_hidden.astype(mx.float32) @ w_proj.astype(mx.float32).T
        main_x = F.rmsnorm(main_x, self._dense["mtp.0.main_norm.weight"],
                           cfg.get("norm_eps", 1e-20))
        dummy = mx.zeros((main_x.shape[0], cfg["dim"]))
        for stage in range(int(cfg.get("n_mtp_layers", 0))):
            self.dspark_attention(stage, dummy, 0, main_x)
            mx.eval(self._mtp_window[stage])

    def mtp_draft(self, main_hidden: mx.array, input_ids: list[int],
                  start_pos: int) -> list[int]:
        """DSpark draft: propose block_size tokens after input_ids.

        main_hidden: [1, 3*dim] from the target forward that produced the
        logits for input_ids (reference forward_spec).
        Returns output_ids[1:] (block_size draft tokens).
        """
        cfg = self._config
        hc = cfg["hc_mult"]
        B = int(cfg.get("dspark_block_size", 5))
        noise = int(cfg.get("dspark_noise_token_id", 0))
        n_mtp = int(cfg.get("n_mtp_layers", 0))
        if n_mtp < 1 or B < 1 or start_pos <= 0:
            return []
        w_proj = self._dense["mtp.0.main_proj.weight"]
        main_x = main_hidden.astype(mx.float32) @ w_proj.astype(mx.float32).T
        main_x = F.rmsnorm(main_x, self._dense["mtp.0.main_norm.weight"],
                           cfg.get("norm_eps", 1e-20))  # [1, dim]
        draft_ids = [noise] * B
        draft_ids[0] = int(input_ids[0])
        h = self.embed(draft_ids)  # [B, dim]
        h = mx.broadcast_to(h[:, None, :], (B, hc, h.shape[1]))
        pre_mix = mx.zeros((B, hc))
        pre_mix = pre_mix.at[:, 0].add(1.0)
        mx.eval(h, pre_mix, main_x)
        top_k = int(cfg.get("dspark_n_activated_experts", 3))
        for stage in range(n_mtp):
            h, pre_mix = self.block(
                stage, h, start_pos, pre_mix, ns="mtp",
                main_x=main_x, top_k=top_k)
        # forward_head: hc_pre -> mtp.2.norm -> shared head + markov bias
        x = F.hc_pre(h, pre_mix)  # [B, dim]
        x = F.rmsnorm(x, self._dense["mtp.2.norm.weight"],
                      cfg.get("norm_eps", 1e-20))
        wt = getattr(self, "_head_wt", None)
        if wt is None:
            wt = self._dense["head.weight"].astype(mx.float32).T
            mx.eval(wt)
            self._head_wt = wt
            if "head.weight" in self._dense:
                del self._dense["head.weight"]
            mx.clear_cache()
        logits = x.astype(mx.float32) @ wt
        emb_w = self._dense["mtp.2.markov_head.embed.weight"]  # [V, R]
        head_w = self._dense["mtp.2.markov_head.head.weight"]  # [V, R]
        out0 = int(input_ids[0])
        drafts: list[int] = []
        cur = out0
        for i in range(B):
            e = emb_w[cur]  # [R]
            bias = e.astype(mx.float32) @ head_w.astype(mx.float32).T  # [V]
            row = np.asarray(logits[i], dtype=np.float32) + np.asarray(bias, dtype=np.float32)
            # temperature 0 (greedy draft) matches baseline generate default
            nxt = int(np.argmax(row))
            drafts.append(nxt)
            cur = nxt
        return drafts

    def generate_spec(self, prompt_ids: list[int], max_new_tokens: int,
                      temperature: float = 0.0) -> dict:
        """Lossless greedy DSpark speculative decode (#29).

        Draft block_size tokens; verify with one multi-token target forward;
        on partial reject restore caches and re-forward accepted prefix +
        the target's correction token. Acceptance check: drafts[j] against
        argmax(v_logits[j]) (prediction for position L+1+j after chunk j).
        """
        ids = list(prompt_ids)
        times: dict = {}
        stats = {
            "proposed": 0, "accepted": 0, "steps": 0,
            "draft_s": 0.0, "verify_s": 0.0, "target_s": 0.0,
            "rollbacks": 0, "full_accepts": 0,
        }
        start = time.perf_counter()
        logits, main_hidden = self.forward(ids, 0, want_main=True)
        mx.eval(logits, main_hidden)
        times["prefill_s"] = time.perf_counter() - start
        self.mtp_seed(main_hidden)

        out_tokens: list[int] = []
        nxt = self._sample(np.asarray(logits[-1], dtype=np.float32), temperature)
        out_tokens.append(nxt)
        ids.append(nxt)
        if self._trace is not None:
            self._trace.token(nxt)
        tt = time.perf_counter()
        logits, main_hidden = self.forward([nxt], len(ids) - 1, want_main=True)
        mx.eval(logits, main_hidden)
        stats["target_s"] += time.perf_counter() - tt
        decode_s = 0.0

        while len(out_tokens) < max_new_tokens:
            step_t0 = time.perf_counter()
            pos = len(ids) - 1
            t0 = self._sample(np.asarray(logits[-1], dtype=np.float32), temperature)
            td = time.perf_counter()
            drafts = self.mtp_draft(main_hidden, [t0], start_pos=pos)
            stats["draft_s"] += time.perf_counter() - td
            stats["proposed"] += len(drafts)
            if not drafts:
                out_tokens.append(t0)
                ids.append(t0)
                if self._trace is not None:
                    self._trace.token(t0)
                tt = time.perf_counter()
                logits, main_hidden = self.forward([t0], len(ids) - 1, want_main=True)
                mx.eval(logits, main_hidden)
                stats["target_s"] += time.perf_counter() - tt
                stats["steps"] += 1
                decode_s += time.perf_counter() - step_t0
                continue

            # Verify [t0] + drafts as one multi-token target forward at pos+1.
            # v_logits[j] predicts position (pos+1)+(j+1) after consuming chunk j;
            # drafts[j] occupies position pos+1+j and must match argmax(v_logits[j])
            # only for j>=1 relative to after t0... drafts[j] is at index j+1 in chunks,
            # prediction after chunks[j] (index j) is v_logits[j] for position pos+1+j+?:
            # chunk positions: [t0, d0, d1, ...] at L=pos+1.
            # v_logits[0] = pred for L+1 after t0  -> should equal d0
            # v_logits[j] = pred for L+1+j after chunks[j]  -> should equal drafts[j]
            chunks = [t0] + drafts
            L = pos + 1
            snap = self._snapshot_caches()
            tv = time.perf_counter()
            v_logits, v_main = self.forward(chunks, L, want_main=True)
            mx.eval(v_logits, v_main)
            stats["verify_s"] += time.perf_counter() - tv

            # t0 is the target's own sample — always accepted.
            # drafts[0]=d0 must match v_logits[0], drafts[j] match v_logits[j].
            matched_drafts = 0
            for j, d in enumerate(drafts):
                pred = int(np.argmax(np.asarray(v_logits[j], dtype=np.float32)))
                if d == pred:
                    matched_drafts += 1
                else:
                    break
            stats["accepted"] += matched_drafts
            accepted = 1 + matched_drafts  # including t0

            if accepted == len(chunks):
                stats["full_accepts"] += 1
                out_tokens.extend(chunks)
                ids.extend(chunks)
                if len(out_tokens) > max_new_tokens:
                    out_tokens = out_tokens[:max_new_tokens]
                    ids = ids[:len(prompt_ids) + max_new_tokens]
                if self._trace is not None:
                    for t in out_tokens[-len(chunks):]:
                        self._trace.token(t)
                main_hidden = v_main[-1:]
                logits = v_logits
            else:
                # First bad token is drafts[matched_drafts] at position L+matched_drafts
                # (= position of chunks[accepted]). Prediction after chunks[accepted-1]
                # is v_logits[accepted-1].
                correction = int(np.argmax(
                    np.asarray(v_logits[accepted - 1], dtype=np.float32)))
                commit = chunks[:accepted] + [correction]
                self._restore_caches(snap)
                stats["rollbacks"] += 1
                tt = time.perf_counter()
                logits, main_hidden = self.forward(commit, L, want_main=True)
                mx.eval(logits, main_hidden)
                stats["target_s"] += time.perf_counter() - tt
                out_tokens.extend(commit)
                ids.extend(commit)
                if self._trace is not None:
                    for t in commit:
                        self._trace.token(t)
                if len(out_tokens) > max_new_tokens:
                    out_tokens = out_tokens[:max_new_tokens]
                    ids = ids[:len(prompt_ids) + max_new_tokens]

            stats["steps"] += 1
            decode_s += time.perf_counter() - step_t0

        times["decode_s"] = decode_s
        times["tokens"] = len(out_tokens)
        times["tok_s"] = len(out_tokens) / decode_s if decode_s else 0.0
        acc = stats["accepted"]
        prop = stats["proposed"]
        stats["acceptance"] = acc / prop if prop else 0.0
        return {"tokens": out_tokens, "timings": times, "spec": stats}

    def generate(self, prompt_ids: list[int], max_new_tokens: int,
                 temperature: float = 0.0) -> dict:
        """Prefill + greedy/sampled decode with timing + trace events."""
        ids = list(prompt_ids)
        times = {}
        start = time.perf_counter()
        logits = self.forward(ids, 0)
        mx.eval(logits)
        times["prefill_s"] = time.perf_counter() - start
        out_tokens = []
        decode_s = 0.0
        for _ in range(max_new_tokens):
            last = np.asarray(logits[-1], dtype=np.float32)
            if temperature == 0:
                nxt = int(np.argmax(last))
            else:
                probs = np.exp(last / max(temperature, 1e-5))
                probs /= probs.sum()
                nxt = int(np.random.choice(len(probs), p=probs))
            out_tokens.append(nxt)
            ids.append(nxt)
            step = time.perf_counter()
            logits = self.forward([nxt], len(ids) - 1)
            mx.eval(logits)
            decode_s += time.perf_counter() - step
            if self._trace is not None:
                self._trace.token(nxt)
        times["decode_s"] = decode_s
        times["tokens"] = len(out_tokens)
        times["tok_s"] = len(out_tokens) / decode_s if decode_s else 0.0
        times["expert_loads"] = self.expert_loads
        times["expert_cache_hits"] = self.expert_cache_hits
        times["expert_evictions"] = self.expert_evictions
        total = self.expert_loads + self.expert_cache_hits
        times["expert_hit_rate"] = (
            self.expert_cache_hits / total if total else 0.0)
        return {"tokens": out_tokens, "timings": times}
