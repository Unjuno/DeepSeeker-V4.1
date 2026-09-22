"""MLX-native DeepSeek-V4.1-Flash runner (roadmap: runner epic).

Batch 1, text-only decode (+prefill), exact port of inference/model.py
math (components in mlx_forward). All tensors [S, ...] (batch squeezed).

Weight flow: dense bf16 resident (~15GB) via DenseLoader; routed
experts streamed per token via ExpertLoader (#33 dequant); engram
embed rows streamed via EngramRowCache (#26).

Trace hooks record authoritative routing/KV/Engram events (#20) with
zero effect on numerics. Deviations: bf16 caches (not fp8-quantized),
no distributed, no vision/DSpark/MTP.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from deepseeker import mlx_forward as F
from deepseeker.engram_cache import EngramRowCache
from deepseeker.mlx_weights import DenseLoader, ExpertLoader


class Runner:
    """Owns weights, caches, and streaming loaders for one session."""

    def __init__(self, model_root: Path | str, manifest: dict, config: dict,
                 trace_hook=None) -> None:
        self._root = Path(model_root)
        self._manifest = manifest
        self._config = config
        self._trace = trace_hook
        self._expert_loader = ExpertLoader(model_root, manifest)
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
        self._expert_cache: dict[tuple[int, int], dict] = {}
        self._expert_lru: list[tuple[int, int]] = []
        self._expert_cache_cap = 64  # experts; ~4.5GB bf16 working set
        self.expert_loads = 0
        self.expert_cache_hits = 0
        self.expert_evictions = 0
        try:
            mx.set_cache_limit(6 * 1024**3)
            mx.set_memory_limit(48 * 1024**3)
        except (RuntimeError, AttributeError):
            pass  # non-Metal backends ignore memory governance

    def W(self, name: str) -> mx.array:
        return self._dense[name]

    def shutdown(self) -> None:
        self._engram_cache.shutdown()

    # -- embedding / head ------------------------------------------------
    def embed(self, ids: list[int]) -> mx.array:
        table = next(v for k, v in self._dense.items() if k.endswith("embed.weight")
                     and "engram" not in k)
        return table[mx.array(ids)]

    def head_logits(self, h: mx.array) -> mx.array:
        cfg = self._config
        normed = F.rmsnorm(h, self._dense["norm.weight"], cfg.get("norm_eps", 1e-20))
        head_w = self._dense["head.weight"]
        return normed.astype(mx.float32) @ head_w.astype(mx.float32).T

    # -- MoE --------------------------------------------------------------
    def _expert_fn(self, layer: int):
        def fetch(expert: int):
            key = (layer, expert)
            hit = self._expert_cache.get(key)
            if hit is not None:
                self.expert_cache_hits += 1
                try:
                    self._expert_lru.remove(key)
                except ValueError:
                    pass
                self._expert_lru.append(key)
                return hit["w1"], hit["w3"], hit["w2"]
            self.expert_loads += 1
            d = self._expert_loader.load_expert(layer, expert)
            self._expert_cache[key] = d
            self._expert_lru.append(key)
            while len(self._expert_cache) > self._expert_cache_cap:
                old = self._expert_lru.pop(0)
                if old in self._expert_cache:
                    del self._expert_cache[old]
                    self.expert_evictions += 1
            return d["w1"], d["w3"], d["w2"]
        return fetch

    def _shared_fn(self, layer: int):
        prefix = f"layers.{layer}.ffn.shared_experts"
        w1 = self._dense[prefix + ".w1.weight"]
        w3 = self._dense[prefix + ".w3.weight"]
        w2 = self._dense[prefix + ".w2.weight"]

        def run(x: mx.array) -> mx.array:
            g = mx.minimum(x.astype(mx.float32) @ w1.astype(mx.float32).T, 10.0)
            u = mx.clip(x.astype(mx.float32) @ w3.astype(mx.float32).T, -10.0, 10.0)
            h = (g / (1.0 + mx.exp(-g))) * u
            return (h @ w2.astype(mx.float32).T).astype(x.dtype)

        return run

    def moe(self, layer: int, x: mx.array) -> mx.array:
        gate_w = self._dense[f"layers.{layer}.ffn.gate.weight"]
        gate_b = self._dense[f"layers.{layer}.ffn.gate.bias"]

        def hook(layer_id, idx, w):
            if self._trace is not None:
                self._trace.router(layer_id, idx, w)

        return F.moe_layer(x, gate_w, gate_b, self._expert_fn(layer),
                           self._shared_fn(layer), 6, 10.0, hook, layer)

    # -- rotary -------------------------------------------------------------
    def _freqs_for(self, length: int) -> mx.array:
        cfg = self._config
        if self._freqs is None:
            self._freqs = F.rope_freqs(
                cfg["rope_head_dim"], cfg.get("max_seq_len") or 65536,
                cfg.get("original_seq_len", 0), cfg["rope_theta"],
                cfg.get("rope_factor", 40.0), cfg.get("beta_fast", 32),
                cfg.get("beta_slow", 1))
            mx.eval(self._freqs)
        return self._freqs

    # -- attention ------------------------------------------------------------
    @staticmethod
    def _window_topk(win: int, seqlen: int, start_pos: int) -> mx.array:
        if start_pos == 0:
            end = np.arange(seqlen)[:, None]
            idxs = np.clip(end - win + 1, 0, None) + np.arange(min(seqlen, win))[None, :]
            idxs = np.where(idxs > end, -1, idxs)
        else:
            oldest = start_pos % win + 1
            idxs = np.concatenate([np.arange(oldest, win), np.arange(oldest)])[None, :]
            idxs = np.where(idxs > start_pos, -1, idxs)
        return mx.array(idxs.astype(np.int32))

    def _sparse_attn(self, q: mx.array, kv: mx.array, topk_idxs: mx.array,
                     sink: mx.array, scale: float) -> mx.array:
        """Gather + online softmax + sink (sink feeds denominator only)."""
        kvf = np.asarray(kv.astype(mx.float32))
        qf = np.asarray(q.astype(mx.float32))
        sink_np = np.asarray(sink.astype(mx.float32))
        m, h, d = qf.shape
        idx = np.asarray(topk_idxs, dtype=np.int64).reshape(m, -1)
        outs = []
        for qi in range(m):
            rows = idx[qi]
            valid = rows[rows >= 0]
            if valid.size == 0:
                outs.append(np.zeros((h, d), dtype=np.float32))
                continue
            k = kvf[valid]
            scores = np.einsum("hd,kd->hk", qf[qi], k) * scale
            mrow = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - mrow)
            denom = exp_s.sum(axis=1, keepdims=True) + np.exp(sink_np[:, None] - mrow)
            outs.append((exp_s / denom) @ k)
        return mx.array(np.stack(outs, axis=0)).astype(q.dtype)

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
            kv = x.astype(mx.float32) @ self._dense[p + "compressor.wkv.weight"].astype(mx.float32).T \
                if p + "compressor.wkv.weight" in self._dense else \
                x.astype(mx.float32) @ self._dense[p + "wkv.weight"].astype(mx.float32).T
            norm_w = (self._dense[p + "compressor.norm.weight"]
                      if p + "compressor.norm.weight" in self._dense
                      else self._dense[p + "kv_norm.weight"])
            return F.rmsnorm(kv, norm_w, 1e-20)
        wkv = self._dense[p + "compressor.wkv.weight"]
        wgate = self._dense[p + "compressor.wgate.weight"]
        xf = x.astype(mx.float32)
        kv, score = xf @ wkv.astype(mx.float32).T, xf @ wgate.astype(mx.float32).T
        if start_pos == 0:
            if seqlen >= ratio:
                rem = seqlen % ratio
                cut = seqlen - rem
                if rem:
                    self._kv_state[layer] = kv[cut:]
                    self._score_state[layer] = score[cut:]
                kv = kv[:cut].reshape(-1, ratio, head_dim)
                score = score[:cut].reshape(-1, ratio, head_dim)
                kv = (kv * mx.softmax(score, axis=1)).sum(axis=1)
            else:
                self._kv_state[layer] = kv
                self._score_state[layer] = score
                return None
        else:
            slot = start_pos % ratio
            st = self._kv_state.get(layer)
            sc = self._score_state.get(layer)
            if st is None:
                st = mx.zeros((ratio, head_dim))
                sc = mx.full((ratio, head_dim), float("-inf"))
                mx.eval(st, sc)
            st = st.at[slot].add(kv[0] - st[slot])
            sc = sc.at[slot].add(score[0] - sc[slot])
            self._kv_state[layer], self._score_state[layer] = st, sc
            if (start_pos + 1) % ratio != 0:
                return None
            kv = (st * mx.softmax(sc, axis=0)).sum(axis=0, keepdims=True)
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
        freqs_all = self._freqs_for(1)
        if owns_k and latent is not None:
            k = F.rmsnorm(
                latent.astype(mx.float32)
                @ self._dense[p + "indexer.wk.weight"].astype(mx.float32).T,
                self._dense[p + "indexer.k_norm.weight"], 1e-20)
            n_lat = k.shape[0]
            if start_pos == 0:
                kfreq = freqs_all[: n_lat * ratio:ratio][:n_lat]
            else:
                kfreq = freqs_all[start_pos + 1 - ratio:start_pos + 2 - ratio]
            k = mx.concatenate([k[..., :-rd], F.apply_rotary(k[..., -rd:], kfreq)], axis=-1)
            k_cache = self._indexk.get(layer)
            pos = start_pos // max(ratio, 1)
            if k_cache is None:
                k_cache = mx.zeros((pos + n_lat + 64, index_dim))
                mx.eval(k_cache)
            k_cache = k_cache.at[pos:pos + n_lat].add(k - k_cache[pos:pos + n_lat])
            self._indexk[layer] = k_cache
            self._shared_indexk = k_cache
        q = (qr @ self._dense[p + "indexer.wq_b.weight"].astype(mx.float32).T).reshape(
            seqlen, n_heads, index_dim)
        q = mx.concatenate([q[..., :-rd], F.apply_rotary(
            q[..., -rd:], self._freqs_for(1)[start_pos:start_pos + seqlen])], axis=-1)
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
        score = np.einsum("shd,td->sht", np.asarray(q.astype(mx.float32)),
                          np.asarray(index_k.astype(mx.float32)))
        score = np.maximum(score, 0) * np.asarray(weights)[:, :, None]
        score = score.sum(axis=1)
        if start_pos == 0:
            cl = (np.arange(1, seqlen + 1) // ratio)[:, None]
            score = np.where(np.arange(score.shape[1]) >= cl, -np.inf, score)
            compress_lens = cl
        else:
            compress_lens = end_pos // max(ratio, 1)
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
        freqs = self._freqs_for(1)[start_pos:start_pos + seqlen]

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
        else:
            ring = ring.at[start_pos % win].add(
                kv_new[0].astype(mx.bfloat16) - ring[start_pos % win])
            window_kv = ring
            idxs = self._window_topk(win, 1, start_pos)
        self._window[layer] = ring

        topk_all = idxs
        if ratio:
            latent = None
            if layer in cfg["kv_source_layers"]:
                latent = self._run_compressor(layer, x, start_pos)
                if latent is not None:
                    ck = self._register_compress(layer, self._rotate_latent(
                        latent, ratio, start_pos, seqlen), ratio)
                    self._shared_compress = ck
            ck = self._shared_compress
            cidx = self.indexer_topk(layer, x, qr, latent, start_pos, seqlen)
            if ck is not None:
                window_kv = mx.concatenate([window_kv, ck], axis=0)
                topk_all = self._shift_compress_idxs(
                    idxs, cidx, int(window_kv.shape[0]) - int(ck.shape[0]))
        sink = self._dense[p + "attn_sink"].astype(mx.float32)
        out = self._sparse_attn(q, window_kv, topk_all, sink, head_dim**-0.5)
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

    def _rotate_latent(self, latent: mx.array, ratio: int, start_pos: int, seqlen: int) -> mx.array:
        """Rotate each latent row with its group-start position frequencies."""
        rd = self._config["rope_head_dim"]
        freqs_all = self._freqs_for(1)
        n = latent.shape[0]
        if start_pos == 0:
            rows = np.minimum(np.arange(n) * ratio, freqs_all.shape[0] - 1)
            freqs = mx.array(np.asarray(freqs_all)[rows])
        else:
            freqs = freqs_all[start_pos + 1 - ratio:start_pos + 2 - ratio]
            freqs = mx.broadcast_to(freqs, (n, freqs.shape[-1]))
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

    def engram_forward(self, layer: int, x: mx.array, hash_ids: np.ndarray) -> mx.array:
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
            self._trace.engram(layer, hash_ids.tolist())
        return mx.array(out.astype(np.float32)).astype(x.dtype)

    # -- block + transformer ------------------------------------------------------
    def block(self, layer: int, x: mx.array, start_pos: int, pre_mix: mx.array) -> tuple[mx.array, mx.array]:
        """One Block: hc mixes -> attn -> hc_post -> hc mixes -> MoE -> hc_post."""
        cfg = self._config
        p = f"layers.{layer}."
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
        xa = self.attention(layer, xin, start_pos)
        x = F.hc_post(xa, x, attn_post, attn_comb)
        ffn_pre, ffn_post, ffn_comb = F.hc_mixes(
            x, self._dense[p + "hc_ffn_fn"],
            self._dense[p + "hc_ffn_scale"], self._dense[p + "hc_ffn_base"],
            hc, eps, iters, hc_eps)
        xin = F.hc_pre(x, attn_pre)
        xin = F.rmsnorm(xin, self._dense[p + "ffn_norm.weight"], cfg.get("norm_eps", 1e-20))
        xf = self.moe(layer, xin)
        x = F.hc_post(xf, x, ffn_post, ffn_comb)
        return x, ffn_pre

    def forward(self, token_ids: list[int], start_pos: int) -> mx.array:
        """Text path: embed -> hc expand -> [engram] 40 blocks -> norm -> head."""
        cfg = self._config
        hc = cfg["hc_mult"]
        h = self.embed(token_ids)  # [S, dim]
        h = mx.broadcast_to(h[:, None, :], (h.shape[0], hc, h.shape[1]))
        pre_mix = mx.zeros((h.shape[0], hc))
        pre_mix = pre_mix.at[:, 0].add(1.0)  # identity pre-mix, not uniform
        mx.eval(h, pre_mix)
        hashes = None
        if cfg.get("engram_layer_ids"):
            hashes = self._engram_hash_ids(token_ids, start_pos)
        for layer in range(cfg["n_layers"]):
            if hashes is not None and layer in cfg["engram_layer_ids"]:
                li = cfg["engram_layer_ids"].index(layer)
                h = self.engram_forward(layer, h, hashes[:, li, :])
            h, pre_mix = self.block(layer, h, start_pos, pre_mix)
        final = F.hc_pre(h, pre_mix)
        return self.head_logits(final)

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
        return {"tokens": out_tokens, "timings": times}
