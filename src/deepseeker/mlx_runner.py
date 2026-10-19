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

import os
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from deepseeker import mlx_forward as F
from deepseeker.cache_policy import expert_lru_budget
from deepseeker.engram_cache import EngramRowCache
from deepseeker.mlx_weights import DenseLoader, ExpertLoader

# Steps whose device-side argmax output is held before one mx.eval drains them.
# See generate_batch(): too large re-creates the superlinear single-drain cost,
# too small loses the amortization of the completion wait.
DRAIN_EVERY = 128


def _np_f32(a) -> np.ndarray:
    """numpy float32 view of an MLX array.

    ``np.asarray`` on an MLX bfloat16 array raises ``ValueError: 'bfloat16'
    is not a valid PEP 3118 buffer format string``, so the cast has to go
    through MLX first. Matters once the LM head runs in bf16 (--head-bf16).
    """
    if isinstance(a, mx.array):
        a = a.astype(mx.float32)
    return np.asarray(a, dtype=np.float32)


class Runner:
    """Owns weights, caches, and streaming loaders for one session."""

    def __init__(
        self,
        model_root: Path | str,
        manifest: dict,
        config: dict,
        trace_hook=None,
        expert_cache_cap: int = 256,
        expert_prefill_cache_cap: int = 0,
        expert_prefetch_workers: int = 8,
        expert_policy: str = "lru",
        expert_pin: dict[int, list[int]] | None = None,
        expert_pin_gather: bool = False,
        max_batch: int = 1,
        batch_prefetch: bool = False,
        union_budget: int = 0,
        union_strict: bool = False,
        top_k: int = 0,
        no_shared: bool = False,
        hc_iters: int = 0,
        window: int = 0,
        attn_bf16: bool = False,
        skip_every: int = 0,
        skip_block: int = 0,
        no_engram: bool = False,
        head_bf16: bool = False,
        proj_bf16: bool = False,
        keep_layers: int = -1,
        head_quant: bool = False,
        head_vocab: int = 0,
        scale_inv_head: bool = True,
        head_compose: bool = True,
        step_cpu: bool = True,
        moe_pipeline: bool = False,
        indexer_gpu_topk: bool = False,
    ) -> None:
        self._root = Path(model_root)
        # Decode drops rmsnorm's row reduction (see head_logits): rstd is a
        # positive per-row scalar and cancels under argmax, so tokens are
        # unchanged while a whole kernel leaves the step. Only the low-precision
        # head paths (bf16 / 8-bit) can disagree, because scaling the input
        # changes their rounding -- measured 0/512 positions on an f32 head,
        # 10/512 on bf16, 27/512 on the 8-bit head. Turn off for exact logits.
        self._scale_inv_head = bool(scale_inv_head)
        # keep=0 decode is memoryless, so embed -> norm scale -> head are three
        # constants in a row and collapse into one [n_ids, n_head] table. See
        # _composed_head. Inert unless --head-vocab bounds the head.
        self._head_compose = bool(head_compose)
        self._head_mt: mx.array | bool | None = None
        # The composed step is a gather of 32 rows plus an argmax over
        # head_vocab columns: ~2us in numpy against ~13us for the same two
        # Metal dispatches, because the work is latency-bound and the table
        # fits in cache. See _step_device_cpu in generate_batch.
        self._step_cpu = bool(step_cpu)
        self._head_mt_np: np.ndarray | None = None
        # Opt-in async MoE pipeline (experimental, reversible): overlap SSD
        # expert loads with GPU qmms within one layer. Default off; enable
        # with --moe-pipeline for A/B. See F.moe_layer pipeline branch.
        self._moe_pipeline = bool(moe_pipeline)
        # Experimental quality-preserving pre-attention expert prefetch.
        # The checkpoint router remains authoritative; this only reads the
        # pre-attention gate's top-1 candidate ahead of attention.
        self._pre_attn_prefetch_enabled = os.environ.get("DEEPSEEKER_PREATTN_PREFETCH") == "1"
        self._pre_attn_top_k = max(1, int(os.environ.get("DEEPSEEKER_PREATTN_TOPK", "1")))
        self._pre_attn_pending: dict[tuple[str, int], dict] = {}
        self._pre_attn_predictions = 0
        self._pre_attn_cache_hits = 0
        self._pre_attn_correct = 0
        self._pre_attn_wrong = 0
        self._pre_attn_reads = 0
        self._pre_attn_used = 0
        self._pre_attn_used_bytes = 0
        self._pre_attn_wasted = 0
        self._pre_attn_wasted_bytes = 0
        self._pre_attn_read_worker_s = 0.0
        self._pre_attn_pack_worker_s = 0.0
        # Opt-in GPU indexer Top-K (experimental, reversible): for ordinary
        # decode index-source layers, keep score reduction/masking/Top-K on
        # device instead of transferring the full score tensor to NumPy.
        # Candidate block selection still materializes scores on CPU. Default
        # off; enable with --indexer-gpu-topk for A/B. Speed unmeasured.
        self._indexer_gpu_topk = bool(indexer_gpu_topk)
        self._batch_prefetch = bool(batch_prefetch)
        # Per-layer cap on the batch's expert union (see F.cap_union). 0 = exact.
        self._union_budget = int(union_budget)
        self._union_strict = bool(union_strict)
        # Activated experts per token. 0 = config value (n_activated_experts).
        # Lowering it cuts both route compute and expert bytes proportionally.
        self._top_k = int(top_k) if top_k else int(config.get("n_activated_experts", 6))
        # Speed mode: drop the always-on shared expert (costs output quality).
        self._no_shared = bool(no_shared)
        # 0 = keep the checkpoint's hc_sinkhorn_iters (exact routing maths).
        self._hc_iters = int(hc_iters)
        # 0 = keep the checkpoint's window_size (exact attention maths).
        self._window_size = int(window)
        # Speed mode: run the sparse attention core in bf16 (costs numerics).
        self._attn_bf16 = bool(attn_bf16)
        # 0 = exact (every layer runs its FFN). N>0 = bypass the whole MoE
        # sublayer on layers where layer % N == 0 (attention still runs).
        self._skip_every = int(skip_every)
        # 0 = exact. N>0 = bypass the whole Block on layers where
        # layer % N == 0: no hc mix, no attention, no MoE, no weight reads.
        self._skip_block = int(skip_block)
        # Speed mode: drop the Engram memory path entirely (row fetch, fp8
        # dequant, wkv and gate). Measured as the single largest remaining
        # phase -- f_eng_rows was 9.88s of a 17.61s decode at batch 124 --
        # because the 384M-row tables make every fetch a cold random SSD read
        # with zero page reuse.
        self._no_engram = bool(no_engram)
        # Speed mode: keep the LM head in bf16 instead of materialising it as
        # f32. The head is 5120x~130k, so f32 is 2.66GiB re-read every step
        # through an f32 GEMM that misses the tensor cores -- f_head was 1.14s
        # of a 4.45s decode (26%) at batch 124. Costs numerics.
        self._head_bf16 = bool(head_bf16)
        # Speed mode: run the wq_a / wkv contractions in bf16. They are the
        # only two f32 GEMMs left in the attention path (wq_b, wo_a, wo_b
        # are already bf16), and Apple's AMX only accelerates bf16 matmul.
        # Costs numerics.
        self._proj_bf16 = bool(proj_bf16)
        # -1: keep every layer the skip-block guard protects (the default).
        # N >= 0: keep only the first N index-source layers; 0 skips the whole
        # stack and prices out the residual embed/head cost.
        self._keep_layers = int(keep_layers)
        # Speed mode: store the LM head as 8-bit affine groups instead of
        # bf16. The head GEMM is bandwidth-bound, not compute-bound, once the
        # batch is small (M=32: 13.8ms reads of a 1.32GiB weight, ~0.8 TFLOP
        # of work), so the win is purely "less bytes to read". Scales/biases
        # at gs=128 add ~22MB; total 683MB vs 1324MB -> 8.0ms (1.73x). The
        # kernel wants the weight contiguous in [V, K] with transpose=True --
        # quantizing the transposed [K, V] view instead is 0.5x as fast.
        # Costs numerics.
        self._head_quant = bool(head_quant)
        self._head_q: tuple | None = None
        # Speed mode: keep only the first N rows of the output embedding, i.e.
        # restrict generation to the N lowest token ids (BPE ids are merge
        # rank, so this is roughly the N commonest tokens). The head GEMM is
        # 2*M*K*V and V is 129280 -- it is 90% of a keep=0 decode, so the
        # win is linear in V: 129280 -> 16384 takes f_head 8.20ms -> 1.37ms.
        # 0 = full vocab. Costs the tail of the vocabulary outright.
        self._head_vocab = int(head_vocab)
        self._route_stats: dict[str, float] = {"coverage": 0.0, "calls": 0.0}
        self._manifest = manifest
        self._config = config
        self._trace = trace_hook
        self._att_stats: dict | None = None
        self._expert_loader = ExpertLoader(model_root, manifest)
        # Opt-in packed representation for the exact single-row gate/up qmm.
        # The expert cache stores w13+w2 only; batch rows use zero-copy w1/w3
        # views so their existing separate-qmm arithmetic is retained.
        self._expert_gateup_stack = os.environ.get("DEEPSEEKER_GATEUP_STACK") == "1"
        # DEEPSEEKER_FAKE_EXPERTS=1 serves every routed expert from one
        # resident stand-in: identical GPU GEMM/weight traffic, zero SSD.
        # Used to split decode into its SSD and GPU floors.
        self._fake_experts = os.environ.get("DEEPSEEKER_FAKE_EXPERTS") == "1"
        self._route_pred = None
        if os.environ.get("DEEPSEEKER_ROUTE_PRED") == "1":
            from deepseeker.route_pred import RoutePred

            self._route_pred = RoutePred()
        self._fake_w: tuple | None = None
        self._expert_prefetch_workers = expert_prefetch_workers
        self._expert_policy = expert_policy if expert_policy in ("lru", "lfu") else "lru"
        self._expert_freq: dict[tuple[str, int, int], int] = {}
        self._expert_pinned: set[tuple[str, int, int]] = {
            ("layers", int(ly), int(e)) for ly, es in (expert_pin or {}).items() for e in es
        }
        self._expert_pin_gather = bool(expert_pin_gather)
        if self._expert_pin_gather and not self._expert_pinned:
            raise ValueError("expert_pin_gather requires at least one pinned expert")
        self._expert_pin_banks: dict[int, dict] = {}
        self._pin_resident = 0
        print("loading dense weights...", flush=True)
        self._dense = DenseLoader(model_root, manifest).load_all()
        print(
            f"dense resident: {sum(a.nbytes for a in self._dense.values()) / 2**30:.1f} GiB",
            flush=True,
        )
        # Dequant allocates a transient twin of every dense tensor, so the
        # allocator retires ~28 GiB into its free list. Left unreturned the OS
        # loses its page cache (sysfree drops to ~1 GiB) and every expert
        # re-read falls back to SSD: measured 4.0x slower decode for the exact
        # same byte count. Hand it back before the first real allocation wave.
        self._trim_metal_pool(force=True)
        from deepseeker.engram import table_refs_from_manifest

        self._refs = table_refs_from_manifest(manifest)
        # The lookup tables are ~101GiB per layer (384M rows), so rows are
        # cold random SSD reads: 4 workers serialized them into 1.6s/decode.
        # DEEPSEEKER_ENGRAM_DIRECT=1 swaps the page/ThreadPool machinery for a
        # resident mmap (see EngramRowCache._fetch_direct).
        self._engram_cache = EngramRowCache(
            model_root,
            self._refs,
            int(os.environ.get("DEEPSEEKER_ENGRAM_CACHE_MB", "256")) * 1024**2,
            4096,
            max_workers=int(os.environ.get("DEEPSEEKER_ENGRAM_WORKERS", "16")),
            direct=os.environ.get("DEEPSEEKER_ENGRAM_DIRECT", "0") == "1",
        )
        self._engram_np: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._engram_mx: dict[int, mx.array] = {}
        self._engram_wkv_mx: dict[int, mx.array] = {}
        self._engram_s = 0.0
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
        # Resident f32 copies of weights the graph consumes as f32
        # (see _f32_weight): one dequant/cast per build, originals dropped.
        self._fused: dict[str, mx.array] = {}
        # OrderedDict = O(1) LRU. Separate protected set pins the previous
        # decode step's working set so temporal prefetch can actually hit
        # (cap must hold ~2 steps or pin is useless).
        self._expert_cache: OrderedDict[tuple[str, int, int], dict] = OrderedDict()
        self._expert_protected: set[tuple[str, int, int]] = set()
        self._expert_used_now: set[tuple[str, int, int]] = set()
        self._mtp_window: dict[int, mx.array] = {}
        # Lockstep batch decode: _batch > 0 means every leading axis is a
        # separate sequence all sitting on the same absolute position.
        self._batch = 0
        self._hash_row = 0
        self._hash_bsz = max(1, int(max_batch))
        # With no static pins, 40 layers x 6 routes needs at least 240 cache
        # entries. When pins are configured, the caller may have subtracted
        # them from a total cache budget; honor that reserved LRU remainder.
        self._expert_cache_cap = expert_lru_budget(
            expert_cache_cap, has_static_pins=bool(self._expert_pinned)
        )
        self._expert_decode_cache_cap = self._expert_cache_cap
        self._expert_prefill_cache_cap = max(
            self._expert_decode_cache_cap, int(expert_prefill_cache_cap or 0)
        )
        self.expert_loads = 0
        self.expert_cache_hits = 0
        self.expert_evictions = 0
        # Last-step routing per layer: temporal locality for decode prefetch.
        self._last_route: dict[tuple[str, int], list[int]] = {}
        self._prefetch_pool = None
        try:
            # Dense ~18GiB + experts (cap*70.8MB) + head f32; leave KV/engram slack.
            # 2GiB is the measured optimum: 3-4GiB retention pushes free RAM to
            # 0 during decode and the resulting reclaim stalls cost 2.7x (0.337->0.136).
            mx.set_cache_limit(int(os.environ.get("DEEPSEEKER_MLX_POOL_GB", "2")) * 1024**3)
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
    def embed(self, ids: list[int] | mx.array) -> mx.array:
        table = self._dense["embed.weight"]
        # The device-resident decode loop hands over an mx.array so the argmax
        # never has to come back to the host; only the host path pays mx.array().
        return table[ids if isinstance(ids, mx.array) else mx.array(ids)]

    def _head_q_matmul(self, normed: mx.array) -> mx.array:
        """8-bit affine LM head. ``normed`` stays f32; the kernel takes it."""
        q = self._head_q
        if q is None:
            w = self._dense.pop("head.weight", None)
            if w is None:
                # A bf16/f32 head_logits call already consumed the raw copy.
                wt = getattr(self, "_head_wt_bf", None)
                if wt is None:
                    wt = getattr(self, "_head_wt", None)
                if wt is None:
                    raise RuntimeError("--head-quant: head.weight unavailable")
                w = wt.T
            if self._head_vocab:
                w = w[: self._head_vocab]
            q = mx.quantize(w.astype(mx.bfloat16), group_size=128, bits=8, mode="affine")
            mx.eval(*q)
            self._head_q = q
            del w
            mx.clear_cache()
        w, sc, bi = q
        return mx.quantized_matmul(
            normed, w, sc, bi, transpose=True, group_size=128, bits=8, mode="affine"
        )

    _COMPOSE_MAX_VOCAB = 4096

    def _composed_head(self) -> mx.array | None:
        """[n_ids, n_head] table equal to head_logits(embed(id)), or None.

        With every block skipped the decode step reads nothing but the current
        token, so the map token -> logits is a fixed matrix: (head_w * w) @ E.T
        transposed. Materialising it once turns the step into one gather --
        10.8us against 48.8us for the expanded gather + norm-multiply + GEMM,
        because the expanded gather walks a 1.3 GiB embedding table while the
        composed table for a bounded head is a few MiB.

        The head stays at full precision on this path, so --head-quant does not
        apply. Returns None when disabled, when the head is the full vocabulary
        (the table would be 66 GiB), or when the raw head weight is already gone.
        """
        if not self._head_compose:
            return None
        if self._head_mt is None:
            self._head_mt = self._build_composed_head()
        return self._head_mt if isinstance(self._head_mt, mx.array) else None

    def _build_composed_head(self) -> mx.array | bool:
        hv = int(self._head_vocab)
        if hv <= 0 or hv > self._COMPOSE_MAX_VOCAB:
            return False
        E = self._dense.get("embed.weight")
        hw = self._dense.get("head.weight")
        w = self._dense.get("norm.weight")
        if E is None or hw is None or w is None:
            return False
        # A is the whole linear map the head applies: head weights scaled by
        # the rmsnorm weight. Embedding the product with A.T gives mt directly.
        A = hw[:hv].astype(mx.float32) * w.astype(mx.float32)[None, :]
        # In row blocks, evaluated one at a time. Upcasting the full embedding
        # in one go is 2.65 GiB on top of the weights, and under memory
        # pressure that was getting the process jetsam-killed mid-load.
        n = E.shape[0]
        blocks: list[mx.array] = []
        for i in range(0, n, 16384):
            blk = E[i : i + 16384].astype(mx.float32) @ A.T
            mx.eval(blk)
            blocks.append(blk)
        mt = mx.concatenate(blocks, axis=0)
        mx.eval(mt)
        del blocks, A
        mx.clear_cache()
        if self._step_cpu:
            self._head_mt_np = np.array(mt)  # same bits; host step reads these
        return mt

    def head_logits(self, h: mx.array, *, scale_invariant: bool = False) -> mx.array:
        cfg = self._config
        if scale_invariant:
            # Argmax-only consumer. rmsnorm is (h * rstd * weight), and rstd is
            # one positive scalar per row, so dividing a whole row of logits by
            # it cannot change which column wins -- it cancels exactly under
            # argmax. Keep the per-column weight (that one does not cancel) and
            # drop the row reduction, which measured about half of a keep=0
            # decode step. Raw logit magnitudes come back scaled per row, so
            # callers that need true probabilities must stay off this path.
            w = getattr(self, "_norm_w_f32", None)
            if w is None:
                w = self._dense["norm.weight"].astype(mx.float32)
                mx.eval(w)
                self._norm_w_f32 = w
            normed = h * w
        else:
            normed = F.rmsnorm(h, self._dense["norm.weight"], cfg.get("norm_eps", 1e-20))
        return self._head_mm(normed)

    def _head_mm(self, normed: mx.array) -> mx.array:
        """Head projection alone (norm already applied by the caller)."""
        if self._head_quant:
            return self._head_q_matmul(normed)
        if self._head_bf16:
            wt = getattr(self, "_head_wt_bf", None)
            if wt is None:
                wt = self._dense["head.weight"]
                if self._head_vocab:
                    wt = wt[: self._head_vocab]
                wt = wt.T
                mx.eval(wt)
                self._head_wt_bf = wt
                del self._dense["head.weight"]  # drop untransposed copy
                mx.clear_cache()
            return normed.astype(mx.bfloat16) @ wt
        wt = getattr(self, "_head_wt", None)
        if wt is None:
            # 1.26GiB bf16 -> f32.T once (~2.5GiB); avoids ~700ms/token cast.
            wt = self._dense["head.weight"]
            if self._head_vocab:
                wt = wt[: self._head_vocab]
            wt = wt.astype(mx.float32).T
            mx.eval(wt)
            self._head_wt = wt
            del self._dense["head.weight"]  # drop bf16 copy
            mx.clear_cache()
        return normed.astype(mx.float32) @ wt

    # -- MoE --------------------------------------------------------------
    def _lfu_victim(
        self, cache, keep: tuple[str, int, int] | None = None
    ) -> tuple[str, int, int] | None:
        """Lowest activation count among eligible keys; ties go to oldest."""
        freq = self._expert_freq
        for pass_id in (0, 1, 2):
            best = None
            best_f = None
            for k in cache:
                if k == keep:
                    continue
                if pass_id == 0 and (k in self._expert_used_now or k in self._expert_protected):
                    continue
                if pass_id == 1 and k in self._expert_used_now:
                    continue
                f = freq.get(k, 0)
                if best_f is None or f < best_f:
                    best, best_f = k, f
            if best is not None:
                return best
        return None

    def _expert_touch(
        self, key: tuple[str, int, int], weights: dict | None = None, *, stack_gateup: bool = False
    ) -> dict:
        """Touch/insert an expert.

        ``_expert_cache_cap`` budgets NON-pinned entries only; statically
        pinned hot experts stay resident for the whole run.
        """
        cache = self._expert_cache
        self._expert_freq[key] = self._expert_freq.get(key, 0) + 1
        is_pin = key in self._expert_pinned
        if weights is not None:
            if stack_gateup and "w13" not in weights and "w1" in weights and "w3" in weights:
                weights = F.stack_gate_up_expert(weights)
                mx.eval(*weights["w13"])
            if key not in cache and is_pin:
                self._pin_resident += 1
            cache[key] = weights
            self._expert_used_now.add(key)
        elif key in cache:
            cache.move_to_end(key)
            self._expert_used_now.add(key)
        lfu = self._expert_policy == "lfu"
        pinned = self._expert_pinned
        while len(cache) - self._pin_resident > self._expert_cache_cap:
            if lfu:
                victim = self._lfu_victim(cache, keep=key)
            else:
                victim = None
                # Pass 1: unprotected, not used this step, not pinned.
                for old in cache:
                    if (
                        old not in self._expert_used_now
                        and old not in self._expert_protected
                        and old not in pinned
                    ):
                        victim = old
                        break
                # Pass 2: protected but idle (prev-step history); still not pinned.
                if victim is None:
                    for old in cache:
                        if old not in self._expert_used_now and old not in pinned:
                            victim = old
                            break
                # Pass 3: everything in use — oldest non-pinned (never the new key).
                if victim is None:
                    victim = next((old for old in cache if old != key and old not in pinned), None)
            if victim is None:
                break
            del cache[victim]
            if victim in pinned:
                self._pin_resident -= 1
            self._expert_freq.pop(victim, None)
            self.expert_evictions += 1
        return cache[key]

    def _trim_metal_pool(self, *, force: bool = False, threshold_gib: float = 8.0) -> None:
        """Return the allocator's idle free list to the OS.

        Nothing else on the machine can reclaim Metal-resident memory, so a
        bloated free list starves the file page cache and turns warm expert
        re-reads back into SSD reads. Cheap counter read; only acts when idle.
        """
        if force or mx.get_cache_memory() > int(threshold_gib * 2**30):
            mx.clear_cache()

    def _expert_begin_step(self, *, new_sequence: bool = False) -> None:
        """Start of forward: pin prev-step keys; clear pin on new sequence."""
        self._trim_metal_pool()
        if new_sequence:
            self._expert_protected = set()
            self._expert_used_now = set()
            return
        if self._batch and not self._batch_prefetch:
            # A batch step touches ~B x the single-stream union, so pinning the
            # whole previous union would leave no room for the current one.
            # Plain LRU + the per-layer prefetch is the better trade.
            self._expert_protected = set()
            self._expert_used_now = set()
            return
        if self._expert_used_now:
            self._expert_protected = set(self._expert_used_now)
        self._expert_used_now = set()

    def shrink_expert_cache(self, new_cap: int) -> int:
        """Set the non-pinned cache cap, evicting LRU excess safely."""
        new_cap = max(1, int(new_cap))
        self._expert_cache_cap = new_cap
        evicted = 0
        while len(self._expert_cache) - self._pin_resident > new_cap:
            victim = None
            for k in self._expert_cache:
                if k not in self._expert_protected and k not in self._expert_pinned:
                    victim = k
                    break
            if victim is None:
                victim = next((k for k in self._expert_cache if k not in self._expert_pinned), None)
            if victim is None:
                break
            del self._expert_cache[victim]
            self.expert_evictions += 1
            evicted += 1
        return evicted

    def _use_prefill_cache(self) -> None:
        if self._expert_prefill_cache_cap > self._expert_decode_cache_cap:
            self._expert_cache_cap = self._expert_prefill_cache_cap

    def _use_decode_cache(self) -> None:
        if self._expert_cache_cap != self._expert_decode_cache_cap:
            self.shrink_expert_cache(self._expert_decode_cache_cap)

    def _warm_pinned(self) -> None:
        """Eagerly resident-load the statically pinned hot experts (decode path)."""
        if not self._expert_pinned:
            return
        if self._expert_pin_gather:
            if self._expert_pin_banks:
                return
            specs = [(ly, ex, ns) for (ns, ly, ex) in sorted(self._expert_pinned)]
            self._expert_pin_banks = self._expert_loader.load_expert_banks(
                specs, workers=self._expert_prefetch_workers
            )
            return
        missing = [
            (ly, ex, ns)
            for (ns, ly, ex) in self._expert_pinned
            if (ns, ly, ex) not in self._expert_cache
        ]
        if not missing:
            return
        loaded = self._expert_loader.load_experts_parallel(
            missing, workers=self._expert_prefetch_workers
        )
        for (prefix, ly, ex), d in loaded.items():
            key = (prefix, ly, ex)
            if key not in self._expert_cache:
                self._expert_touch(key, d)

    def _submit_pre_attn_expert(self, layer: int, expert: int):
        """Start one speculative expert read without admitting it to the LRU."""
        key = ("layers", int(layer), int(expert))
        if key in self._expert_cache or key in self._expert_pinned:
            return None
        if self._prefetch_pool is None:
            self._prefetch_pool = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="expert-pre-attn"
            )
        loader = self._expert_loader

        def _work():
            t0 = time.perf_counter()
            parts = loader._read_expert_parts(layer, expert, "layers")
            read_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            packed: dict[str, tuple[mx.array, mx.array]] = {}
            nbytes = 0
            for role, wdata, sdata, (rows, pair_cols) in parts:
                nbytes += len(wdata) + len(sdata)
                packed[role] = loader._pack_expert(wdata, sdata, rows, pair_cols)
            pack_s = time.perf_counter() - t1
            return packed, nbytes, read_s, pack_s

        return self._prefetch_pool.submit(_work)

    def _expert_fn(self, layer: int, ns: str = "layers", stack_gateup: bool | None = None):
        stack_gateup = self._expert_gateup_stack if stack_gateup is None else stack_gateup

        def fetch(expert: int):
            if self._fake_experts:
                if self._fake_w is None:
                    d = self._expert_loader.load_expert(layer, expert, prefix=ns)
                    self._fake_w = (d["w1"], d["w3"], d["w2"])
                return self._fake_w
            key = (ns, layer, expert)
            if self._expert_pin_gather and key in self._expert_pinned:
                raise RuntimeError("banked pinned experts must be consumed by gather_qmm")
            hit = self._expert_cache.get(key)
            if hit is not None:
                self.expert_cache_hits += 1
                if stack_gateup and "w13" not in hit:
                    hit = self._expert_touch(key, hit, stack_gateup=True)
                else:
                    self._expert_touch(key)
                if stack_gateup or "w13" in hit:
                    return hit
                return hit["w1"], hit["w3"], hit["w2"]
            self.expert_loads += 1
            d = self._expert_loader.load_expert(layer, expert, prefix=ns)
            cached = self._expert_touch(key, d, stack_gateup=stack_gateup)
            if stack_gateup:
                return cached
            return d["w1"], d["w3"], d["w2"]

        return fetch

    def prefetch_experts(
        self, layer: int, experts, ns: str = "layers", *, stack_gateup: bool = False
    ) -> None:
        """Warm the expert cache for one layer's activated experts in parallel."""
        if not experts or self._fake_experts:
            return
        missing = []
        for e in experts:
            key = (ns, layer, int(e))
            if self._expert_pin_gather and key in self._expert_pinned:
                continue  # resident in the layer bank, not the LRU dictionary
            if key not in self._expert_cache:
                missing.append((layer, int(e), ns))
            else:
                self._expert_touch(key)
        if not missing:
            return
        loaded = self._expert_loader.load_experts_parallel(
            missing, workers=self._expert_prefetch_workers
        )
        for (prefix, ly, ex), d in loaded.items():
            key = (prefix, ly, ex)
            if key in self._expert_cache:
                self._expert_touch(key)
                continue
            self.expert_loads += 1
            self._expert_touch(key, d, stack_gateup=stack_gateup)

    def _f32_weight(self, key: str, name: str, drop: bool = True) -> mx.array:
        """Materialize ``_dense[name]`` as f32 once, then free the bf16 copy.

        ``f32 x bf16`` makes MLX convert the weight before the GEMM anyway;
        handing it a resident f32 copy is bit-identical (maxabs 0 at K=5120
        for N=512/1280/2304) and skips that per-step conversion. Measured at
        M=8: shared w1/w3 1.433 -> 1.040 ms, w2 1.181 -> 0.929 ms.

        Concatenating two same-K weights into a single GEMM is faster still
        (1.040 -> 0.885 ms) but reorders accumulation (maxabs 5.3e-4), which
        is enough to move router coverage 0.3238 -> 0.3187. Not worth it.
        """
        fused = getattr(self, "_fused", None)
        if fused is None:
            fused = self._fused = {}
        w = fused.get(key)
        if w is None:
            a = self._dense[name]
            w = a.astype(mx.float32)
            mx.eval(w)
            fused[key] = w
            if drop:
                del self._dense[name]
        return w

    def _shared_fn(self, layer: int, ns: str = "layers"):
        if ns == "mtp":
            prefix = f"mtp.{layer}.ffn.shared_experts"
        else:
            prefix = f"layers.{layer}.ffn.shared_experts"
        # All three take f32 inputs, so all three upcast today: make the f32
        # weight the canonical one instead of re-deriving it every layer step.
        w1 = self._f32_weight(prefix + "._w1f", prefix + ".w1.weight")
        w3 = self._f32_weight(prefix + "._w3f", prefix + ".w3.weight")
        w2 = self._f32_weight(prefix + "._w2f", prefix + ".w2.weight")

        def run(x: mx.array) -> mx.array:
            xf = x.astype(mx.float32)
            g = mx.minimum(xf @ w1.T, 10.0)
            u = mx.clip(xf @ w3.T, -10.0, 10.0)
            h = (g / (1.0 + mx.exp(-g))) * u
            return (h @ w2.T).astype(x.dtype)

        return run

    def moe(
        self,
        layer: int,
        x: mx.array,
        token_start: int = 0,
        ns: str = "layers",
        top_k: int | None = None,
    ) -> mx.array:
        # None -> the runner-wide override (Runner(top_k=...)), else config.
        if top_k is None:
            top_k = int(getattr(self, "_top_k", 6))
        stack_gateup = (
            getattr(self, "_expert_gateup_stack", False)
            and ns == "layers"
            and x.shape[0] == 1
            and not self._batch
        )
        base = f"mtp.{layer}" if ns == "mtp" else f"layers.{layer}"
        gate_w = self._dense[f"{base}.ffn.gate.weight"]
        gate_b = self._dense[f"{base}.ffn.gate.bias"]

        rp = getattr(self, "_route_pred", None)

        def hook(layer_id, idx, w):
            if rp is not None and ns == "layers":
                rp.observe(gate_w, gate_b, np.asarray(idx, dtype=np.int64))
            # Temporal prefetch is for decode (S=1). Prefill unions across
            # tokens can exceed the cache and thrash LRU — skip those.
            # Lockstep batch decode keys off the union of every row instead.
            if self._batch:
                self._last_route[(ns, layer_id)] = sorted({int(e) for row in idx for e in row})
            elif len(idx) == 1:
                self._last_route[(ns, layer_id)] = sorted({int(e) for e in idx[0]})
            # Backbone schema only records layers [0, n_layers); skip MTP drafts.
            if self._trace is not None and ns == "layers":
                self._trace.router(layer_id, token_start, idx, w)

        def prefetch(layer_id, experts):
            record = self._pre_attn_pending.pop((ns, layer_id), None)
            if record is not None:
                for candidate in record["candidates"]:
                    predicted = candidate["expert"]
                    future = candidate["future"]
                    payload = future.result() if future is not None else None
                    if predicted in experts:
                        self._pre_attn_correct += 1
                        if payload is not None:
                            packed, nbytes, read_s, pack_s = payload
                            key = (ns, layer_id, predicted)
                            self._expert_loader.bytes_read += nbytes
                            self._expert_loader.load_calls += 1
                            self._expert_loader.pre_route_read_worker_s += read_s
                            self._expert_loader.pre_route_pack_worker_s += pack_s
                            self._pre_attn_read_worker_s += read_s
                            self._pre_attn_pack_worker_s += pack_s
                            self._pre_attn_used += 1
                            self._pre_attn_used_bytes += nbytes
                            self.expert_loads += 1
                            self._expert_touch(key, packed, stack_gateup=stack_gateup)
                    else:
                        self._pre_attn_wrong += 1
                        if payload is not None:
                            _packed, nbytes, read_s, pack_s = payload
                            self._expert_loader.bytes_read += nbytes
                            self._expert_loader.load_calls += 1
                            self._expert_loader.pre_route_read_worker_s += read_s
                            self._expert_loader.pre_route_pack_worker_s += pack_s
                            self._pre_attn_read_worker_s += read_s
                            self._pre_attn_pack_worker_s += pack_s
                            self._pre_attn_wasted += 1
                            self._pre_attn_wasted_bytes += nbytes
            self.prefetch_experts(layer_id, experts, ns=ns, stack_gateup=stack_gateup)

        # getattr: tests build Runner via __new__ and set only what they need.
        shared = None if getattr(self, "_no_shared", False) else self._shared_fn(layer, ns)
        # Opt-in async pipeline callbacks (experimental, reversible). All cache
        # ops + counters run in main thread (deterministic, no races); worker
        # threads do SSD+pack only (no shared state). prefetch_fn is skipped in
        # pipeline mode (async submits replace it); expert_fn is used only for
        # per-Future fallback (no double-admit: failed Futures never admitted).
        use_pipe = (
            bool(getattr(self, "_moe_pipeline", False))
            and not getattr(self, "_fake_experts", False)
            and not getattr(self, "_expert_pin_gather", False)
        )
        async_submit = None
        async_finalize = None
        if use_pipe:
            loader = self._expert_loader
            pool = loader._io_pool  # private, same package; OK for experiment

            def async_submit(layer_id: int, expert: int):
                # Check cache (read-only, fast, NO side effects: no touch, no
                # counters). Hits return completed Futures (no I/O); misses
                # submit background SSD+pack (no cache touch, no counters).
                # ALL cache ops + counters deferred to end sorted pass in
                # moe_layer (deterministic, matches baseline final LRU order
                # exactly: sorted touches, sorted admits). Main thread only
                # for check; worker threads do SSD+pack only (no shared state).
                key = (ns, layer_id, int(expert))
                hit = self._expert_cache.get(key)
                if hit is not None:
                    fut = Future()
                    fut.set_result((hit, 0, True, 0.0, 0.0))
                    return fut

                def _work():
                    read_t0 = time.perf_counter()
                    parts = loader._read_expert_parts(layer_id, int(expert), ns)
                    read_s = time.perf_counter() - read_t0
                    pack_t0 = time.perf_counter()
                    out: dict = {}
                    nbytes = 0
                    for role, wdata, sdata, bdata, shape, scale_shape in parts:
                        nbytes += len(wdata) + len(sdata) + (len(bdata) if bdata else 0)
                        out[role] = loader._role_from_parts(wdata, sdata, bdata, shape, scale_shape)
                    pack_s = time.perf_counter() - pack_t0
                    return out, nbytes, False, read_s, pack_s

                return pool.submit(_work)

            def async_finalize(
                layer_id: int, sorted_ids: list[int], saved: dict[int, tuple]
            ) -> None:
                # Replay baseline cache effects after overlapping I/O/QMM:
                # prefetch touches initial hits, admits initial misses, then
                # expert_fn performs its normal sorted lookup for every expert.
                # Pass A: prefetch_experts touches only keys that existed when
                # futures were submitted. Pass B: admit all initial misses in
                # sorted order; each worker miss was one actual expert read.
                for ee in sorted_ids:
                    key = (ns, layer_id, int(ee))
                    if saved[ee][4] and key in self._expert_cache:
                        self._expert_touch(key)
                for ee in sorted_ids:
                    if saved[ee][4]:
                        continue
                    key = (ns, layer_id, int(ee))
                    packed, nbytes = saved[ee][2], saved[ee][3]
                    loader.pipeline_read_worker_s += saved[ee][5]
                    loader.pipeline_pack_worker_s += saved[ee][6]
                    loader.bytes_read += nbytes
                    if key in self._expert_cache:
                        self._expert_touch(key)
                    else:
                        self.expert_loads += 1
                        self._expert_touch(key, packed, stack_gateup=stack_gateup)
                if any(not saved[ee][4] for ee in sorted_ids):
                    # The baseline parallel batch increments load_calls once.
                    loader.load_calls += 1

                # Pass C: call the actual lookup. This handles experts evicted
                # by Pass B exactly like baseline, including fallback I/O.
                fetch = self._expert_fn(layer_id, ns, stack_gateup=stack_gateup)
                for ee in sorted_ids:
                    fetch(int(ee))

        pinned_gather = None
        if (
            getattr(self, "_expert_pin_gather", False)
            and ns == "layers"
            and not getattr(self, "_fake_experts", False)
        ):

            def pinned_gather(layer_id, activations, indices, route_weights):
                entry = self._expert_pin_banks.get(layer_id)
                if entry is None:
                    raise RuntimeError(f"pinned expert bank for layer {layer_id} is not loaded")
                selected = set(map(int, indices.reshape(-1)))
                if not (selected & entry["slots"].keys()):
                    return {}
                result = F.gather_pinned_contributions(
                    activations, indices, route_weights, entry["banks"], entry["slots"]
                )
                self.expert_cache_hits += len(result)
                return result

        out = F.moe_layer(
            x,
            gate_w,
            gate_b,
            self._expert_fn(layer, ns, stack_gateup=stack_gateup),
            shared,
            top_k,
            10.0,
            hook,
            layer,
            prefetch_fn=prefetch,
            # Only lockstep decode has rows that are separate
            # sequences; a prefill's rows are one sequence's
            # tokens and must never be traded away.
            union_budget=(self._union_budget if self._batch else 0),
            union_strict=getattr(self, "_union_strict", False),
            route_stats=self._route_stats,
            pipeline=use_pipe,
            async_submit_fn=async_submit,
            async_finalize_fn=async_finalize,
            pinned_gather_fn=pinned_gather,
            stack_gateup=(stack_gateup and not self._fake_experts),
        )
        return out

    def route_coverage(self) -> float:
        """Share of total gate weight kept by the union cap (1.0 = exact)."""
        st = self._route_stats
        return st["coverage"] / st["calls"] if st["calls"] else 1.0

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
                if self._expert_pin_gather and key in self._expert_pinned:
                    continue
                if key not in self._expert_cache:
                    specs.append((layer, e, ns))
        if not specs:
            return None
        # With static pins the LRU budget is only the unpinned remainder.
        # Replaying more previous-token misses than that remainder would
        # stream a working set that cannot stay resident, evicting its own
        # early entries before the current token reaches those layers.
        if self._expert_pinned and len(specs) > self._expert_cache_cap:
            return None
        if self._prefetch_pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._prefetch_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="expert-warm"
            )

        def _work():
            loaded = self._expert_loader.load_experts_parallel(
                specs, workers=self._expert_prefetch_workers
            )
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
            cfg["rope_head_dim"],
            cfg.get("max_seq_len") or 65536,
            original,
            base,
            cfg.get("rope_factor", 40.0),
            cfg.get("beta_fast", 32),
            cfg.get("beta_slow", 1),
        )
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

    def _o_proj(self, out: mx.array, p: str, n_rows: int, cfg: dict) -> mx.array:
        """Attention output projection.

        Identical contraction to ``einsum("sgd,grd->sgr")`` on
        ``wo_a.reshape(g, r, d)`` and ``out.reshape(n, g, d)``, written as a
        batched matmul over the ``o_groups`` axis instead.
        """
        o_groups = cfg.get("o_groups", 8)
        o_lora = cfg.get("o_lora_rank", 1024)
        # bf16 end to end: DenseLoader already stores these bf16, and casting
        # wo_a to f32 allocated 134 MiB and doubled DRAM traffic per layer step.
        wo_a = self._dense[p + "wo_a.weight"].reshape(o_groups, o_lora, -1)
        og = out.reshape(n_rows, o_groups, -1).astype(mx.bfloat16)
        og = mx.einsum("sgd,grd->sgr", og, wo_a)
        return (og.reshape(n_rows, -1) @ self._dense[p + "wo_b.weight"].T).astype(mx.float32)

    def _sparse_attn(
        self, q: mx.array, kv: mx.array, topk_idxs: mx.array, sink: mx.array, scale: float
    ) -> mx.array:
        """Gather + online softmax + sink (sink feeds denominator only)."""
        # Stay on device: previous numpy path forced GPU->CPU every layer.
        dt = mx.bfloat16 if getattr(self, "_attn_bf16", False) else mx.float32
        qf = q.astype(dt)
        kvf = kv.astype(dt)
        sink_f = sink.astype(dt)
        m, _, d = qf.shape
        idx = topk_idxs.reshape(m, -1)
        mask = idx >= 0
        safe = mx.maximum(idx, 0).astype(mx.int32)
        if kvf.ndim == 3:
            # Batched: each row has its own [K, d] slot table.
            k_len = kvf.shape[1]
            flat = kvf.reshape(-1, d)
            base = (mx.arange(m, dtype=mx.int32) * k_len)[:, None]
            kg = flat[(safe + base).reshape(-1)].reshape(m, -1, d)
        else:
            kg = kvf[safe.reshape(-1)].reshape(m, -1, d)  # [m, k, d]
        scores = mx.einsum("mhd,mkd->mhk", qf, kg) * scale
        scores = mx.where(mask[:, None, :], scores, mx.full(scores.shape, -1e30, dtype=dt))
        mrow = mx.max(scores, axis=-1, keepdims=True)
        exp_s = mx.exp(scores - mrow)
        exp_s = mx.where(mask[:, None, :], exp_s, mx.zeros_like(exp_s))
        denom = mx.sum(exp_s, axis=-1, keepdims=True) + mx.exp(sink_f[None, :, None] - mrow)
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
        rows = x.shape[0]
        if ratio == 1:
            kv = (
                x.astype(mx.float32) @ self._dense[p + "compressor.wkv.weight"].T
                if p + "compressor.wkv.weight" in self._dense
                else x.astype(mx.float32) @ self._dense[p + "wkv.weight"].T
            )
            norm_w = (
                self._dense[p + "compressor.norm.weight"]
                if p + "compressor.norm.weight" in self._dense
                else self._dense[p + "kv_norm.weight"]
            )
            return F.rmsnorm(kv, norm_w, 1e-20)
        wkv = self._dense[p + "compressor.wkv.weight"]
        wgate = self._dense[p + "compressor.wgate.weight"]
        xf = x.astype(mx.float32)
        kv, score = xf @ wkv.T, xf @ wgate.T
        st = self._kv_state.get(layer)
        sc = self._score_state.get(layer)
        if st is None:
            if self._batch:
                st = mx.zeros((rows, ratio, head_dim))
                sc = mx.full((rows, ratio, head_dim), float("-inf"))
            else:
                st = mx.zeros((ratio, head_dim))
                sc = mx.full((ratio, head_dim), float("-inf"))
            mx.eval(st, sc)
            self._kv_state[layer], self._score_state[layer] = st, sc
        if start_pos == 0:
            if self._batch:
                raise RuntimeError(
                    "batched prefill is not supported; prefill sequences one at a time"
                )
            if rows >= ratio:
                rem = rows % ratio
                cut = rows - rem
                if rem:
                    st = mx.concatenate([kv[cut:], st[rem:]], axis=0)
                    sc = mx.concatenate([score[cut:], sc[rem:]], axis=0)
                    self._kv_state[layer], self._score_state[layer] = st, sc
                kv = kv[:cut].reshape(-1, ratio, head_dim)
                score = score[:cut].reshape(-1, ratio, head_dim)
                kv = (kv * mx.softmax(score, axis=1)).sum(axis=1)
            else:
                st = mx.concatenate([kv, st[rows:]], axis=0)
                sc = mx.concatenate([score, sc[rows:]], axis=0)
                self._kv_state[layer], self._score_state[layer] = st, sc
                return None
        elif self._batch:
            slot = start_pos % ratio
            st = mx.concatenate([st[:, :slot], kv[:, None, :], st[:, slot + 1 :]], axis=1)
            sc = mx.concatenate([sc[:, :slot], score[:, None, :], sc[:, slot + 1 :]], axis=1)
            self._kv_state[layer], self._score_state[layer] = st, sc
            if (start_pos + 1) % ratio != 0:
                return None
            kv = (st * mx.softmax(sc, axis=1)).sum(axis=1)
        elif rows == 1:
            slot = start_pos % ratio
            st = mx.concatenate([st[:slot], kv[:1], st[slot + 1 :]], axis=0)
            sc = mx.concatenate([sc[:slot], score[:1], sc[slot + 1 :]], axis=0)
            self._kv_state[layer], self._score_state[layer] = st, sc
            if (start_pos + 1) % ratio != 0:
                return None
            kv = (st * mx.softmax(sc, axis=0)).sum(axis=0, keepdims=True)
        else:
            # Multi-token decode: walk each absolute position through the ring state.
            latents = []
            for i in range(rows):
                pos = start_pos + i
                slot = pos % ratio
                st = mx.concatenate([st[:slot], kv[i : i + 1], st[slot + 1 :]], axis=0)
                sc = mx.concatenate([sc[:slot], score[i : i + 1], sc[slot + 1 :]], axis=0)
                if (pos + 1) % ratio == 0:
                    latents.append((st * mx.softmax(sc, axis=0)).sum(axis=0, keepdims=True))
            self._kv_state[layer], self._score_state[layer] = st, sc
            if not latents:
                return None
            kv = mx.concatenate(latents, axis=0)
        return F.rmsnorm(kv, self._dense[p + "compressor.norm.weight"], 1e-20)

    def _select_candidates(
        self, index_score: mx.array, compress_lens, topk_blocks: int, block_size: int
    ) -> mx.array:
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

    def _run_indexer(
        self, layer: int, x: mx.array, qr: mx.array, latent, start_pos: int, seqlen: int
    ) -> mx.array:
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
        n_rows = x.shape[0]
        if owns_k and latent is not None:
            k = F.rmsnorm(
                latent.astype(mx.float32)
                @ self._dense[p + "indexer.wk.weight"].astype(mx.float32).T,
                self._dense[p + "indexer.k_norm.weight"],
                1e-20,
            )
            if self._batch:
                n_lat = 1  # one new latent per sequence per step
            else:
                n_lat = k.shape[0]
            if start_pos == 0:
                kfreq = freqs_all[: n_lat * ratio : ratio][:n_lat]
            elif ratio == 1:
                kfreq = freqs_all[start_pos : start_pos + n_lat]
            else:
                comps = [
                    pos + 1 - ratio
                    for pos in range(start_pos, start_pos + seqlen)
                    if (pos + 1) % ratio == 0
                ]
                if len(comps) != n_lat:
                    raise RuntimeError(f"index k rows {n_lat} vs completions {len(comps)}")
                kfreq = freqs_all[comps]
            k = mx.concatenate([k[..., :-rd], F.apply_rotary(k[..., -rd:], kfreq)], axis=-1)
            if self._batch:
                k = k[:, None, :]  # [rows, 1, index_dim]
            k_cache = self._indexk.get(layer)
            pos = start_pos // max(ratio, 1)
            need = pos + n_lat
            axis = 1 if self._batch else 0
            if k_cache is None:
                shape = (n_rows, need + 64, index_dim) if self._batch else (need + 64, index_dim)
                k_cache = mx.zeros(shape)
                mx.eval(k_cache)
            elif k_cache.shape[axis] < need:
                pad_shape = (
                    (n_rows, need - k_cache.shape[axis] + 64, index_dim)
                    if self._batch
                    else (need - k_cache.shape[axis] + 64, index_dim)
                )
                k_cache = mx.concatenate([k_cache, mx.zeros(pad_shape)], axis=axis)
            if self._batch:
                k_cache = mx.concatenate([k_cache[:, :pos], k, k_cache[:, pos + n_lat :]], axis=1)
            else:
                k_cache = k_cache.at[pos : pos + n_lat].add(k - k_cache[pos : pos + n_lat])
            self._indexk[layer] = k_cache
            self._shared_indexk = k_cache
        q = (qr @ self._dense[p + "indexer.wq_b.weight"].astype(mx.float32).T).reshape(
            n_rows, n_heads, index_dim
        )
        q = mx.concatenate(
            [
                q[..., :-rd],
                F.apply_rotary(
                    q[..., -rd:], self._freqs_for(layer)[start_pos : start_pos + seqlen]
                ),
            ],
            axis=-1,
        )
        end_pos = start_pos + seqlen
        index_k = self._shared_indexk
        if index_k is None:
            empty = mx.full((n_rows, 0), -1, dtype=mx.int32)
            self._shared_topk = empty
            return empty
        if self._batch:
            index_k = index_k[:, : end_pos // max(ratio, 1)]
        else:
            index_k = index_k[: end_pos // max(ratio, 1)]
        weights = (
            x.astype(mx.float32)
            @ self._dense[p + "indexer.weights_proj.weight"].astype(mx.float32).T
        )
        weights = weights * (index_dim**-0.5 * n_heads**-0.5)
        # GPU score: avoid per-layer GPU->CPU sync of full [S,H,T] tensor.
        if self._batch:
            score_mx = mx.einsum("rhd,rtd->rht", q.astype(mx.float32), index_k.astype(mx.float32))
        else:
            score_mx = mx.einsum("shd,td->sht", q.astype(mx.float32), index_k.astype(mx.float32))
        score_mx = mx.maximum(score_mx, 0) * weights[:, :, None]
        # Opt-in GPU indexer Top-K (experimental, reversible, decode non-batch
        # only). For ordinary index-source layers, avoids transferring full
        # [S,H,T] score tensors to NumPy. Candidate-source layer still calls
        # _select_candidates(), which materializes the reduced scores on CPU.
        # Only the argsort/top-k pipeline on synthetic score vectors has been
        # checked; full score-reduction parity and speed are unmeasured. Default
        # off; CPU path below unchanged.
        if getattr(self, "_indexer_gpu_topk", False) and start_pos > 0 and not self._batch:
            _Tlen = int(score_mx.shape[2])
            _sc = mx.sum(score_mx, axis=1)
            _cl = ((mx.arange(start_pos, start_pos + seqlen) + 1) // max(ratio, 1))[:, None]
            _sc = mx.where(mx.arange(_Tlen)[None, :] >= _cl, mx.array(-float("inf")), _sc)
            _csrc = int(cfg.get("candidate_source_layer", -1))
            if layer == _csrc:
                self._candidates = self._select_candidates(
                    _sc, _cl, cfg["candidate_topk_blocks"], cfg["candidate_block_size"]
                )
            elif _csrc >= 0 and layer > _csrc:
                _mask = self._candidates
                if _mask.shape != _sc.shape:
                    raise RuntimeError(
                        f"candidate mask {_mask.shape} vs score {_sc.shape} "
                        f"at layer {layer} start_pos {start_pos} seqlen {seqlen}"
                    )
                _sc = mx.where(_mask, _sc, mx.array(-float("inf")))
            _topk = min(int(cfg["index_topk"]), end_pos // max(ratio, 1))
            if _topk <= 0:
                empty = mx.full((n_rows, 0), -1, dtype=mx.int32)
                self._shared_topk = empty
                return empty
            _idx = mx.argsort(-_sc, axis=-1)[:, :_topk]
            _idx = mx.sort(_idx, axis=-1)
            _out = mx.where(_idx < _cl, _idx, mx.array(-1)).astype(mx.int32)
            self._shared_topk = _out
            return _out
        score = np.einsum("sht->st", np.asarray(score_mx))
        if start_pos == 0:
            cl = (np.arange(1, seqlen + 1) // ratio)[:, None]
            score = np.where(np.arange(score.shape[1]) >= cl, -np.inf, score)
            compress_lens = cl
        elif self._batch:
            # Every row sits on the same absolute position.
            cl = np.full((n_rows, 1), (end_pos) // max(ratio, 1))
            score = np.where(np.arange(score.shape[1]) >= cl, -np.inf, score)
            compress_lens = cl
        else:
            # Per-query: query i at absolute pos start_pos+i sees (pos+1)//ratio groups.
            cl = ((np.arange(start_pos, start_pos + seqlen) + 1) // max(ratio, 1))[:, None]
            score = np.where(np.arange(score.shape[1]) >= cl, -np.inf, score)
            compress_lens = cl
        if layer == cfg.get("candidate_source_layer", -1):
            self._candidates = self._select_candidates(
                mx.array(score),
                compress_lens,
                cfg["candidate_topk_blocks"],
                cfg["candidate_block_size"],
            )
        elif cfg.get("candidate_source_layer", -1) >= 0 and layer > cfg["candidate_source_layer"]:
            mask = np.asarray(self._candidates)
            if mask.shape != score.shape:
                raise RuntimeError(
                    f"candidate mask {mask.shape} vs score {score.shape} "
                    f"at layer {layer} start_pos {start_pos} seqlen {seqlen}"
                )
            score = np.where(mask, score, -np.inf)
        topk = min(cfg["index_topk"], end_pos // max(ratio, 1))
        if topk <= 0:
            empty = mx.full((n_rows, 0), -1, dtype=mx.int32)
            self._shared_topk = empty
            return empty
        idx = np.argsort(-score, axis=-1, kind="stable")[:, :topk]
        idx = np.sort(idx, axis=-1)
        if start_pos == 0:
            cl = np.arange(1, seqlen + 1) // ratio
            out = np.where(idx < cl[:, None], idx, -1)
        else:
            out = np.where(idx < compress_lens, idx, -1)
        out_mx = mx.array(out.astype(np.int32))
        self._shared_topk = out_mx
        return out_mx

    def indexer_topk(
        self, layer: int, x: mx.array, qr: mx.array, latent, start_pos: int, seqlen: int
    ) -> mx.array:
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
        win = getattr(self, "_window_size", 0) or cfg["window_size"]
        ratio = cfg["compress_ratios"][layer]
        n_rows = x.shape[0]
        B = self._batch
        seqlen = 1 if B else n_rows  # state-machine length (rows are sequences)
        freqs = self._freqs_for(layer)[start_pos : start_pos + seqlen]
        _t = self._att_stats
        if _t is None:
            _t = {
                "proj": 0.0,
                "kv": 0.0,
                "ring": 0.0,
                "compress": 0.0,
                "attn": 0.0,
                "out": 0.0,
                "build": 0.0,
            }
            self._att_stats = _t
        prof = bool(getattr(self, "_att_phase_prof", False))
        if prof:
            for _k in ("e_proj", "e_kv", "e_ring", "e_c", "e_attn", "e_out"):
                _t.setdefault(_k, 0.0)
        t0 = time.perf_counter()

        # wq_a and wkv are both f32 x bf16 GEMMs off the same row (K=5120);
        # keep the two GEMMs but hold the weights in f32. Only wq_a is freed,
        # so the compressor path can still fall back to wkv.weight.
        # --proj-bf16 runs the same two contractions in bf16 instead: the
        # AMX in Apple Silicon accelerates bf16 matmul only, so at batch 512
        # the f32 form leaves that unit idle and reads twice the weight bytes.
        if self._proj_bf16:
            xb = x.astype(mx.bfloat16)
            qr = F.rmsnorm(
                (xb @ self._dense[p + "wq_a.weight"].T).astype(mx.float32),
                self._dense[p + "q_norm.weight"],
                1e-20,
            )
            kv_bf = xb @ self._dense[p + "wkv.weight"].T
            kv_new = F.rmsnorm(kv_bf.astype(mx.float32), self._dense[p + "kv_norm.weight"], 1e-20)
        else:
            wq_a = self._f32_weight(p + "_wqaf", p + "wq_a.weight")
            wkv = self._f32_weight(p + "_wkvf", p + "wkv.weight", drop=False)
            xf = x.astype(mx.float32)
            qr = F.rmsnorm(xf @ wq_a.T, self._dense[p + "q_norm.weight"], 1e-20)
            kv_new = F.rmsnorm(xf @ wkv.T, self._dense[p + "kv_norm.weight"], 1e-20)
        # bf16: an f32 @ bf16 matmul promotes the weight, and wq_b is 84 MiB
        # -- cast per layer per step. qr stays f32 for the indexer.
        q = (qr.astype(mx.bfloat16) @ self._dense[p + "wq_b.weight"].T).reshape(
            n_rows, n_heads, head_dim
        )
        q = mx.concatenate([q[..., :-rd], F.apply_rotary(q[..., -rd:], freqs)], axis=-1)
        t1 = time.perf_counter()
        _t["proj"] = _t.get("proj", 0.0) + t1 - t0
        if prof:
            _te = time.perf_counter()
            mx.eval(q)
            _t["e_proj"] += time.perf_counter() - _te
            t1 = time.perf_counter()

        kv_new = mx.concatenate(
            [kv_new[..., :-rd], F.apply_rotary(kv_new[..., -rd:], freqs)], axis=-1
        )
        t_kv = time.perf_counter()
        _t["kv"] = _t.get("kv", 0.0) + t_kv - t1
        ring = self._window.get(layer)
        if ring is None:
            ring = (
                mx.zeros((B, win, head_dim), dtype=mx.bfloat16)
                if B
                else mx.zeros((win, head_dim), dtype=mx.bfloat16)
            )
            mx.eval(ring)
        if start_pos == 0:
            if B:
                raise RuntimeError("batched prefill is not supported")
            if seqlen <= win:
                ring = ring.at[:seqlen].add(kv_new.astype(mx.bfloat16) - ring[:seqlen])
            else:
                cutoff = seqlen % win
                tail_new = kv_new[-win:].astype(mx.bfloat16)
                ring = (
                    mx.concatenate([tail_new[win - cutoff :], tail_new[: win - cutoff]], axis=0)
                    if cutoff
                    else tail_new
                )
            window_kv = kv_new
            idxs = self._window_topk(win, seqlen, 0)
        elif B:
            # Lockstep batch: one token per sequence, all at start_pos.
            slot = start_pos % win
            kv_bf = kv_new.astype(mx.bfloat16)
            ring = mx.concatenate([ring[:, :slot], kv_bf[:, None, :], ring[:, slot + 1 :]], axis=1)
            window_kv = ring
            idxs = mx.broadcast_to(self._window_topk(win, 1, start_pos), (B, win))
        elif seqlen == 1:
            ring = ring.at[start_pos % win].add(
                kv_new[0].astype(mx.bfloat16) - ring[start_pos % win]
            )
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
        t_ring = time.perf_counter()
        _t["ring"] = _t.get("ring", 0.0) + t_ring - t_kv
        if prof:
            _te = time.perf_counter()
            mx.eval(window_kv, topk_all)
            _t["e_ring"] += time.perf_counter() - _te
            t_ring = time.perf_counter()
        if ratio:
            compress_len = (start_pos + seqlen) // ratio
            latent = None
            if layer in cfg["kv_source_layers"]:
                latent = self._run_compressor(layer, x, start_pos)
                if latent is not None:
                    ck = self._register_compress(
                        layer, self._rotate_latent(layer, latent, ratio, start_pos, seqlen), ratio
                    )
                    self._shared_compress = ck
                own = self._compress.get(layer)
                if own is None:
                    own = mx.zeros((B, 0, cfg["head_dim"])) if B else mx.zeros((0, cfg["head_dim"]))
                    mx.eval(own)
                self._shared_compress = own
            ck = self._shared_compress
            cidx = self.indexer_topk(layer, x, qr, latent, start_pos, seqlen)
            if ck is not None:
                if B:
                    ck = ck[:, :compress_len]
                    window_kv = mx.concatenate([window_kv, ck], axis=1)
                    topk_all = self._shift_compress_idxs(
                        idxs, cidx, int(window_kv.shape[1]) - int(ck.shape[1])
                    )
                else:
                    ck = ck[:compress_len]
                    window_kv = mx.concatenate([window_kv, ck], axis=0)
                    topk_all = self._shift_compress_idxs(
                        idxs, cidx, int(window_kv.shape[0]) - int(ck.shape[0])
                    )
        t_c = time.perf_counter()
        _t["compress"] = _t.get("compress", 0.0) + t_c - t_ring
        if prof:
            _te = time.perf_counter()
            mx.eval(window_kv, topk_all)
            _t["e_c"] += time.perf_counter() - _te
            t_c = time.perf_counter()
            _t["compress"] -= time.perf_counter() - _te
        sink = self._dense[p + "attn_sink"].astype(mx.float32)
        n_compress = int(ck.shape[-2]) if ratio and ck is not None else 0
        out = self._sparse_attn(q, window_kv, topk_all, sink, head_dim**-0.5)
        t_a = time.perf_counter()
        _t["attn"] = _t.get("attn", 0.0) + t_a - t_c
        if prof:
            _te = time.perf_counter()
            mx.eval(out)
            _t["e_attn"] += time.perf_counter() - _te
            t_a = time.perf_counter()
        if self._trace is not None:
            self._trace.kv(layer, start_pos, n_compress, int(np.asarray(topk_all).shape[-1]))
        out = mx.concatenate([out[..., :-rd], self._unrotary(out[..., -rd:], freqs)], axis=-1)
        result = self._o_proj(out, p, n_rows, cfg)
        t_e = time.perf_counter()
        _t["out"] = _t.get("out", 0.0) + t_e - t_a
        _t["build"] = _t.get("build", 0.0) + t_e - t0
        if prof:
            _te = time.perf_counter()
            mx.eval(result)
            _t["e_out"] += time.perf_counter() - _te
        return result

    @staticmethod
    def _shift_compress_idxs(idxs: mx.array, cidx: mx.array, base: int) -> mx.array:
        left = np.asarray(idxs, dtype=np.int64)
        right = np.asarray(cidx, dtype=np.int64)
        right = np.where(right >= 0, right + base, -1)
        return mx.array(np.concatenate([left, right], axis=-1).astype(np.int32))

    def _rotate_latent(
        self, layer: int, latent: mx.array, ratio: int, start_pos: int, seqlen: int
    ) -> mx.array:
        """Rotate each latent row with its group-start position frequencies."""
        rd = self._config["rope_head_dim"]
        freqs_all = self._freqs_for(layer)
        n = 1 if self._batch else latent.shape[0]
        if start_pos == 0:
            rows = np.minimum(np.arange(n) * ratio, freqs_all.shape[0] - 1)
            freqs = mx.array(np.asarray(freqs_all)[rows])
        elif ratio == 1:
            freqs = freqs_all[start_pos : start_pos + n]
        else:
            rows = [
                pos + 1 - ratio
                for pos in range(start_pos, start_pos + seqlen)
                if (pos + 1) % ratio == 0
            ]
            if len(rows) != n:
                raise RuntimeError(
                    f"latent rows {n} vs completions {len(rows)} "
                    f"start_pos={start_pos} seqlen={seqlen} ratio={ratio}"
                )
            freqs = mx.array(np.asarray(freqs_all)[rows])
        tail = F.apply_rotary(latent[..., -rd:], freqs)
        return mx.concatenate([latent[..., :-rd], tail], axis=-1)

    def _register_compress(self, layer: int, latent: mx.array, ratio: int) -> mx.array:
        cache = self._compress.get(layer)
        new = latent.astype(mx.bfloat16)
        axis = 1
        if self._batch:
            if new.ndim == 2:
                new = new[:, None, :]  # [rows, 1, head_dim]
        else:
            axis = 0
        if cache is None:
            cache = new
        else:
            cache = mx.concatenate([cache, new], axis=axis)
        self._compress[layer] = cache
        mx.eval(cache)
        return cache

    # -- engram -----------------------------------------------------------------
    def _engram_hash_ids(
        self, token_ids: list[int], start_pos: int, row: int = 0, batched: bool = False
    ):
        """Exact upstream hashes (torch CPU): [L, n_engram_layers, n_cols].

        ``row`` routes a single-sequence prefill into that sequence's row of
        the shared ngram cache (a slice view, so only that row is touched).
        ``batched`` treats ``token_ids`` as one token per sequence and uses
        all ``B`` cache rows at once -- the rolling lookback then reads each
        sequence's own history.
        """
        import sys as _sys

        snap = str(
            Path(__file__).resolve().parents[2]
            / "upstream"
            / (
                "DeepSeek-V4.1-Flash-1017"
                if os.environ.get("DEEPSEEKER_VARIANT") == "1017"
                else "DeepSeek-V4.1-Flash"
            )
            / "inference"
        )
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
                max_batch_size=self._hash_bsz,
                max_seq_len=cfg.get("max_seq_len", 65536),
            )
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(self._root))
            layout = _up.EngramLayout.from_args(args)
            self._hash_state = _up.NgramHashState(args, layout, tok)
            self._hash_ncols = (cfg["engram_max_ngram_size"] - 1) * cfg["engram_n_heads"]
        if batched:
            ids = _torch.tensor(token_ids, dtype=_torch.int64).reshape(len(token_ids), 1)
            with _torch.inference_mode():
                hashes = self._hash_state(ids, start_pos)
            return np.asarray(hashes).reshape(len(token_ids), -1, self._hash_ncols)
        ids = _torch.tensor([token_ids])
        cache = self._hash_state.cache
        if row:
            self._hash_state.cache = cache[row : row + 1]
        try:
            with _torch.inference_mode():
                hashes = self._hash_state(ids, start_pos)
        finally:
            if row:
                self._hash_state.cache = cache
        return np.asarray(hashes[0]).reshape(len(token_ids), -1, self._hash_ncols)

    def _engram_row(self, layer: int, row_id: int) -> np.ndarray:
        """Dequantized embed row (256 fp8 + 8 ue8m0 scales) as float32."""
        from deepseeker.mlx_weights import E4M3_TABLE, decode_u8m0

        rows = self._engram_cache.fetch_rows(layer, [int(row_id)])
        w_raw = np.frombuffer(rows[0][0], dtype=np.uint8)
        s_raw = np.frombuffer(rows[0][1], dtype=np.uint8)
        return (E4M3_TABLE[w_raw.astype(np.int64)] * np.repeat(decode_u8m0(s_raw), 32)).astype(
            np.float32
        )

    def _engram_numpy(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """(wkv f32, q*k f32) for one engram layer, materialized once.

        These used to be re-cast from fp8 and re-downloaded from the GPU on
        every `engram_forward` call -- 629 MiB of f32 for wkv alone -- even
        though the weights are constant for the whole run.
        """
        cached = self._engram_np.get(layer)
        if cached is None:
            wkv = np.asarray(
                self._dense[f"layers.{layer}.engram.wkv.weight"].astype(mx.float32),
                dtype=np.float32,
            )
            q_w = np.asarray(self._dense[f"layers.{layer}.engram.q_weight"].astype(mx.float32))
            k_w = np.asarray(self._dense[f"layers.{layer}.engram.k_weight"].astype(mx.float32))
            cached = (wkv, q_w * k_w)
            self._engram_np[layer] = cached
        return cached

    def engram_forward(
        self, layer: int, start_pos: int, x: mx.array, hash_ids: np.ndarray
    ) -> mx.array:
        """Exact Engram math: embed rows -> wkv -> gated add into stream.

        The row gather and the constant wkv matmul stay on numpy (they only
        touch host-side row caches).  The gate itself runs on MLX: doing it in
        numpy required ``np.asarray(x)``, which forces a full device drain in
        the middle of the block stack for every engram layer.
        """
        t0 = time.perf_counter()
        cfg = self._config
        dim, hc = cfg["dim"], cfg["hc_mult"]
        # One batched fetch + one vectorised fp8 dequant per call. The old
        # path called fetch_rows([id]) per position: thousands of python-level
        # page lookups per decode step, and it was 10s of every 43s of decode.
        from deepseeker.mlx_weights import E4M3_TABLE, decode_u8m0

        shape = hash_ids.shape
        flat = np.asarray(hash_ids).reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        fetched = self._engram_cache.fetch_rows(layer, [int(r) for r in uniq])
        w_raw = np.frombuffer(b"".join(p[0] for p in fetched), dtype=np.uint8)
        s_raw = np.frombuffer(b"".join(p[1] for p in fetched), dtype=np.uint8)
        w_raw = w_raw.reshape(len(uniq), -1)
        s_raw = s_raw.reshape(len(uniq), -1)
        dec = E4M3_TABLE[w_raw.astype(np.int64)] * np.repeat(decode_u8m0(s_raw), 32, axis=-1)
        rows = dec[inv].reshape(shape[0], shape[1], -1)
        self._ph_tick("f_eng_rows")
        kv = rows.reshape(rows.shape[0], -1).astype(np.float32)
        wkv_np, weight = self._engram_numpy(layer)
        # kv @ wkv.T is ~1.7 GFLOP per call; in numpy that was 72ms of the
        # 2.3s engram phase. Run it on the GPU and keep key/value there too.
        wkv_mx = self._engram_wkv_mx.get(layer)
        if wkv_mx is None:
            wkv_mx = mx.array(np.ascontiguousarray(wkv_np.T))
            self._engram_wkv_mx[layer] = wkv_mx
        kv_mx = mx.matmul(mx.array(kv), wkv_mx)
        self._ph_tick("f_eng_gemm", kv_mx)
        key = kv_mx[..., : hc * dim].reshape(-1, hc, dim)
        value = kv_mx[..., hc * dim :].reshape(-1, 1, dim)
        w = self._engram_mx.get(layer)
        if w is None:
            w = mx.array(weight.astype(np.float32))
            self._engram_mx[layer] = w
        eps = cfg.get("norm_eps", 1e-20)
        h = x.astype(mx.float32)
        key = key.astype(mx.float32)
        rstd = mx.rsqrt((h * h).mean(-1) + eps) * mx.rsqrt((key * key).mean(-1) + eps)
        dot = ((h * w * key).sum(-1)) * rstd * (dim**-0.5)
        mag = mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
        gate = mx.sigmoid(mx.where(dot >= 0, mag, -mag))
        out = h + gate[..., None] * value
        if self._trace is not None:
            self._trace.engram(layer, start_pos, hash_ids.tolist())
        out_mlx = out.astype(x.dtype)
        self._engram_s += time.perf_counter() - t0
        return out_mlx

    # -- block + transformer ------------------------------------------------------
    def _ph_tick(self, name: str, *arrs: mx.array) -> None:
        """Optional decode-phase clock (see Runner._phase): eval, then bank."""
        box = getattr(self, "_phase_box", None)
        if box is None:
            return
        if arrs:
            mx.eval(*arrs)
        now = time.perf_counter()
        ph = self._phase
        ph[name] = ph.get(name, 0.0) + now - box[0]
        box[0] = now

    def _skip_block_required(self, cfg: dict) -> frozenset[int]:
        """Memoised: block() asks once per layer per step (40 x 512/run)."""
        cached = self.__dict__.get("_skip_block_set")
        if cached is not None:
            return cached
        return self._build_skip_block_set(cfg)

    def _build_skip_block_set(self, cfg: dict) -> frozenset[int]:
        """Layers a whole-Block bypass must leave running.

        index_source_layers publish ``_shared_topk`` / ``_shared_indexk`` that
        every other layer reads back, and candidate_source_layer builds the
        candidate mask. Skipping any of them breaks the stack outright -- a
        ``--skip-block 2`` sweep dropped all eight index sources and raised
        ``no shared topk published yet`` on the first non-source layer.

        ``keep_layers`` trims that set down to the first N index sources
        (candidate included when N > 0). Under ``skip_block == 1`` every layer
        matches the bypass, so no surviving layer ever calls ``indexer_topk``
        -- the set only has to cover whatever still runs attention, and N=0
        legitimately leaves nothing. Quality is not a target for the frontier
        sweep; this exists to price the per-layer cost of the decode loop.
        """
        n = int(getattr(self, "_keep_layers", -1))
        if n >= 0:
            src = sorted(int(v) for v in cfg.get("index_source_layers", ()))
            keep = set(src[:n])
            cand = cfg.get("candidate_source_layer", -1)
            if n > 0 and cand is not None and cand >= 0:
                keep.add(int(cand))
            self._skip_block_set = frozenset(keep)
            return self._skip_block_set
        cand = cfg.get("candidate_source_layer", -1)
        self._skip_block_set = frozenset(cfg.get("index_source_layers", ())) | (
            frozenset([int(cand)]) if cand is not None and cand >= 0 else frozenset()
        )

    def _all_blocks_bypassed(self, cfg: dict) -> bool:
        """True when the whole layer stack takes the whole-Block early-return."""
        cached = self.__dict__.get("_all_bypassed")
        if cached is None:
            sb = int(getattr(self, "_skip_block", 0))
            req = self._skip_block_required(cfg) if sb else frozenset()
            cached = bool(sb) and all(
                layer % sb == 0 and layer not in req for layer in range(cfg["n_layers"])
            )
            self._all_bypassed = bool(cached)
        return self._all_bypassed

    def block(
        self,
        layer: int,
        x: mx.array,
        start_pos: int,
        pre_mix: mx.array,
        ns: str = "layers",
        main_x: mx.array | None = None,
        top_k: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """One Block: hc mixes -> attn -> hc_post -> hc mixes -> MoE -> hc_post.

        ns='mtp' uses mtp.{layer}. weights; main_x routes to DSparkAttention.
        """
        cfg = self._config
        sb = getattr(self, "_skip_block", 0) if ns == "layers" else 0
        if sb and layer % sb == 0 and layer not in self._skip_block_required(cfg):
            # Whole-Block bypass: the layer is the identity, so it reads no
            # weights at all. pre_mix carries forward untouched -- the next
            # Block's hc_pre still wants a [S, hc_mult] mix, and shapes are
            # what the pipeline cares about here, not this layer's contribution.
            self._ph_tick("skip_block")
            return x, pre_mix
        rp = getattr(self, "_route_pred", None)
        if rp is not None:
            rp.stash("x0", x)
        p = f"mtp.{layer}." if ns == "mtp" else f"layers.{layer}."
        hc = cfg["hc_mult"]
        eps = cfg.get("norm_eps", 1e-20)
        iters = getattr(self, "_hc_iters", 0) or int(cfg.get("hc_sinkhorn_iters", 20))
        hc_eps = cfg.get("hc_eps", 1e-6)
        attn_pre, attn_post, attn_comb = F.hc_mixes(
            x,
            self._dense[p + "hc_attn_fn"],
            self._dense[p + "hc_attn_scale"],
            self._dense[p + "hc_attn_base"],
            hc,
            eps,
            iters,
            hc_eps,
        )
        xpre = F.hc_pre(x, pre_mix)
        if rp is not None:
            rp.stash("xpre", xpre)
        pre_route_candidates = []
        if (
            self._pre_attn_prefetch_enabled
            and not self._moe_pipeline
            and ns == "layers"
            and start_pos > 0
            and xpre.ndim == 2
            and xpre.shape[0] == 1
        ):
            gate_w = self._dense[p + "ffn.gate.weight"]
            gate_b = self._dense[p + "ffn.gate.bias"]
            pred_scores = xpre.astype(mx.float32) @ gate_w.astype(mx.float32).T + gate_b.astype(
                mx.float32
            )
            mx.eval(pred_scores)
            scores_np = np.asarray(pred_scores).reshape(-1)
            candidate_k = min(max(1, int(self._pre_attn_top_k)), 6, scores_np.size)
            candidate_ids = np.argpartition(scores_np, -candidate_k)[-candidate_k:]
            candidate_ids = candidate_ids[np.argsort(scores_np[candidate_ids])[::-1]]
            for candidate_id in candidate_ids:
                expert = int(candidate_id)
                self._pre_attn_predictions += 1
                pre_route_key = (ns, layer, expert)
                cache_hit = (
                    pre_route_key in self._expert_cache or pre_route_key in self._expert_pinned
                )
                if cache_hit:
                    self._pre_attn_cache_hits += 1
                    future = None
                else:
                    self._pre_attn_reads += 1
                    future = self._submit_pre_attn_expert(layer, expert)
                pre_route_candidates.append(
                    {
                        "expert": expert,
                        "future": future,
                        "cache_hit": cache_hit,
                    }
                )
        xin = F.rmsnorm(xpre, self._dense[p + "attn_norm.weight"], cfg.get("norm_eps", 1e-20))
        if rp is not None:
            rp.stash("xin", xin)
        self._ph_tick("hc1", attn_pre, attn_post, attn_comb, xin)
        if ns == "mtp":
            xa = self.dspark_attention(layer, xin, start_pos, main_x)
        else:
            xa = self.attention(layer, xin, start_pos)
        self._ph_tick("att", xa)
        x = F.hc_post(xa, x, attn_post, attn_comb)
        self._ph_tick("post1", x)
        ffn_pre, ffn_post, ffn_comb = F.hc_mixes(
            x,
            self._dense[p + "hc_ffn_fn"],
            self._dense[p + "hc_ffn_scale"],
            self._dense[p + "hc_ffn_base"],
            hc,
            eps,
            iters,
            hc_eps,
        )
        xin2 = F.hc_pre(x, attn_pre)
        xin2 = F.rmsnorm(xin2, self._dense[p + "ffn_norm.weight"], cfg.get("norm_eps", 1e-20))
        self._ph_tick("hc2", ffn_pre, ffn_post, ffn_comb, xin2)
        if pre_route_candidates:
            self._pre_attn_pending[(ns, layer)] = {
                "candidates": pre_route_candidates,
            }
        skip = getattr(self, "_skip_every", 0) and ns == "layers" and layer % self._skip_every == 0
        if skip:
            # Bypass the MoE sublayer entirely: attention output stands, and
            # ffn_pre still feeds the next layer's hc pipeline.
            return x, ffn_pre
        xf = self.moe(layer, xin2, start_pos, ns=ns, top_k=top_k)
        self._ph_tick("moe", xf)
        x = F.hc_post(xf, x, ffn_post, ffn_comb)
        self._ph_tick("post2", x)
        return x, ffn_pre

    def forward(
        self,
        token_ids: list[int] | mx.array,
        start_pos: int,
        want_main: bool = False,
        *,
        scale_invariant: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Text path: embed -> hc expand -> [engram] 40 blocks -> norm -> head.

        want_main also returns cat of target-layer (37,38,39) mean-pooled hiddens
        for the DSpark MTP draft head (#29).
        """
        cfg = self._config
        hc = cfg["hc_mult"]
        if getattr(self, "_phase", None) is not None:
            self._phase_box = [time.perf_counter()]
        self._expert_begin_step(new_sequence=(start_pos == 0))
        if start_pos == 0:
            self._warm_pinned()
        warm_future = None
        if start_pos == 0:
            # New sequence: drop temporal map so we don't prefetch stale keys.
            self._last_route.clear()
        # Temporal prefetch for every step, not just prefill. The cache is far
        # smaller than the working set, so without this each layer's first
        # access stalls on the SSD. Only misses are loaded, and only while the
        # embedding/attention work is in flight, so the load latency hides
        # behind compute instead of adding to the step.
        warm_future = self.prefetch_last_routes()
        engram_on = bool(cfg.get("engram_layer_ids")) and not self._no_engram
        if (
            not engram_on
            and not want_main
            and warm_future is None
            and self._all_blocks_bypassed(cfg)
        ):
            mt = self._composed_head()
            if mt is not None:
                ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
                logits = mt[ids]
                self._ph_tick("f_embed", logits)
                self._ph_tick("f_hash")
                self._ph_tick("f_tail", logits)
                self._ph_tick("f_head", logits)
                return logits
            h = self.embed(token_ids)
            # Nothing below ever touches h, and pre_mix starts as the identity
            # mix ([1, 0, ...]), so F.hc_pre(broadcast(h), pre_mix) == h
            # exactly. The broadcast + hc_pre were doing that identity maths on
            # a [B, hc, dim] buffer every step for nothing; skip the whole
            # hc expansion. Prefill keeps the normal path (warm_future).
            self._ph_tick("f_embed", h)
            self._ph_tick("f_hash")
            self._ph_tick("f_tail", h)
            logits = self.head_logits(h, scale_invariant=scale_invariant)
            self._ph_tick("f_head", logits)
            return logits
        h = self.embed(token_ids)
        h = mx.broadcast_to(h[:, None, :], (h.shape[0], hc, h.shape[1]))
        pre_mix = mx.zeros((h.shape[0], hc))
        pre_mix = pre_mix.at[:, 0].add(1.0)  # identity pre-mix, not uniform
        # Nothing below reads h eagerly -- the blocks and hc_pre are lazy
        # chains and _ph_tick re-evals when profiling -- so this unconditional
        # sync is pure overhead on the decode hot path. It was ~0.4ms of the
        # 0.92ms/step keep=0 floor. Keep it only when something downstream
        # really does inspect h (engram) or when a phase clock needs it.
        if getattr(self, "_phase", None) is not None or (
            cfg.get("engram_layer_ids") and not self._no_engram
        ):
            mx.eval(h, pre_mix)
        self._ph_tick("f_embed", h, pre_mix)
        hashes = None
        if cfg.get("engram_layer_ids") and not self._no_engram:
            # Skipping the hash also skips engram_forward below (hashes is
            # the only gate), so both the torch-CPU ngram hashing and the
            # row fetch/dequant/wkv/gate disappear together.
            hashes = self._engram_hash_ids(
                token_ids, start_pos, row=self._hash_row, batched=bool(self._batch)
            )
        targets = set(cfg.get("dspark_target_layer_ids") or ())
        self._ph_tick("f_hash")
        if warm_future is not None:
            warm_future.result()  # block only once, after embed/engram setup
        main_parts: list[mx.array] = []
        if hashes is None and not want_main and self._all_blocks_bypassed(cfg):
            # Every layer would take the whole-Block early-return, so building
            # 40 per-layer f-strings and guard calls buys nothing. keep=0.
            self._ph_tick("skip_block")
            final = F.hc_pre(h, pre_mix)
            self._ph_tick("f_tail", final)
            logits = self.head_logits(final, scale_invariant=scale_invariant)
            self._ph_tick("f_head", logits)
            return logits
        for layer in range(cfg["n_layers"]):
            if hashes is not None and layer in cfg["engram_layer_ids"]:
                li = cfg["engram_layer_ids"].index(layer)
                h = self.engram_forward(layer, start_pos, h, hashes[:, li, :])
                self._ph_tick("f_eng_gate", h)
            if want_main and layer in targets:
                main_parts.append(mx.mean(h, axis=1))  # [S, dim]
            h, pre_mix = self.block(layer, h, start_pos, pre_mix, top_k=self._top_k)
        final = F.hc_pre(h, pre_mix)
        self._ph_tick("f_tail", final)
        logits = self.head_logits(final, scale_invariant=scale_invariant)
        self._ph_tick("f_head", logits)
        if want_main:
            main_hidden = mx.concatenate(main_parts, axis=-1)  # [S, 3*dim]
            return logits, main_hidden
        return logits

    def dspark_attention(
        self, stage: int, x: mx.array, start_pos: int, main_x: mx.array
    ) -> mx.array:
        """DSparkAttention: window ring seeded by main_kv; draft queries only.

        compress_ratio == 0 (window-only). start_pos==0 seeds the ring from
        main_x and returns x unchanged (prefill path).
        """
        cfg = self._config
        p = f"mtp.{stage}.attn."
        n_heads, head_dim = cfg["n_heads"], cfg["head_dim"]
        rd = cfg["rope_head_dim"]
        win = getattr(self, "_window_size", 0) or cfg["window_size"]
        freqs_m = self._freqs_for(stage, "mtp")[start_pos : start_pos + main_x.shape[0]]
        # main_kv from main_x (projected target hidden), roped at main positions
        kv_norm = self._dense[p + "kv_norm.weight"]
        eps = cfg.get("norm_eps", 1e-20)
        mkv = F.rmsnorm(main_x.astype(mx.float32) @ self._dense[p + "wkv.weight"].T, kv_norm, eps)
        mkv = mx.concatenate([mkv[..., :-rd], F.apply_rotary(mkv[..., -rd:], freqs_m)], axis=-1)
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
                ring = (
                    mx.concatenate([tail[win - cutoff :], tail[: win - cutoff]], axis=0)
                    if cutoff
                    else tail
                )
            self._mtp_window[stage] = ring
            return x
        block = int(x.shape[0])
        seqlen = int(main_x.shape[0])
        freqs_d = self._freqs_for(stage, "mtp")[start_pos + seqlen : start_pos + seqlen + block]
        qr = F.rmsnorm(
            x.astype(mx.float32) @ self._dense[p + "wq_a.weight"].T,
            self._dense[p + "q_norm.weight"],
            1e-20,
        )
        q = (qr @ self._dense[p + "wq_b.weight"].T).reshape(block, n_heads, head_dim)
        q = mx.concatenate([q[..., :-rd], F.apply_rotary(q[..., -rd:], freqs_d)], axis=-1)
        kv_new = F.rmsnorm(x.astype(mx.float32) @ self._dense[p + "wkv.weight"].T, kv_norm, eps)
        kv_new = mx.concatenate(
            [kv_new[..., :-rd], F.apply_rotary(kv_new[..., -rd:], freqs_d)], axis=-1
        )
        # write single main_kv row (decode seqlen==1) into ring at start_pos
        ring = ring.at[start_pos % win].add(mkv[0].astype(mx.bfloat16) - ring[start_pos % win])
        self._mtp_window[stage] = ring
        hist = min(win, start_pos + 1)
        kv_all = mx.concatenate([ring, kv_new.astype(mx.bfloat16)], axis=0)
        base = np.concatenate([np.arange(hist), win + np.arange(block)])
        idxs = np.tile(base.astype(np.int32), (block, 1))
        sink = self._dense[p + "attn_sink"].astype(mx.float32)
        out = self._sparse_attn(q, kv_all, mx.array(idxs), sink, head_dim**-0.5)
        out = mx.concatenate([out[..., :-rd], self._unrotary(out[..., -rd:], freqs_d)], axis=-1)
        return self._o_proj(out, p, block, cfg)

    def _snapshot_caches(self, *, include_hash_state: bool = False) -> dict:
        snap = {
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
        if include_hash_state:
            # Engram's upstream NgramHashState writes token IDs into a Torch
            # cache in place. The MLX sequence caches above are replaced with
            # new arrays, so their shallow snapshot is sufficient; preserve
            # this one mutable row separately for reversible diagnostic probes.
            hash_state = getattr(self, "_hash_state", None)
            hash_row = int(getattr(self, "_hash_row", 0))
            hash_row_cache = None
            hash_ncols = getattr(self, "_hash_ncols", None)
            if hash_state is not None:
                if not 0 <= hash_row < hash_state.cache.shape[0]:
                    raise RuntimeError("Engram hash row is outside its state cache")
                hash_row_cache = hash_state.cache[hash_row : hash_row + 1].clone()
            snap["_hash_state_ref"] = hash_state
            snap["_hash_state_row"] = hash_row
            snap["_hash_state_row_cache"] = hash_row_cache
            snap["_hash_ncols"] = hash_ncols
        return snap

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
        if "_hash_state_ref" in snap:
            hash_state = snap["_hash_state_ref"]
            if hash_state is None:
                self.__dict__.pop("_hash_state", None)
                self.__dict__.pop("_hash_ncols", None)
            else:
                row = snap["_hash_state_row"]
                cache = snap["_hash_state_row_cache"]
                if cache is None or hash_state.cache[row : row + 1].shape != cache.shape:
                    raise RuntimeError("Engram hash-state snapshot is invalid")
                self._hash_state = hash_state
                hash_state.cache[row : row + 1].copy_(cache)
                self._hash_ncols = snap["_hash_ncols"]

    def reset_seq_state(self) -> None:
        """Drop sequence-dependent KV/compress/index state (expert cache stays warm)."""
        self._window = {}
        self._compress = {}
        self._indexk = {}
        self._kv_state = {}
        self._score_state = {}
        self._shared_compress = None
        self._shared_topk = None
        self._shared_indexk = None
        self._candidates = None
        self._mtp_window = {}
        self._last_route.clear()

    # -- lockstep batch decode -------------------------------------------------
    _BATCH_STATE_KEYS = ("window", "compress", "indexk", "kv_state", "score_state")

    def begin_batch(self, snaps: list[dict]) -> None:
        """Enter lockstep batch decode from N single-sequence cache snapshots.

        Every per-sequence cache gains a leading batch axis. All N sequences
        must sit on the same absolute position (one token each per step).
        """
        n = len(snaps)
        if n < 1:
            raise ValueError("empty batch")
        if n > self._hash_bsz:
            raise ValueError(f"batch {n} exceeds max_batch {self._hash_bsz}")
        for key in self._BATCH_STATE_KEYS:
            layers = set(snaps[0][key])
            for s in snaps[1:]:
                if set(s[key]) != layers:
                    raise ValueError(f"inconsistent {key} layers across batch")
            merged = {ly: mx.stack([s[key][ly] for s in snaps], axis=0) for ly in layers}
            setattr(self, "_" + key, merged)
        # Per-row state stacks cleanly. shared_topk / candidates are shaped by
        # the prefill row count, not the batch, so they are dropped and rebuilt
        # on the first batch step (layer 2 owns both).
        for key in ("shared_indexk", "shared_compress"):
            vals = [s[key] for s in snaps]
            setattr(
                self, "_" + key, None if any(v is None for v in vals) else mx.stack(vals, axis=0)
            )
        self._shared_topk = None
        self._candidates = None
        self._last_route.clear()
        self._expert_protected = set()
        self._expert_used_now = set()
        self._batch = n
        mx.eval(*[a for key in self._BATCH_STATE_KEYS for a in getattr(self, "_" + key).values()])

    def end_batch(self) -> None:
        """Leave batch mode; per-sequence caches are dropped."""
        self._batch = 0
        self.reset_seq_state()

    def generate_batch(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        force_tokens: list[list[int]] | None = None,
        record_top: int = 0,
        dedup: bool = False,
    ) -> dict:
        """Prefill each sequence on its own, then decode them lockstep together.

        Every prompt must be the same length: positions are shared across the
        batch. Returns aggregate throughput plus each sequence's tokens.

        force_tokens feeds a reference decode instead of each step's argmax, so
        two runs can be compared under identical inputs (teacher forcing).
        record_top stores the top-K logits per step for that comparison.

        dedup collapses byte-identical prompts to one representative: their
        state is identical at every step, so one prefill + one row per step
        reproduces all copies bit for bit. The bench's batch fills by cycling a
        32-prompt pool, so batch 512 is exactly 16 copies of each -- running
        the duplicates is pure waste. Off (and impossible) under force_tokens,
        where copies can be driven by different inputs.
        """
        if not prompts:
            raise ValueError("empty prompts")
        lengths = {len(p) for p in prompts}
        if len(lengths) != 1:
            raise ValueError(f"prompt lengths must match, got {sorted(lengths)}")
        n = len(prompts)
        dedup = bool(dedup) and force_tokens is None
        reps: list[int] = []
        slot_of = list(range(n))
        if dedup:
            seen: dict[tuple[int, ...], int] = {}
            for i, p in enumerate(prompts):
                k = tuple(p)
                if k not in seen:
                    seen[k] = len(reps)
                    reps.append(i)
                slot_of[i] = seen[k]
        else:
            reps = list(range(n))
        self._composed_head()  # build outside decode timing; head.weight intact
        prefill_s = 0.0
        rep_snap: dict[int, dict] = {}
        rep_nxt: dict[int, int] = {}
        self._use_prefill_cache()
        for row in reps:
            self.reset_seq_state()
            self._hash_row = row
            t0 = time.perf_counter()
            logits = self.forward(list(prompts[row]), 0)
            mx.eval(logits)
            prefill_s += time.perf_counter() - t0
            rep_nxt[row] = int(np.argmax(_np_f32(logits[-1])))
            rep_snap[row] = self._snapshot_caches()
        self._use_decode_cache()
        self._hash_row = 0
        self.begin_batch([rep_snap[r] for r in reps])
        nxt = [rep_nxt[reps[slot_of[i]]] for i in range(n)]
        # Prefill is timed separately; reset the MoE sync counters so the
        # reported cpu/gpu_wait are decode-only.
        F.SYNC["gpu_wait"] = 0.0
        F.SYNC["cpu"] = 0.0
        F.SYNC["n"] = 0.0
        # Drop prefill from the optional phase counters: decode-only numbers.
        if getattr(self, "_phase", None) is not None:
            self._phase.clear()
        if self._att_stats is not None:
            self._att_stats.clear()
        n = len(prompts)
        out: list[list[int]] = [[] for _ in prompts]
        steps: list[float] = []
        decode_s = 0.0
        pos = lengths.pop()
        ld = self._expert_loader
        pre_io, pre_bytes, pre_calls = ld.io_s, ld.bytes_read, ld.load_calls
        pre_pipe_read = ld.pipeline_read_worker_s
        pre_pipe_pack = ld.pipeline_pack_worker_s
        pre_route = (
            self._pre_attn_predictions,
            self._pre_attn_cache_hits,
            self._pre_attn_correct,
            self._pre_attn_wrong,
            self._pre_attn_reads,
            self._pre_attn_used,
            self._pre_attn_used_bytes,
            self._pre_attn_wasted,
            self._pre_attn_wasted_bytes,
            self._pre_attn_read_worker_s,
            self._pre_attn_pack_worker_s,
        )
        pre_hits, pre_loads = self.expert_cache_hits, self.expert_loads
        self._route_stats = {"coverage": 0.0, "calls": 0.0}
        tops: list[tuple[np.ndarray, np.ndarray]] = []
        # Warm the decode path before the clock starts. Metal compiles each
        # kernel's pipeline state on its first dispatch and that costs ~12ms
        # here; paid inside decode_s it made tok/s depend on --max-tokens, since
        # at 1024 steps the one-off was most of a step's budget. One throwaway
        # step compiles every kernel the loop will use, and it advances no state
        # because the result is discarded (weight loading and prefill are
        # likewise outside the decode clock).
        # A throwaway forward is not state-free when Engram is enabled: its
        # upstream NgramHashState advances in place. Keep golden decoding
        # intact and include first-dispatch compile cost in this case.
        warmup_safe = not (self._config.get("engram_layer_ids") and not self._no_engram)
        if max_new_tokens and getattr(self, "_phase", None) is None and warmup_safe:
            warm_ids = mx.array([nxt[r] for r in reps])
            warm = self.forward(
                warm_ids, pos, scale_invariant=self._scale_inv_head and record_top == 0
            )
            warm_out = mx.argmax(warm, axis=-1)
            # ...and the drain's stack + device-to-host copy, which likewise
            # only run on the first chunk.
            np.array(mx.stack([warm_out, warm_out]))
            mx.eval(warm_out)
        if force_tokens is None and not record_top and self._no_engram:
            # Device-resident decode: argmax stays on the GPU and is fed
            # straight back into the embedding table, so the loop makes no
            # host round trip. Every DRAIN_EVERY steps the pending chain is
            # evaluated and folded into `rows`; all drain time lands inside
            # decode_s, so the accounting stays honest (decode_s is wall time
            # for all max_new_tokens steps, drains included). steps[] only
            # records dispatch, which is why it no longer sums to decode_s.
            # Chunking matters: a single eval over the whole run encodes one
            # enormous Metal graph, whose cost grows faster than linearly with
            # run length and made tok/s depend on --max-tokens. A bounded
            # chunk keeps one completion wait per chunk (amortized to ~3us/step
            # at 64) and keeps the graph small. Engram off is required because
            # its CPU hashing needs the ids as a Python list, which would force
            # a sync.
            profiling = getattr(self, "_phase", None) is not None
            if (
                not profiling
                and self._step_cpu
                and self._composed_head() is not None
                and self._head_mt_np is not None
            ):
                # Host step. The composed table is already on the CPU as
                # float32, so decode is just "gather 32 rows, argmax across
                # head_vocab" -- a chain whose cost is dispatch, not
                # arithmetic. Metal pays ~13us per step for two dependent
                # kernel submissions; numpy pays ~2.5us and keeps the table in
                # cache. The host array is a bit-for-bit copy of the same
                # mx.array, so the outputs are unchanged.
                t_total = time.perf_counter()
                tbl = self._head_mt_np
                cur_h = np.asarray([nxt[r] for r in reps], dtype=np.int64)
                grid = np.empty((max_new_tokens, len(reps)), dtype=np.int64)
                # steps[] is a diagnostic, but timing every step costs two
                # perf_counter() calls, ~0.14us against a ~1.3us step: 10% of
                # the measurement, for nothing. So it is sampled instead; each
                # entry is the per-step mean over a window of 32 steps.
                win = 32
                t_win = t_total
                for step in range(max_new_tokens):
                    grid[step] = cur_h
                    # ndarray.take, not tbl[cur_h]: integer-array indexing goes
                    # through numpy's mapiter, which costs ~0.5us more per step
                    # than take's dedicated path. Same rows, same values.
                    cur_h = tbl.take(cur_h, 0).argmax(axis=1)
                    if step & (win - 1) == win - 1:
                        now = time.perf_counter()
                        steps.append((now - t_win) / win)
                        t_win = now
                tail = max_new_tokens - len(steps) * win
                if tail:
                    now = time.perf_counter()
                    steps.append((now - t_win) / tail)
                decode_s = time.perf_counter() - t_total
            else:
                # Device-resident decode: argmax stays on the GPU and is fed
                # straight back into the embedding table, so the loop makes no
                # host round trip. Every DRAIN_EVERY steps the pending chain is
                # evaluated and folded into `rows`; all drain time lands inside
                # decode_s, so the accounting stays honest. steps[] records
                # dispatch only, which is why it no longer sums to decode_s.
                # Chunking matters: a single eval over the whole run encodes
                # one enormous Metal graph whose cost grows faster than
                # linearly with run length. Engram off is required because its
                # CPU hashing needs the ids as a Python list, forcing a sync.
                t_total = time.perf_counter()
                mt = None if profiling else self._composed_head()
                # argmax's output dtype must match the priming entry, or the
                # drain stack would see two element types.
                prim = mx.array(
                    [nxt[r] for r in reps], dtype=mx.argmax(mx.zeros((2, 2)), axis=-1).dtype
                )
                cur = prim
                fed: list[mx.array] = []
                rows: list[np.ndarray] = []
                for _ in range(max_new_tokens):
                    fed.append(cur)
                    t0 = time.perf_counter()
                    logits = (
                        mt[cur]
                        if mt is not None
                        else self.forward(cur, pos, scale_invariant=self._scale_inv_head)
                    )
                    cur = mx.argmax(logits, axis=-1)
                    steps.append(time.perf_counter() - t0)
                    pos += 1
                    if len(fed) >= DRAIN_EVERY:
                        # One stack then one device-to-host copy per chunk: an
                        # individual np.array() per step pays the copy's fixed
                        # latency 4096 times (~6ms of a 67ms decode).
                        rows.append(np.array(mx.stack(fed)))
                        fed.clear()
                if fed:
                    rows.append(np.array(mx.stack(fed)))
                decode_s = time.perf_counter() - t_total
                grid = np.concatenate(rows) if len(rows) > 1 else rows[0]
            for i in range(n):
                out[i] = [int(v) for v in grid[:, slot_of[i]]]
        else:
            for step in range(max_new_tokens):
                if force_tokens is not None:
                    nxt = [force_tokens[i][step] for i in range(n)]
                for i, tok in enumerate(nxt):
                    out[i].append(tok)
                sub = [nxt[r] for r in reps]
                t0 = time.perf_counter()
                logits = self.forward(
                    list(sub), pos, scale_invariant=self._scale_inv_head and record_top == 0
                )
                mx.eval(logits)
                dt = time.perf_counter() - t0
                decode_s += dt
                steps.append(dt)
                if record_top:
                    lg = _np_f32(logits)[slot_of]
                    k = min(int(record_top), lg.shape[-1])
                    part = np.argpartition(-lg, k - 1, axis=-1)[:, :k]
                    vals = np.take_along_axis(lg, part, axis=-1)
                    order = np.argsort(-vals, axis=-1)
                    tops.append(
                        (
                            np.take_along_axis(part, order, axis=-1),
                            np.take_along_axis(vals, order, axis=-1),
                        )
                    )
                sub_next = [int(v) for v in np.argmax(_np_f32(logits), axis=-1)]
                nxt = [sub_next[s] for s in slot_of]
                pos += 1
        self.end_batch()
        tokens = n * max_new_tokens
        return {
            "batch": n,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "steps_s": steps,
            "tokens": tokens,
            "tok_s": tokens / decode_s if decode_s else 0.0,
            "per_seq_tok_s": (max_new_tokens / decode_s) if decode_s else 0.0,
            "decode_io_s": ld.io_s - pre_io,
            "decode_pipeline_read_worker_s": (ld.pipeline_read_worker_s - pre_pipe_read),
            "decode_pipeline_pack_worker_s": (ld.pipeline_pack_worker_s - pre_pipe_pack),
            "decode_bytes": ld.bytes_read - pre_bytes,
            "decode_load_calls": ld.load_calls - pre_calls,
            "pre_attn_predictions": self._pre_attn_predictions - pre_route[0],
            "pre_attn_cache_hits": self._pre_attn_cache_hits - pre_route[1],
            "pre_attn_correct": self._pre_attn_correct - pre_route[2],
            "pre_attn_wrong": self._pre_attn_wrong - pre_route[3],
            "pre_attn_reads": self._pre_attn_reads - pre_route[4],
            "pre_attn_used": self._pre_attn_used - pre_route[5],
            "pre_attn_used_bytes": self._pre_attn_used_bytes - pre_route[6],
            "pre_attn_wasted": self._pre_attn_wasted - pre_route[7],
            "pre_attn_wasted_bytes": self._pre_attn_wasted_bytes - pre_route[8],
            "pre_attn_read_worker_s": self._pre_attn_read_worker_s - pre_route[9],
            "pre_attn_pack_worker_s": self._pre_attn_pack_worker_s - pre_route[10],
            "expert_cache_hits": self.expert_cache_hits - pre_hits,
            "expert_loads": self.expert_loads - pre_loads,
            "route_coverage": self.route_coverage(),
            "top_logits": tops,
            "outputs": out,
        }

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
        main_x = F.rmsnorm(
            main_x, self._dense["mtp.0.main_norm.weight"], cfg.get("norm_eps", 1e-20)
        )
        dummy = mx.zeros((main_x.shape[0], cfg["dim"]))
        for stage in range(int(cfg.get("n_mtp_layers", 0))):
            self.dspark_attention(stage, dummy, 0, main_x)
            mx.eval(self._mtp_window[stage])

    def mtp_draft(self, main_hidden: mx.array, input_ids: list[int], start_pos: int) -> list[int]:
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
        main_x = F.rmsnorm(
            main_x, self._dense["mtp.0.main_norm.weight"], cfg.get("norm_eps", 1e-20)
        )  # [1, dim]
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
                stage, h, start_pos, pre_mix, ns="mtp", main_x=main_x, top_k=top_k
            )
        # forward_head: hc_pre -> mtp.2.norm -> shared head + markov bias
        x = F.hc_pre(h, pre_mix)  # [B, dim]
        x = F.rmsnorm(x, self._dense["mtp.2.norm.weight"], cfg.get("norm_eps", 1e-20))
        if self._head_quant:
            logits = self._head_q_matmul(x)
        else:
            wt = getattr(self, "_head_wt", None)
            if wt is None:
                bf = getattr(self, "_head_wt_bf", None)
                if bf is not None:  # head_logits ran in bf16 mode
                    wt = bf.astype(mx.float32)
                else:
                    wt = self._dense["head.weight"].astype(mx.float32).T
                    if "head.weight" in self._dense:
                        del self._dense["head.weight"]
                mx.eval(wt)
                self._head_wt = wt
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
            row = _np_f32(logits[i]) + _np_f32(bias)
            # temperature 0 (greedy draft) matches baseline generate default
            nxt = int(np.argmax(row))
            drafts.append(nxt)
            cur = nxt
        return drafts

    def generate_spec(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        batch_probe: bool = False,
    ) -> dict:
        """Lossless greedy DSpark speculative decode (#29).

        Draft block_size tokens, then validate and commit them using the
        serial target path. ``batch_probe`` optionally runs an extra batched
        verifier for diagnostic comparison; its outputs never determine the
        committed tokens or cache state.
        """
        ids = list(prompt_ids)
        times: dict = {}
        stats = {
            "proposed": 0,
            "accepted": 0,
            "steps": 0,
            "draft_s": 0.0,
            "verify_s": 0.0,
            "target_s": 0.0,
            "rollbacks": 0,
            "serial_rejects": 0,
            "full_accepts": 0,
            "batch_probe_enabled": bool(batch_probe),
        }
        start = time.perf_counter()
        logits, main_hidden = self.forward(ids, 0, want_main=True)
        mx.eval(logits, main_hidden)
        times["prefill_s"] = time.perf_counter() - start
        self.mtp_seed(main_hidden)

        out_tokens: list[int] = []
        nxt = self._sample(_np_f32(logits[-1]), temperature)
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
            t0 = self._sample(_np_f32(logits[-1]), temperature)
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

            chunks = [t0] + drafts
            L = pos + 1
            if batch_probe:
                # Diagnostic only: the multi-token results do not commit tokens
                # or cache state. Preserve this arm to quantify the extra work.
                snap = self._snapshot_caches(include_hash_state=True)
                tv = time.perf_counter()
                v_logits, v_main = self.forward(chunks, L, want_main=True)
                mx.eval(v_logits, v_main)
                stats["verify_s"] += time.perf_counter() - tv

                matched_drafts = 0
                for j, draft in enumerate(drafts):
                    pred = int(np.argmax(_np_f32(v_logits[j])))
                    stats.setdefault("draft_hit", [0, 0, 0, 0, 0])
                    stats.setdefault("draft_n", [0, 0, 0, 0, 0])
                    if j < len(stats["draft_n"]):
                        stats["draft_n"][j] += 1
                        if draft == pred:
                            stats["draft_hit"][j] += 1
                    if draft == pred:
                        matched_drafts += 1
                    else:
                        break
                accepted = 1 + matched_drafts  # includes the target's t0
                self._restore_caches(snap)
                if accepted != len(chunks):
                    stats["rollbacks"] += 1

            # The serial target path alone determines committed output and
            # state, whether or not the optional diagnostic batch was run.
            commit = [chunks[0]]
            for offset, token in enumerate(commit):
                tt = time.perf_counter()
                logits, main_hidden = self.forward([token], L + offset, want_main=True)
                mx.eval(logits, main_hidden)
                stats["target_s"] += time.perf_counter() - tt
            exact_matches = 0
            serial_reject = False
            for draft_index, draft in enumerate(drafts):
                pred = int(np.argmax(_np_f32(logits[-1])))
                stats.setdefault("serial_draft_n", 0)
                stats.setdefault("serial_draft_hit", 0)
                stats.setdefault("serial_draft_n_by_idx", [0] * len(drafts))
                stats.setdefault("serial_draft_hit_by_idx", [0] * len(drafts))
                stats["serial_draft_n"] += 1
                stats["serial_draft_n_by_idx"][draft_index] += 1
                if draft != pred:
                    commit.append(pred)
                    serial_reject = True
                    token = pred
                else:
                    exact_matches += 1
                    stats["serial_draft_hit"] += 1
                    stats["serial_draft_hit_by_idx"][draft_index] += 1
                    commit.append(draft)
                    token = draft
                tt = time.perf_counter()
                logits, main_hidden = self.forward([token], L + len(commit) - 1, want_main=True)
                mx.eval(logits, main_hidden)
                stats["target_s"] += time.perf_counter() - tt
                if serial_reject:
                    break
            stats["accepted"] += exact_matches
            if serial_reject:
                stats["serial_rejects"] += 1
            if not serial_reject:
                stats["full_accepts"] += 1
            out_tokens.extend(commit)
            ids.extend(commit)
            if self._trace is not None:
                for t in commit:
                    self._trace.token(t)
            if len(out_tokens) > max_new_tokens:
                out_tokens = out_tokens[:max_new_tokens]
                ids = ids[: len(prompt_ids) + max_new_tokens]

            stats["steps"] += 1
            decode_s += time.perf_counter() - step_t0

        times["decode_s"] = decode_s
        times["tokens"] = len(out_tokens)
        times["tok_s"] = len(out_tokens) / decode_s if decode_s else 0.0
        acc = stats["accepted"]
        prop = stats["proposed"]
        stats["acceptance"] = acc / prop if prop else 0.0
        return {"tokens": out_tokens, "timings": times, "spec": stats}

    def generate(
        self, prompt_ids: list[int], max_new_tokens: int, temperature: float = 0.0
    ) -> dict:
        """Prefill + greedy/sampled decode with timing + trace events."""
        ids = list(prompt_ids)
        times = {}
        start = time.perf_counter()
        self._use_prefill_cache()
        logits = self.forward(ids, 0)
        mx.eval(logits)
        times["prefill_s"] = time.perf_counter() - start
        self._use_decode_cache()
        out_tokens = []
        decode_s = 0.0
        ld = self._expert_loader
        pre_io, pre_pack, pre_up, pre_bytes, pre_calls = (
            ld.io_s,
            ld.dequant_s,
            ld.eval_s,
            ld.bytes_read,
            ld.load_calls,
        )
        pre_pipe_read = ld.pipeline_read_worker_s
        pre_pipe_pack = ld.pipeline_pack_worker_s
        pre_engram = self._engram_s
        pre_hits = self._engram_cache.page_hits
        pre_miss = self._engram_cache.page_misses
        pre_ebytes = self._engram_cache.bytes_read
        pre_rows = self._engram_cache.rows_served
        steps: list[float] = []
        for _ in range(max_new_tokens):
            last = _np_f32(logits[-1])
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
            dt = time.perf_counter() - step
            decode_s += dt
            steps.append(dt)
            if self._trace is not None:
                self._trace.token(nxt)
        times["decode_s"] = decode_s
        times["steps_s"] = steps
        times["tokens"] = len(out_tokens)
        times["tok_s"] = len(out_tokens) / decode_s if decode_s else 0.0
        times["decode_io_s"] = ld.io_s - pre_io
        times["decode_pack_s"] = ld.dequant_s - pre_pack
        times["decode_upload_s"] = ld.eval_s - pre_up
        times["decode_pipeline_read_worker_s"] = ld.pipeline_read_worker_s - pre_pipe_read
        times["decode_pipeline_pack_worker_s"] = ld.pipeline_pack_worker_s - pre_pipe_pack
        times["decode_bytes"] = ld.bytes_read - pre_bytes
        times["decode_expert_loads"] = ld.load_calls - pre_calls
        times["decode_engram_s"] = self._engram_s - pre_engram
        times["engram_page_hits"] = self._engram_cache.page_hits - pre_hits
        times["engram_page_misses"] = self._engram_cache.page_misses - pre_miss
        times["engram_bytes_read"] = self._engram_cache.bytes_read - pre_ebytes
        times["engram_rows"] = self._engram_cache.rows_served - pre_rows
        times["expert_loads"] = self.expert_loads
        times["expert_cache_hits"] = self.expert_cache_hits
        times["expert_evictions"] = self.expert_evictions
        total = self.expert_loads + self.expert_cache_hits
        times["expert_hit_rate"] = self.expert_cache_hits / total if total else 0.0
        return {"tokens": out_tokens, "timings": times}
