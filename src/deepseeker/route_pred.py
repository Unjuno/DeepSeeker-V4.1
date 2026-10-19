"""Measure how well a pre-attention signal predicts MoE top-1 routes.

Decode is strictly serial today: layer L blocks on expert SSD reads
(`prefetch_experts` -> `_io_pool.map`) that can only be issued *after* the
attention that produces `xin2`. Measured floors (B=8, 16 steps):

    GPU only (DEEPSEEKER_FAKE_EXPERTS=1):  decode  7.67s  16.70 tok/s
    full model:                            decode 20.53s   6.24 tok/s
    of which blocking expert I/O:                     8.82s

So layer L's ~13.8ms of SSD could be hidden behind layer L's ~12ms of
attention *if* routes were known at the layer's start. This harness scores
two candidate early signals against the authoritative route:

    xin  - rmsnorm(hc_pre(x0)) computed before attention (the best early
           signal: available the moment the layer opens)
    xpre - hc_pre(x0) without the attention RMSNorm

Reported numbers are what a speculative prefetch could actually achieve:
    top1  - share of rows where predicted top-1 == authoritative top-1
            (a hit means no extra bytes; a miss costs one wasted read)
    top2  - share where the true top-1 is inside the predicted top-2
"""

from __future__ import annotations

import atexit

import mlx.core as mx
import numpy as np


class RoutePred:
    def __init__(self) -> None:
        self.n = 0
        self.hits: dict[str, int] = {}
        self.top2: dict[str, int] = {}
        self.samples: dict[str, int] = {}
        self.steps = 0
        self._stash: dict[str, mx.array] = {}
        atexit.register(self.report)

    def stash(self, key: str, value: mx.array) -> None:
        self._stash[key] = value

    def observe(self, gate_w: mx.array, gate_b: mx.array,
                actual_idx: np.ndarray) -> None:
        if actual_idx.ndim != 2 or actual_idx.shape[1] < 1:
            return
        rows = actual_idx.shape[0]
        for key in sorted(self._stash):
            v = self._stash[key]
            if v is None or v.ndim != 2 or v.shape[0] != rows \
                    or v.shape[1] != gate_w.shape[1]:
                continue
            scores = (v.astype(mx.float32) @ gate_w.astype(mx.float32).T) \
                + gate_b.astype(mx.float32)
            order = mx.argsort(scores, axis=-1)
            mx.eval(order)
            o = np.asarray(order)
            if self.n == 0:
                print(f"route_pred dbg {key}: v={v.shape} gw={gate_w.shape} "
                      f"scores={scores.shape} order={o.shape} "
                      f"actual={actual_idx.shape}", flush=True)
            top1 = o[:, -1]
            true = actual_idx[:, 0]
            self.hits.setdefault(key, 0)
            self.top2.setdefault(key, 0)
            self.samples.setdefault(key, 0)
            self.hits[key] += int((top1 == true).sum())
            self.top2[key] += int((o[:, -2:] == true[:, None]).any(axis=1).sum())
            self.samples[key] += rows
            self.n += rows
        self.steps += 1

    def report(self) -> None:
        if not self.n:
            print("route_pred: no samples", flush=True)
            return
        print(f"\nroute_pred samples={self.n} rows/route-events={self.steps}",
              flush=True)
        for key in sorted(self.hits):
            denom = self.samples[key]
            t1 = self.hits[key] / denom
            t2 = self.top2[key] / denom
            # bytes a speculative prefetch costs, on top of the real read
            print(f"  {key:4s} top1={t1:.4f} top2={t2:.4f} "
                  f"pred_bytes_factor={1 + (1 - t1):.3f}", flush=True)
