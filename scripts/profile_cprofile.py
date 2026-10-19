#!/usr/bin/env python3
"""cProfile the decode loop to find Python hotspots."""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream" / "DeepSeek-V4.1-Flash"))

import mlx.core as mx
import numpy as np
from encoding.encoding import encode_messages
from transformers import AutoTokenizer

from deepseeker.baseline import load_json
from deepseeker.mlx_runner import Runner

GOLDEN = [43, 4571, 2887, 304, 2507, 1613, 539, 43]


def main() -> int:
    snap = REPO / "upstream" / "DeepSeek-V4.1-Flash"
    rev = load_json(snap / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(REPO / "profiles" / "deepseek-v4.1-flash" / rev / "model-manifest.json")
    config = load_json(snap / "inference" / "config.json")
    root = REPO / "models" / "DeepSeek-V4.1-Flash"
    tok = AutoTokenizer.from_pretrained(str(root))
    ids = tok.encode(encode_messages([{"role": "user", "content": "Say hello."}], "chat"))

    r = Runner(root, manifest, config, expert_cache_cap=256)
    try:
        base = r.generate(ids, 8, temperature=0.0)
        assert base["tokens"] == GOLDEN, base["tokens"]
        print(f"BASE ok tok/s={base['timings']['tok_s']:.4f}", flush=True)

        r._window.clear()
        r._compress.clear()
        r._last_route.clear()

        def run_decode():
            logits = r.forward(ids, 0)
            mx.eval(logits)
            toks = []
            for i in range(8):
                nxt = int(np.argmax(np.asarray(logits[-1], dtype=np.float32)))
                toks.append(nxt)
                logits = r.forward([nxt], len(ids) + i)
                mx.eval(logits)
            return toks

        # Warm graph already done by base generate; clear caches for fair decode
        r._window.clear()
        r._compress.clear()
        r._last_route.clear()
        r._expert_protected = set()
        r._expert_used_now = set()

        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        toks = run_decode()
        pr.disable()
        dt = time.perf_counter() - t0
        print(f"PROFILED decode toks={toks} match={toks == GOLDEN} wall={dt:.1f}s", flush=True)

        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(60)
        out = s.getvalue()
        print(out, flush=True)
        prof = REPO / "profiles" / "runs" / "mtp" / "cprofile_decode.txt"
        prof.write_text(out)

        s2 = io.StringIO()
        pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(40)
        out2 = s2.getvalue()
        print(out2, flush=True)
        prof2 = REPO / "profiles" / "runs" / "mtp" / "cprofile_decode_tottime.txt"
        prof2.write_text(out2)
    finally:
        r.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
