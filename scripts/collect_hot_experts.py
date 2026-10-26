#!/usr/bin/env python3
"""Collect per-layer router-frequency hot experts for the 1017 revision.

Spy-wraps Runner.prefetch_experts across a mixed bench corpus (prompt at a
time), aggregates (layer, expert) decode counts, writes
profiles/<1017-revision>/hot_experts.json in the 0905 shape:
{"trace", "decode_records", "n_layers", "topk", "per_layer": [[e0..e63]...]}
(top-64 per layer, descending frequency).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream" / "DeepSeek-V4.1-Flash"))
sys.path.insert(0, str(REPO / "scripts"))

from bench_batch import DEFAULT_PROMPTS
from encoding.encoding import encode_messages
from transformers import AutoTokenizer

from deepseeker.baseline import load_json
from deepseeker.mlx_runner import Runner

STEPS = 16
N_PROMPTS = 32
TOPK = 64

snapshot = REPO / "upstream" / "DeepSeek-V4.1-Flash-1017"
revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
manifest = load_json(
    REPO / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json")
config = load_json(snapshot / "inference" / "config.json")
model_root = REPO / "models" / snapshot.name
tok = AutoTokenizer.from_pretrained(str(model_root))

prompts = [
    tok.encode(encode_messages([{"role": "user", "content": t}], "chat"))
    for t in DEFAULT_PROMPTS[:N_PROMPTS]
]

counts: dict[tuple[int, int], int] = Counter()
runner = Runner(model_root, manifest, config, expert_cache_cap=256)
_real_pf = runner.prefetch_experts


def spy_pf(layer: int, experts, ns: str = "layers", **kw):
    if ns == "layers" and 0 <= int(layer) < 40:
        for e in experts:
            counts[(int(layer), int(e))] += 1
    return _real_pf(layer, experts, ns, **kw)


runner.prefetch_experts = spy_pf  # type: ignore

t0 = time.perf_counter()
for i, ids in enumerate(prompts):
    try:
        runner.generate_batch([ids], STEPS)
    except Exception as exc:  # noqa: BLE001
        # stay in cache state machine: reset between prompts regardless
        runner.reset_seq_state()
        print(f"prompt {i} failed: {exc}", flush=True)
    if (i + 1) % 4 == 0:
        print(f"{i + 1}/{len(prompts)} tok={sum(counts.values())} "
              f"el={time.perf_counter() - t0:.0f}s", flush=True)
runner.shutdown()

per_layer: list[list[int]] = []
for layer in range(40):
    hist = Counter({e: c for (ly, e), c in counts.items() if ly == layer})
    top = [e for e, _ in hist.most_common(TOPK)]
    top += [e for e in range(384) if e not in set(top)][: max(0, TOPK - len(top))]
    per_layer.append(top[:TOPK])

out = REPO / "profiles" / "deepseek-v4.1-flash" / revision / "hot_experts.json"
out.write_text(json.dumps({
    "trace": "collect_hot1017",
    "decode_records": sum(counts.values()),
    "n_layers": 40,
    "topk": TOPK,
    "per_layer": per_layer,
}) + "\n")
uniq = {e for ly in per_layer for e in ly}
print(f"wrote {out} records={sum(counts.values())} uniq={len(uniq)}", flush=True)
