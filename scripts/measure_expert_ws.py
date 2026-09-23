#!/usr/bin/env python3
"""Measure unique expert keys over prefill+decode for the golden prompt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream" / "DeepSeek-V4.1-Flash"))

from encoding.encoding import encode_messages
from transformers import AutoTokenizer

from deepseeker.baseline import load_json
from deepseeker.mlx_runner import Runner

rev = load_json(REPO / "upstream/DeepSeek-V4.1-Flash/DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
manifest = load_json(REPO / "profiles" / "deepseek-v4.1-flash" / rev / "model-manifest.json")
config = load_json(REPO / "upstream/DeepSeek-V4.1-Flash/inference" / "config.json")
model_root = REPO / "models" / "DeepSeek-V4.1-Flash"
tok = AutoTokenizer.from_pretrained(str(model_root))
ids = tok.encode(encode_messages([{"role": "user", "content": "Say hello."}], "chat"))

seen: dict[tuple[int, int], int] = {}
per_layer: dict[int, set[int]] = {i: set() for i in range(40)}

runner = Runner(model_root, manifest, config, expert_cache_cap=10**9)
_real_fetch = None
_real_pf = runner.prefetch_experts


def spy_pf(layer: int, experts, ns: str = "layers"):
    for e in experts:
        key = (layer, int(e))
        if key not in seen:
            seen[key] = len(seen)
        if ns == "layers" and 0 <= int(layer) < 40:
            per_layer[int(layer)].add(int(e))
    return _real_pf(layer, experts, ns)


runner.prefetch_experts = spy_pf  # type: ignore

# Also wrap _expert_fn fetch via moe path: patch load_expert
_real_load = runner._expert_loader.load_expert


def spy_load(layer: int, expert: int, prefix: str = "layers"):
    if prefix == "layers":
        key = (int(layer), int(expert))
        if key not in seen:
            seen[key] = len(seen)
        if 0 <= int(layer) < 40:
            per_layer[int(layer)].add(int(expert))
    return _real_load(layer, expert, prefix)


runner._expert_loader.load_expert = spy_load  # type: ignore

try:
    out = runner.generate(ids, 8, temperature=0.0)
finally:
    runner.shutdown()

per_layer_counts = {k: len(v) for k, v in per_layer.items()}
print(f"tokens={out['timings']['tokens']} tok_s={out['timings']['tok_s']:.4f}")
print(f"unique expert keys={len(seen)}")
print(f"loads={runner.expert_loads} hits={runner.expert_cache_hits} "
      f"evict={runner.expert_evictions}")
print("per-layer unique:", sorted(per_layer_counts.values()))
print(f"max/layer={max(per_layer_counts.values())} min/layer={min(per_layer_counts.values())}")
print(f"VERDICT unique={len(seen)} vs cap256 thrash={len(seen)>256} "
      f"vs cap400 thrash={len(seen)>400} vs cap960 thrash={len(seen)>960}")
# dump for later
Path("/tmp/expert_ws.json").write_text(json.dumps({
    "unique": len(seen),
    "per_layer": {str(k): sorted(v) for k, v in per_layer.items()},
    "tok_s": out["timings"]["tok_s"],
    "loads": runner.expert_loads,
    "hits": runner.expert_cache_hits,
    "evictions": runner.expert_evictions,
}, indent=1))
print("wrote /tmp/expert_ws.json")
