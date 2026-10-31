#!/usr/bin/env python3
"""MoE union-budget quality sweep: what does capping the batch's expert union
cost us in output distribution, and what does it buy in bytes?

The exact run is recorded first; every budgeted run is then decoded under
teacher forcing on that reference (identical inputs, identical states), so the
only difference in the logits is the budget itself. Reports:

  coverage  share of total gate weight kept by cap_union (quality floor)
  flip      share of (step, row) whose greedy token changed anyway
  kl        mean KL(P_exact || P_budget) under identical inputs
  margin    mean logit gap of the exact token under the budgeted model

    python scripts/sweep_union.py --batch 8 --steps 16 --budgets 0,24,16,12,8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench_batch import DEFAULT_PROMPTS
from encoding.encoding import encode_messages
from transformers import AutoTokenizer

from deepseeker.baseline import load_json
from deepseeker.mlx_runner import Runner


def _logsumexp(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))


def compare(ref: list[tuple[np.ndarray, np.ndarray]],
            cand: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    """(KL, flip rate, mean margin) between two teacher-forced logit traces."""
    kl_sum = flips = margin = n = 0.0
    for (i_ref, v_ref), (i_can, v_can) in zip(ref, cand):
        rows = v_ref.shape[0]
        n += rows
        flips += float((i_ref[:, 0] != i_can[:, 0]).sum())
        # does the budgeted model still put the exact token first?
        hit = i_can[:, :1] == i_ref[:, :1]
        where = np.argmax(i_can == i_ref[:, :1], axis=-1)
        picked = np.take_along_axis(v_can, where[:, None], axis=-1)[:, 0]
        margin += float((np.where(hit[:, 0], v_can[:, 0], picked)
                         - v_can[:, 0]).sum())

        p = np.exp(v_ref - _logsumexp(v_ref))
        logp = np.log(p)
        # candidate logit for each reference token; unseen => below its top-K
        match = i_can[:, :, None] == i_ref[:, None, :]
        found = match.any(axis=1)
        slot = np.argmax(match, axis=1)
        looked = np.take_along_axis(v_can, slot, axis=-1)
        floor = v_can[:, -1:] - 5.0
        logq = (np.where(found, looked, floor) - _logsumexp(v_can))
        kl_sum += float((p * (logp - logq)).sum(axis=1).sum())
    return kl_sum / n if n else 0.0, flips / n if n else 0.0, margin / n if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Union-budget quality sweep.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--expert-cap", type=int, default=256)
    ap.add_argument("--budgets", type=str, default="0,24,16,12,8")
    ap.add_argument("--record-top", type=int, default=32)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    budgets = [int(v) for v in args.budgets.split(",") if v.strip()]
    if 0 not in budgets:
        budgets = [0] + budgets
    n = args.batch

    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    manifest = load_json(
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json")
    config = load_json(snapshot / "inference" / "config.json")
    model_root = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash-1017"
    tokenizer = AutoTokenizer.from_pretrained(str(model_root))

    texts = (DEFAULT_PROMPTS * ((n // len(DEFAULT_PROMPTS)) + 1))[:n]
    toks = [tokenizer.encode(encode_messages([{"role": "user", "content": t}], "chat"))
            for t in texts]
    common = min(len(t) for t in toks)
    prompts = [t[:common] for t in toks]
    print(f"sweep batch={n} steps={args.steps} cap={args.expert_cap} "
          f"budgets={budgets} top={args.record_top}", flush=True)

    runner = Runner(model_root, manifest, config,
                    expert_cache_cap=args.expert_cap, max_batch=max(n, 1))
    ref: dict | None = None
    rows = []
    for budget in budgets:
        runner._union_budget = budget
        force = ref["outputs"] if (budget and ref) else None
        t0 = time.perf_counter()
        res = runner.generate_batch(prompts, args.steps, force_tokens=force,
                                    record_top=args.record_top)
        wall = time.perf_counter() - t0
        rec = {"budget": budget, "coverage": res["route_coverage"],
               "tok_s": res["tok_s"], "decode_s": res["decode_s"],
               "bytes": res["decode_bytes"], "wall_s": wall}
        if budget == 0:
            ref = res
            rec.update(flip=0.0, kl=0.0, margin=0.0)
        else:
            kl, flip, margin = compare(ref["top_logits"], res["top_logits"])
            rec.update(flip=flip, kl=kl, margin=margin)
        rows.append(rec)
        print(f"budget={budget:3d} coverage={rec['coverage']:.4f} "
              f"bytes={rec['bytes'] / 2**30:7.2f}GiB "
              f"tok_s={rec['tok_s']:.4f} flip={rec['flip']:.4f} "
              f"kl={rec['kl']:.5f} margin={rec['margin']:+.4f}", flush=True)
    runner.shutdown()

    base = next(r for r in rows if r["budget"] == 0)
    for rec in rows:
        rec["bytes_speedup"] = base["bytes"] / rec["bytes"] if rec["bytes"] else 0.0
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
