#!/usr/bin/env python3
"""Q3 pilot quality gate: greedy Q3 text ranked under the FP4 reference.

Free-runs the bench prompts under the Q3 manifest, then replays those exact
token sequences teacher-forced through the FP4 reference and reports where
the reference ranks each Q3 token. The honest quality question (same design
as quality_union.py, crossed over manifests): not "did the greedy path
diverge" (it always does) but "would the FP4 model have said this anyway".

Also reports greedy agreement between the two free runs per prompt (first
divergence step, per-step match fraction).

    DEEPSEEKER_VARIANT=1017 python scripts/quality_q3.py --batch 8 --steps 24
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
from deepseeker.variant_q3 import paths_for_variant


def ranks_of(tokens: list[list[int]], tops: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Exact-model rank of each replayed token (1 = would have been chosen)."""
    out = []
    for step in range(len(tops) - 1):
        part, _ = tops[step]
        for row, seq in enumerate(tokens):
            want = seq[step + 1]
            hit = np.where(part[row] == want)[0]
            out.append(int(hit[0]) + 1 if hit.size else part.shape[1] + 1)
    return np.asarray(out, dtype=np.int64)


def self_check(r: np.ndarray, label: str) -> None:
    if r.max() != 1:
        raise SystemExit(f"{label}: rank plumbing off (max={r.max()}); refusing to report")


def ranks_of_seq(res: dict, fp4_tops=None) -> np.ndarray:
    """Rank every free-run token under reference tops (one prompt at a time).

    ``res['top_logits']`` and the optional ``fp4_tops`` are lists of
    per-prompt step records, as returned by :func:`seq_generate`.
    """
    tops = fp4_tops if fp4_tops is not None else res["top_logits"]
    return np.concatenate(
        [ranks_of([res["outputs"][i]], tops[i]) for i in range(len(res["outputs"]))]
    )


def pair_stats(a: list[list[int]], b: list[list[int]]) -> dict:
    """Greedy agreement of two free runs: first divergence + match fraction."""
    first_div, match = [], []
    for ra, rb in zip(a, b):
        m = min(len(ra), len(rb))
        div = next((i for i in range(m) if ra[i] != rb[i]), m)
        first_div.append(div)
        if m:
            eq = sum(int(x == y) for x, y in zip(ra[:m], rb[:m])) / m
            match.append(eq)
    return {
        "first_divergence": first_div,
        "match_fraction_mean": float(np.mean(match)),
        "match_fraction_min": float(np.min(match)),
        "identical": sum(
            int(d == len(ra) and len(ra) == len(rb)) for d, ra, rb in zip(first_div, a, b)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Q3-vs-FP4 greedy rank harness.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--expert-cap", type=int, default=256)
    ap.add_argument("--record-top", type=int, default=64)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    n = args.batch
    snapshot = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash-1017"
    revision = load_json(snapshot / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
    fp4_root, fp4_manifest = paths_for_variant(revision, REPO_ROOT, snapshot.name)
    q3_root, q3_manifest = paths_for_variant(revision, REPO_ROOT, snapshot.name, variant="1017-Q3")
    config = load_json(snapshot / "inference" / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(str(fp4_root))

    texts = (DEFAULT_PROMPTS * ((n // len(DEFAULT_PROMPTS)) + 1))[:n]
    prompts = [
        tokenizer.encode(encode_messages([{"role": "user", "content": t}], "chat")) for t in texts
    ]
    print(f"q3 quality prompts={n} steps={args.steps}", flush=True)

    def seq_generate(runner: Runner, mode: str, forced=None) -> dict:
        """Generate one prompt at a time under a single loaded runner.

        Batched lockstep thrashes the expert cache for n>1 (per-step working
        set n*6*40 slots vs a 256-slot cache), so quality runs stream each
        sequence alone; per-sequence outputs match batched (deterministic).
        """
        runs = []
        t0 = time.perf_counter()
        for i, p in enumerate(prompts):
            kw = {"record_top": args.record_top}
            if forced is not None:
                kw["force_tokens"] = [forced[i]]
            out = runner.generate_batch([p], args.steps, **kw)
            runs.append(out)
            if (i + 1) % 4 == 0:
                el = time.perf_counter() - t0
                print(f"  {mode} {i + 1}/{n} tok_s={out['tok_s']:.3f} el={el:.0f}s", flush=True)
        return {
            "outputs": [r["outputs"][0] for r in runs],
            "top_logits": [r["top_logits"] for r in runs],
            "tok_s": float(np.mean([r["tok_s"] for r in runs])),
            "decode_bytes": sum(r["decode_bytes"] for r in runs),
            "route_coverage": float(np.mean([r["route_coverage"] for r in runs])),
        }

    # Pass 1: Q3 free run (its greedy text is the candidate under test).
    t0 = time.perf_counter()
    r_q3 = Runner(
        q3_root, load_json(q3_manifest), config, expert_cache_cap=args.expert_cap, max_batch=1
    )
    try:
        q3 = seq_generate(r_q3, "q3")
        self_check(ranks_of_seq(q3), "q3 self-check")
        print(
            f"q3 free coverage={q3['route_coverage']:.4f} "
            f"bytes={q3['decode_bytes'] / 2**30:.2f}GiB "
            f"tok_s={q3['tok_s']:.4f}",
            flush=True,
        )
    finally:
        r_q3.shutdown()
    print(f"q3 pass wall={time.perf_counter() - t0:.0f}s", flush=True)

    # Pass 2: FP4 reference free run + teacher-forced replay of the Q3 text.
    t0 = time.perf_counter()
    r_fp4 = Runner(
        fp4_root, load_json(fp4_manifest), config, expert_cache_cap=args.expert_cap, max_batch=1
    )
    try:
        fp4 = seq_generate(r_fp4, "fp4")
        self_check(ranks_of_seq(fp4), "fp4 self-check")
        print(
            f"fp4 free coverage={fp4['route_coverage']:.4f} "
            f"bytes={fp4['decode_bytes'] / 2**30:.2f}GiB "
            f"tok_s={fp4['tok_s']:.4f}",
            flush=True,
        )
        replay = seq_generate(r_fp4, "replay", forced=q3["outputs"])
        r = ranks_of_seq(q3, fp4_tops=replay["top_logits"])
    finally:
        r_fp4.shutdown()
    print(f"fp4 pass wall={time.perf_counter() - t0:.0f}s", flush=True)

    stats = pair_stats(q3["outputs"], fp4["outputs"])
    rec = {
        "batch": n,
        "steps": args.steps,
        "tokens_scored": int(r.size),
        "rank_mean": float(r.mean()),
        "rank_p50": int(np.median(r)),
        "rank_p90": int(np.percentile(r, 90)),
        "rank_max": int(r.max()),
        "top1": float((r == 1).mean()),
        "top5": float((r <= 5).mean()),
        "top32": float((r <= 32).mean()),
        "q3_bytes": q3["decode_bytes"],
        "fp4_bytes": fp4["decode_bytes"],
        "q3_tok_s": q3["tok_s"],
        "fp4_tok_s": fp4["tok_s"],
        "greedy_agreement": stats,
        "fp4_free_texts": [tokenizer.decode(t) for t in fp4["outputs"]],
        "q3_free_texts": [tokenizer.decode(t) for t in q3["outputs"]],
    }
    print(
        f"Q3 text under FP4: rank_mean={rec['rank_mean']:.2f} "
        f"p50={rec['rank_p50']} p90={rec['rank_p90']} top1={rec['top1']:.3f} "
        f"top5={rec['top5']:.3f}",
        flush=True,
    )
    print(
        f"greedy agreement: mean match={stats['match_fraction_mean']:.3f} "
        f"identical={stats['identical']}/{n}",
        flush=True,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
