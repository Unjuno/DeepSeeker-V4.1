#!/usr/bin/env python3
"""Issue #17: run the golden harness against a backend.

    python scripts/run_bench.py --backend mlx-floor --record-golden
    python scripts/run_bench.py --backend mlx-floor
    python scripts/run_bench.py --backend mlx-floor --perturb
    python scripts/run_bench.py --backend mlx-floor --repeats 5 --out result.json

Backends: mlx-floor (stub: deterministic pseudo-tokens + MLX timing probe;
synthetic output, validates the harness end-to-end). Real backends
(reference/mlx-stream) implement generate_case() and plug in here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import load_json
from deepseeker.harness import build_result, equivalence

CORPUS = REPO_ROOT / "profiles" / "bench" / "corpus.json"
GOLDEN_DIR = REPO_ROOT / "profiles" / "bench"


def pseudo_tokens(case_id: str, seed: int, n: int, vocab: int = 129280) -> list[int]:
    """Deterministic stub output (synthetic: validates plumbing, not quality)."""
    out = []
    for i in range(n):
        digest = hashlib.sha256(f"{case_id}:{seed}:{i}".encode()).digest()
        out.append(int.from_bytes(digest[:4], "big") % vocab)
    return out


def mlx_probe_ms() -> float:
    """One quick MLX timing probe: a full SwiGLU expert (3 GEMMs, batch 1).

    One token = 240 experts, so probe ~= 1/240 of MoE math per token.
    """
    import mlx.core as mx

    gate = mx.random.normal((5120, 2304), dtype=mx.bfloat16) * 0.02
    up = mx.random.normal((5120, 2304), dtype=mx.bfloat16) * 0.02
    down = mx.random.normal((2304, 5120), dtype=mx.bfloat16) * 0.02
    x = mx.random.normal((5120,), dtype=mx.bfloat16)
    mx.eval(gate, up, down, x)
    start = time.perf_counter()
    for _ in range(5):
        g = x @ gate
        y = (g * mx.sigmoid(g) * (x @ up)) @ down
        mx.eval(y)
    return (time.perf_counter() - start) / 5 * 1000


def generate_case(case: dict, sampling: dict, backend: str) -> dict:
    """Backend dispatch. Returns {tokens, ttft_s, decode_tok_s}."""
    if backend != "mlx-floor":
        raise ValueError(f"unknown backend {backend!r} (have: mlx-floor)")
    n = int(sampling.get("max_new_tokens", 32))
    seed = int(sampling.get("seed", 0))
    probe_ms = mlx_probe_ms()
    tokens = pseudo_tokens(case["id"], seed, n)
    return {
        "tokens": tokens,
        "ttft_s": probe_ms / 1000,
        "decode_tok_s": 1000.0 / (240 * probe_ms),
        "synthetic": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the golden harness.")
    parser.add_argument("--backend", type=str, default="mlx-floor")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--record-golden", action="store_true")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats < 1:
        print("FAIL: repeats must be >= 1")
        return 2

    corpus = load_json(CORPUS)
    sampling = corpus["sampling"]
    golden_path = GOLDEN_DIR / f"golden-{args.backend}.json"
    if args.record_golden:
        golden = {
            case["id"]: generate_case(case, sampling, args.backend)["tokens"]
            for case in corpus["cases"]
        }
        golden_path.write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
        print(f"recorded {golden_path} (synthetic stub output)")
        return 0

    golden = load_json(golden_path) if golden_path.exists() else None
    ttfts, rates, accepted = [], [], 0
    verdicts = []
    for _ in range(args.repeats):
        for case in corpus["cases"]:
            out = generate_case(case, sampling, args.backend)
            tokens = out["tokens"]
            if args.perturb:
                tokens = [tokens[0] ^ 0x1, *tokens[1:]]
            ttfts.append(out["ttft_s"])
            rates.append(out["decode_tok_s"])
            accepted += len(tokens)
            if golden is None:
                verdicts.append("UNCERTAIN")
            else:
                verdicts.append(
                    equivalence(golden[case["id"]], tokens)["verdict"]
                )
    verdict = "FAIL" if "FAIL" in verdicts else ("PASS" if golden else "UNCERTAIN")
    machine = load_json(REPO_ROOT / "profiles" / "m1-max-64gb" / "machine.json")
    revision = load_json(
        REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    )["resolved_revision"]
    result = build_result(
        machine_id=f"{machine['soc']}/{machine['software']['macos_version']}",
        revision=revision,
        backend=args.backend + ("+perturb" if args.perturb else ""),
        corpus_id=corpus["schema"],
        settings=sampling,
        ttft_s=ttfts,
        decode_tok_s=rates,
        accepted_tokens=accepted,
        equivalence_report={"verdict": verdict, "per_case": verdicts},
        io_counters={},
    )
    print(f"equivalence: {verdict}  accepted={accepted} "
          f"decode_tok_s(median)={result['decode_tok_s']['median']:.1f}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
