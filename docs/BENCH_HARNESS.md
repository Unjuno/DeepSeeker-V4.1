# Golden correctness/benchmark harness (Issue #17)

Backend-agnostic oracle: fixed corpus + sampling, TTFT/prefill/decode
split, repeats as median/range, machine-readable results
(`deepseeker.bench/v1`), and equivalence verdicts (PASS / UNCERTAIN /
FAIL). The minimum test — an intentional perturbation must FAIL — runs
in CI without any model.

## Usage

    python scripts/run_bench.py --backend mlx-floor --record-golden  # once
    python scripts/run_bench.py --backend mlx-floor                  # PASS expected
    python scripts/run_bench.py --backend mlx-floor --perturb        # FAIL expected
    python scripts/run_bench.py --backend mlx-floor --repeats 5 --out result.json

`mlx-floor` output is deterministic pseudo-tokens (synthetic:true): it
validates the harness plumbing, not model quality. Golden files live in
`profiles/bench/golden-<backend>.json` and are committed.

## Equivalence rules

- Tokens decide: exact match within `max_mismatch` (default 0).
- Logits refine: ≤1e-3 PASS, ≤1e-2 UNCERTAIN (backend drift band), else FAIL.
- No logits: token-only verdict capped at UNCERTAIN — never claim PASS
  without the stronger signal.

## Plugging a real backend

Implement `generate_case(case, sampling, backend)` returning
`{tokens, ttft_s, decode_tok_s}` for `reference` / `mlx-stream`, record a
new golden, and the same gates apply. Re-record goldens with
`--record-golden`; never relax thresholds to pass.
