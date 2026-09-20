# Async six-range staging (Issue #24, roadmap P4)

Fetches one expert as its real six tensor ranges (w1/w2/w3 + scales),
never span-coalesced across gaps: one pread per range (counted in
tests). QD-controlled workers, dedup to in-flight tickets, best-effort
cancel, manifest byte validation, and a hard staging-byte bound that
raises instead of deadlocking (caller drains, then retries).

## Usage

    python scripts/stage_misses.py --trace /tmp/t.jsonl --slots 6 --qd 8 --max-tokens 4

Miss path: pool lookup → staging submit/wait → pool admit. Service
metrics (mean service ms, queue wait, dedup, failures) print per run
and compare against the #19 envelope: one expert ≈ 1–3ms cold-ish.

## Failure semantics

Short reads and manifest mismatches raise `StagingError`; the engine
stays usable (tested with a flaky reader). Cancelled in-flight bytes
are discarded, never admitted. Unknown/settled tickets raise KeyError.
