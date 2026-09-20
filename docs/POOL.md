# Bounded resident expert pool (Issue #23, roadmap P4)

Real pool, real bytes: experts load from the verified checkpoint via
manifest `file_range` preads into numpy uint8 buffers in unified memory.
GPU-side residency is P7; placement/admission policy here is
device-independent.

## Guarantees (all tested)

- Hard ceiling: `bytes_resident <= slots x max_expert_bytes`, asserted on
  every admit; stress fill (50 experts into 2 slots) holds.
- No stale use: evicted entries unreachable; served bytes verified
  byte-equal to fresh preads (`run_pool.py --verify-every N`).
- Pin safety: pinned entries never evicted; full-pinned admit raises
  instead of violating.
- Miss fallback: demand paging (miss → load → admit → serve).

## Usage

    python scripts/run_pool.py --trace /tmp/t.jsonl --slots 6 --policy lru
    python scripts/run_pool.py --trace /tmp/t.jsonl --slots 6 --max-tokens 8 --json-out pool.json

`--max-tokens` bounds checkpoint IO (full traces re-read gigabytes).
Trace replay uses the schema only — synthetic until #20 unblocks; the
pool cannot distinguish them, which is the point of the interface.

## Policies

LRU (default) / LFU / FIFO. No prefetch here (#25), no kernels (#33).
Per-layer slot maps accepted (`--slots` is uniform today; dict form is
in the API for #13 allocations).
