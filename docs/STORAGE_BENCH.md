# Storage-patterns benchmark (physical I/O baseline)

## Run

```bash
python scripts/benchmark_storage_patterns.py
```

Options: `--file-mib` (default 8192), `--repeats` (default 3),
`--sustain-seconds` (default 60), `--no-verify` (skip the ~10 KB
upstream range check), `--assert-cold "note"` (see below).

Output: `profiles/m1-max-64gb/storage-patterns.json` (+ `.md` summary).

## What it measures

- Single experts (early/mid/late/min-span/max-span, exact six ranges
  from the manifest): exact, 4K/16K-aligned, and 4K-gap-coalesced
  modes at QD1/2/4/8/16.
- Expert groups 1/2/4/8/16/32 (same-layer, cross-layer, random).
- Engram per-token coalesced plans (lengths/counts exact, placement
  scattered: true span ~98 GiB/table exceeds any disposable file).
- Sustained 8-expert read loop (60 s, 10 s buckets) at QD1/QD4.

## Key findings (2026-09-18, advisory-uncached)

- One expert's six tensors live in ONE shard but span 576 MB–6.6 GB
  of file offset. Gap-coalescing across the span is HARMFUL
  (2.5–2.9 ms vs ~1.0 ms exact): fetch the six ranges, don't merge
  across the span.
- Single expert exact QD1 p50 ~0.9–1.0 ms (cache-speed upper bound).
- Groups scale QD1 ~4.7 → QD16 ~22 GiB/s (cache ceiling, not SSD).
- Engram token: 96 ops, QD1 ~8.3 ms → QD16 ~0.86 ms (~10x hiding).
- Sustained 60 s buckets flat (no throttling in the cache path).

## Cache-confidence discipline

Default results are **advisory-uncached**: F_NOCACHE + randomized
placement on an 8 GiB file that fits in 64 GiB RAM, so page-cache
service cannot be excluded. Throughputs above ~5 GiB/s prove cache
service. These numbers are valid UPPER bounds and valid concurrency/
geometry measurements — not SSD latency.

Optional cold procedure (manual, outside the script — it never
invokes sudo itself):

```bash
sudo purge
python scripts/benchmark_storage_patterns.py \
    --assert-cold "sudo purge at <UTC time> by <operator>"
```

Only runs with `--assert-cold` are labeled `cold-proven`, with the
note recorded in the JSON. No cold run has been performed yet.

## Wear policy

The disposable file (default 8 GiB seeded-random + fsync) is written
once and reused across runs; read tests write nothing. Larger files
(`--file-mib 71680`) raise the miss rate but cost proportional SSD
writes — use only deliberately.
