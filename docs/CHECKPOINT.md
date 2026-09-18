# Checkpoint acquisition (pinned revision only)

## Commands

```bash
python scripts/manage_checkpoint.py preflight
python scripts/manage_checkpoint.py download
python scripts/manage_checkpoint.py verify
python scripts/manage_checkpoint.py status [--write-status]
```

Model root precedence: `--model-root` > `DEEPSEEKER_MODEL_ROOT` >
default `models/DeepSeek-V4.1-Flash` (Git-ignored; enforced, not assumed).

## Method

- Expected file list comes from the Issue #2 manifest (48 shards +
  index/tokenizer/config), never hardcoded alone.
- Direct HTTP Range downloads to final paths: exactly ONE physical copy,
  no HF-cache duplicate (measured amplification ~1.0x + state bytes).
- Per-shard resume from `.part` size; SHA-256 is hashed while streaming
  and compared to upstream LFS digests (`get_paths_info`), so no second
  verification read is needed.
- `verify` re-checks size + recorded digest + full header/tensor
  reconciliation against the manifest and marks shards verified.
- `status` prints sanitized state; `--write-status` commits only
  `profiles/m1-max-64gb/checkpoint-status.json` (revision, counts,
  bytes, method — no absolute paths).

## Interruption / recovery

`download` handles SIGTERM/SIGINT gracefully (partial `.part` kept) and
is idempotent: completed shards are reused by size + recorded digest,
interrupted shards resume from current `.part` size. Demonstrated by
killing the downloader mid-shard and rerunning (monotonic byte growth,
zero re-download).

## Git safety

`src/deepseeker/checkpoint.py::git_safety_scan` fails on any tracked or
staged `*.safetensors`, `models/`, `.cache/`, or >100 MB non-allowlisted
file. `models/`, `*.safetensors`, `.cache/` are gitignored.

## Measured first materialization (2026-09-18)

- Expected: 510,310,549,975 remote bytes (48 shards + 4 small files).
- Free space at preflight: ~2.6 TB; headroom ~2.1 TB.
- Observed throughput ~8 MiB/s aggregate (route-limited, 8 streams);
  full download is a multi-hour background task, polled via the state
  file and `status`.
