# Engram access summary

Revision: `dba1be0a40aa45a94ad051997016db3960a90277`
Generated: 2026-09-18T01:43:43.927812+00:00

## Verified geometry (derived from pinned config + manifest)

- Engram layers: [1, 14] (2 layers)
- Hash cols per layer/token: 24 (2/3/4-grams x 8 heads)
- Row pairs per token: 48 (2 layers x 24)
- Weight row: 256 B, scale row: 8 B (pair 264 B, from manifest shapes)
- Useful table bytes/token: 12672 B
- Dense per-layer weights (wkv/q/k, resident candidates): {'1': 157521920, '14': 157521920} B
- Weight and scale share a shard per layer: True

## Per-corpus results

### english_prose (192 tokens)

- unique row pairs: 2400 (48.0/token)
- consecutive overlap mean: 0.0 rows
- 4 KiB: 105344 B/token (x8.313), 16 KiB: 411819 B/token (x32.498)
- row-LRU 64 MiB hit rate: 0.7396, 1 GiB: 0.7396
- coalesced (4 KiB gap): 25.0 ops/token, 3300 B/token

### japanese_prose (299 tokens)

- unique row pairs: 3680 (48.0/token)
- consecutive overlap mean: 0.0 rows
- 4 KiB: 103921 B/token (x8.201), 16 KiB: 405326 B/token (x31.986)
- row-LRU 64 MiB hit rate: 0.7436, 1 GiB: 0.7436
- coalesced (4 KiB gap): 24.58 ops/token, 3326 B/token

### source_code (297 tokens)

- unique row pairs: 4464 (48.0/token)
- consecutive overlap mean: 0.0 rows
- 4 KiB: 127197 B/token (x10.038), 16 KiB: 494940 B/token (x39.058)
- row-LRU 64 MiB hit rate: 0.6869, 1 GiB: 0.6869
- coalesced (4 KiB gap): 30.01 ops/token, 4056 B/token

### repetitive (240 tokens)

- unique row pairs: 384 (48.0/token)
- consecutive overlap mean: 0.0 rows
- 4 KiB: 13602 B/token (x1.073), 16 KiB: 52702 B/token (x4.159)
- row-LRU 64 MiB hit rate: 0.9667, 1 GiB: 0.9667
- coalesced (4 KiB gap): 3.2 ops/token, 422 B/token

### random_tokens (256 tokens)

- unique row pairs: 12288 (48.0/token)
- consecutive overlap mean: 0.0 rows
- 4 KiB: 404240 B/token (x31.9), 16 KiB: 1572224 B/token (x124.071)
- row-LRU 64 MiB hit rate: 0.0, 1 GiB: 0.0
- coalesced (4 KiB gap): 95.58 ops/token, 13428 B/token

### short_prompt (4 tokens)

- unique row pairs: 192 (48.0/token)
- consecutive overlap mean: 0.0 rows
- 4 KiB: 410624 B/token (x32.404), 16 KiB: 1601536 B/token (x126.384)
- row-LRU 64 MiB hit rate: 0.0, 1 GiB: 0.0
- coalesced (4 KiB gap): 96.0 ops/token, 12672 B/token

### long_prompt (2664 tokens)

- unique row pairs: 10351 (48.0/token)
- consecutive overlap mean: 0.0 rows
- 4 KiB: 32759 B/token (x2.585), 16 KiB: 127327 B/token (x10.048)
- row-LRU 64 MiB hit rate: 0.9191, 1 GiB: 0.9191
- coalesced (4 KiB gap): 7.74 ops/token, 1083 B/token

## Conclusion

Useful Engram table payload is 12672 B/token (2 layers x 24 rows x 264 B), versus 189.13 GiB of tables. Large tables are conditional sparse memory: row-stream with a modest row/page cache, keep the ~315 MB dense per-layer weights resident. Physical SSD latency remains unmeasured (simulation only).

## Transport

- Tokenizer bytes downloaded: 6368058
- Weight payload bytes downloaded: 0
