# Resident-memory plan

Revision: `dba1be0a40aa45a94ad051997016db3960a90277`
Generated: 2026-09-18T02:26:51.838713+00:00
Operating budget: 44.00 GiB (from safe-capacity probe)
Lookup tables resident: 0 B (conditional row-stream + cache (Issue #3); never fully resident)

## Scenarios (expert slots / % of 15,360)

- S1@4K: 1889 slots (12.3%, 47.23/layer even)
- S1@32K: 1679 slots (10.93%, 41.98/layer even)
- S1@128K: 957 slots (6.23%, 23.93/layer even)
- S1@512K: 0 slots (0.0%, 0.0/layer even)
- S1@1024K: 0 slots (0.0%, 0.0/layer even)
- S2@4K: 1849 slots (12.04%, 46.23/layer even)
- S2@32K: 1639 slots (10.67%, 40.98/layer even)
- S2@128K: 917 slots (5.97%, 22.93/layer even)
- S2@512K: 0 slots (0.0%, 0.0/layer even)
- S2@1024K: 0 slots (0.0%, 0.0/layer even)
- S3@4K: 1838 slots (11.97%, 45.95/layer even)
- S3@32K: 1627 slots (10.59%, 40.67/layer even)
- S4@4K: 1889 slots (12.3%, 47.23/layer even)
- S4@32K: 1679 slots (10.93%, 41.98/layer even)
- S4@128K: 957 slots (6.23%, 23.93/layer even)
- S4@512K: 0 slots (0.0%, 0.0/layer even)
- S4@1024K: 0 slots (0.0%, 0.0/layer even)

## Storage modes (expert service p50)

- optimistic: 0.8981249993667006 ms (measured-cache)
- advisory: 0.9561250044498593 ms (measured-advisory)
- unknown-cold: 2.87 ms (assumed)

## Engram cache benefit (row-LRU hit rate, 64 MiB)

- english_prose: 0.7396
- japanese_prose: 0.7436
- source_code: 0.6869
- repetitive: 0.9667
- random_tokens: 0.0
- short_prompt: 0.0
- long_prompt: 0.9191

Provenance travels per component in resident-plan.json: measured / manifest / arch-estimate / assumed.
