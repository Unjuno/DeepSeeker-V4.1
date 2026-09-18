# Storage patterns benchmark

Revision: `dba1be0a40aa45a94ad051997016db3960a90277`
Generated: 2026-09-18T01:59:12.858431+00:00
Cache confidence: `advisory-uncached`
Bench file: 8589934592 bytes (reused existing file)

## Single experts (18,800,640 useful bytes each)

- early: span 604407552 B; exact QD1 p50 0.96 ms (4.07 GiB/s useful); coalesced QD1 p50 2.81 ms
- mid: span 609650432 B; exact QD1 p50 1.00 ms (5.15 GiB/s useful); coalesced QD1 p50 2.86 ms
- late: span 604407552 B; exact QD1 p50 0.96 ms (4.60 GiB/s useful); coalesced QD1 p50 2.67 ms
- min_span: span 604407552 B; exact QD1 p50 0.97 ms (5.29 GiB/s useful); coalesced QD1 p50 2.87 ms
- max_span: span 6963160832 B; exact QD1 p50 0.90 ms (3.61 GiB/s useful); coalesced QD1 p50 2.49 ms

## Expert groups (same-layer, QD scaling, useful GiB/s)

- n1 (18 MiB): QD1 4.25, QD2 6.17, QD4 8.71, QD8 8.62, QD16 8.78 GiB/s
- n2 (36 MiB): QD1 4.81, QD2 7.50, QD4 10.61, QD8 12.89, QD16 12.99 GiB/s
- n4 (72 MiB): QD1 4.46, QD2 4.09, QD4 8.87, QD8 16.39, QD16 16.40 GiB/s
- n8 (143 MiB): QD1 4.56, QD2 7.66, QD4 12.24, QD8 19.91, QD16 19.60 GiB/s
- n16 (287 MiB): QD1 4.66, QD2 7.71, QD4 13.54, QD8 20.59, QD16 21.16 GiB/s
- n32 (574 MiB): QD1 4.77, QD2 8.33, QD4 14.22, QD8 21.19, QD16 22.14 GiB/s

## Engram per-token (coalesced 4K-gap plan)

- english_prose: 96.0 ops/token, 12672 req B/token, QD1 elapsed/token 8.35 ms
- japanese_prose: 96.0 ops/token, 12672 req B/token, QD1 elapsed/token 8.22 ms
- source_code: 96.0 ops/token, 12672 req B/token, QD1 elapsed/token 8.22 ms
- repetitive: 96.0 ops/token, 12672 req B/token, QD1 elapsed/token 8.33 ms
- random_tokens: 96.0 ops/token, 12672 req B/token, QD1 elapsed/token 8.55 ms

## Sustained (8-expert same-layer group)

- qd1: avg 5665.7 MiB/s over 60.0s; buckets [5702.4 5694.3 5756.5 5738.4 5583.0 5519.5] MiB/s
- qd4: avg 15284.3 MiB/s over 60.0s; buckets [15145.4 15099.0 15108.8 15265.8 15432.7 15663.5] MiB/s

## Cache-confidence conclusion

Default results are advisory-uncached (F_NOCACHE, no purge). Nothing here is cold-proven unless --assert-cold records a manual purge procedure.
