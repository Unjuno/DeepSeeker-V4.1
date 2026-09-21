# Engram row/page cache runtime (Issue #26, roadmap P5)

Bounded LRU page cache (4 KiB / 16 KiB) fronting exact (256B weight +
8B scale) row assembly from the real 189 GiB tables. Geometry from
Issue #3 refs; hashes and math untouched; tables never fully resident.

## Usage

    python scripts/run_engram_cache.py --budgets 16,64,256 --pages 4096,16384 --ops 2000

Every sweep validates the first batch byte-equal against direct preads
(exactness gate, FAIL otherwise) and reports hit rate, bytes read, and
read amplification (bytes_read / useful 264B rows).

## Measured (2000-row stream x 2 passes, layer 1)

| budget | page | hit_rate | read | ampl |
|---|---|---|---|---|
| 16MB | 4K | 0.502 | 15.9MB | 15.8x |
| 16MB | 16K | 0.005 | 125MB | 124x |
| 64MB | 4K | 0.502 | 15.9MB | 15.8x |
| 64MB | 16K | 0.506 | 62MB | 61.8x |

Second pass reuses everything when the 16MB working set fits: 4K pages
win on amplification (15.8x vs 61.8x). The 16MB×16K cell is the
capacity cliff (65MB working set churns). All rows byte-exact.
Takeaway for P5: 4K granularity + recurrent working sets fit in tens
of MB; uniform streams get ~zero reuse at any budget (matches #3).

## Design notes

- Pages, not rows, are cached: a row usually spans two pages, so the
  minimum useful budget is a few pages, not a few rows.
- Capacity enforced at call boundaries; acquisitions are exact-range
  preads (no span-coalescing across table gaps).
- Access generator: seeded hotspot/uniform streams stand in for #20
  traces; swap the stream source when online traces land, nothing else.
