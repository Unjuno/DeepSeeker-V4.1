# Read-mostly SSD policy (Issue #41, cross-cutting)

Hierarchy: verified checkpoint on SSD → manifest-addressed reads →
bounded unified-memory pools → GPU. Evicted data is discarded, never
written back; the SSD sees reads, not writes.

## Write classes

- Required: logs, user outputs, small metadata (incl. the checkpoint
  state JSON resume cursor — bytes, not weights).
- Optional (opt-in only): cache/predictor persistence, temp artifacts.
- Forbidden: expert/Engram writeback, per-run repacking, transient
  checkpoint copies, steady-state swap dependence.

## Enforcement

- `open_backing_file()`: checkpoint payload opens O_RDONLY only, with
  `..`/absolute-path rejection. All src readers use it.
- Static test: no write-flag `os.open` in `src/` outside the state-JSON
  writer; remaining opens must show `O_RDONLY`.
- `SwapMonitor`: vm_stat/swapusage before/after with OK/WARN/STOP
  verdicts. STOP (swap ≥2GB or runaway delta) forbids reporting
  benchmark numbers from that run.

## Usage

    python -c "from deepseeker.ssd_policy import SwapMonitor; ..."
