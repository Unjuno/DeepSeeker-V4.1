# Machine Profile — M1 Max 64GB (target machine, sanitized)

Date: 2026-09-18 (JST)
DeepSeeker commit: `a9b722a8737540d413c6bf97a247152af89a06d9`
Upstream `deepseek-ai/DeepSeek-V4.1-Flash` revision (resolved via HfApi, `main`):
`dba1be0a40aa45a94ad051997016db3960a90277`
Full `python scripts/fetch_upstream.py` (code-only download + manifest): pending.

> Privacy: this file contains only performance-relevant data.
> No serial numbers, hardware UUIDs, volume UUIDs, hostnames, usernames,
> home-directory paths, account or network data are recorded here,
> per `docs/MACHINE_PROFILE.md` §13. Raw `system_profiler` outputs were
> kept in `/tmp` and not committed.

## 1. Static hardware (sanitized)

- Model: MacBook Pro, Model Identifier MacBookPro18,2
- SoC: Apple M1 Max
- CPU: 10 cores (8 Performance + 2 Efficiency)
  - `hw.ncpu=10`, `hw.physicalcpu=10`, `hw.logicalcpu=10`
  - P-cluster L2: 12582912 bytes (12 MiB, 4 CPUs per L2)
  - E-cluster L2: 4194304 bytes (4 MiB, 2 CPUs per L2)
- GPU: Apple M1 Max, 32 cores, Metal 4 (`Metal Support: Metal 4`)
- Display: Built-in Liquid Retina XDR, 3456x2234, Internal connection.
  Recorded only because it can affect GPU resource availability.
- Unified Memory: `hw.memsize=68719476736` = 64 GiB
- Storage: APPLE SSD AP4096R, 4 TB (APFS container ~3.996 TB), Internal Yes,
  Protocol Apple Fabric, S.M.A.R.T. Verified, TRIM Yes.
  Free ~2.62 TB at measurement time (2026-09-18).
- Power at measurement: `pmset` system-wide `powermode 2` (AC).
  No Low Power Mode indication observed. `displaysleep 10`, `disksleep 10`.
  (Fanless-relevant throttling / sustained-power behavior: not yet measured.)

Collection commands used:

```bash
sysctl hw.ncpu hw.physicalcpu hw.logicalcpu hw.memsize hw.cpufamily hw.cpusubfamily
sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
sysctl hw.perflevel0.l2cachesize hw.perflevel1.l2cachesize
system_profiler SPHardwareDataType   # sanitized, raw not committed
system_profiler SPDisplaysDataType   # sanitized, raw not committed
system_profiler SPStorageDataType SPNVMeDataType  # sanitized, raw not committed
pmset -g
```

## 2. Software environment

- macOS: 26.6.2, Build 25G83 (`sw_vers`)
- Kernel: Darwin 25.6.0, `kern.osrevision=199506`,
  `root:xnu-12377.161.14~5/RELEASE_ARM64_T6000`, `arm64`
- Xcode: full Xcode.app NOT installed (`/Applications/Xcode.app` missing).
  Active developer dir is Command Line Tools only:
  `/Library/Developer/CommandLineTools`, SDK `MacOSX.sdk`.
  `xcodebuild -version` fails (CLT instance). Full-Xcode install is pending
  (user-approved) and required for the Metal toolchain.
- Metal toolchain: `xcrun metal -v` → `unable to find utility "metal"`.
  Status: unavailable until full Xcode is installed. GPU kernel baselines (§6)
  are therefore pending.
- Compiler: Apple clang 21.0.0 (`clang-2100.3.34.2`),
  Target `arm64-apple-darwin25.6.0`
- Python: 3.14.5 (both system `python3` and Homebrew `python3` report 3.14.5)
- MLX: not installed in Homebrew Python. PyTorch/MPS: not installed.
- DeepSeeker: commit `a9b722a` (`docs: link target-machine profiling checklist`)

## 3. Memory baseline

`vm_stat` snapshot at measurement (page size 16384 bytes):

```text
Pages free: 58754
Pages active: 1938596
Pages inactive: 1562917
Pages speculative: 376427
Pages wired down: 181771
Pages purgeable: 55470
Pages stored in compressor: 0
Pages occupied by compressor: 0
Compressions: 0, Decompressions: 0
Pageins: 6068186
```

- `sysctl vm.swapusage`: `total = 0.00M used = 0.00M free = 0.00M (encrypted)`
  → no swap in use at measurement; sustained-pressure behavior under
  inference load is still unmeasured (TODO).
- Stable resident-model budget, safety margin, expert-pool size, KV/Engram
  footprints: pending real-model runs (TODO, `MACHINE_PROFILE.md` §3).

### Quick CPU memory-bandwidth sanity check (numpy, NOT peak)

Method: `numpy.copyto` on 64 MiB and 256 MiB buffers, 10 timed iterations
after 3 warmups, single process, buffer counts as read+write:

- 64 MiB buffer: ~1.6 ms/iter → ~78.9 GiB/s
- 256 MiB buffer: ~6.1 ms/iter → ~81.7 GiB/s

This is a CPU-only streaming baseline, not the M1 Max unified-memory peak
and not a CPU+GPU simultaneous measurement. Simultaneous CPU+GPU bandwidth,
FP4/FP8-decode bandwidth, and multi-minute sustained runs: TODO (§4).

## 4. Internal SSD — first measurements (disposable file, F_NOCACHE)

Method: 1 GiB disposable file under `/tmp`, written and read with
`fcntl(F_NOCACHE)` (macOS 48) + `fsync`, then deleted. Single run only —
not yet a sustained multi-minute result. Representative sizes 4 KiB / 64 KiB /
1 MiB covered in a cached pilot; 4 MiB / 16–32 MiB expert-sized reads pending
actual expert layout inspection.

Uncached results (single run, 2026-09-18):

- Sequential write 1 GiB: ~0.34 s → ~2977 MiB/s
- Sequential read 1 GiB: ~0.47 s → ~2193 MiB/s
- Random 1 MiB x50: ~235 us avg → ~4252 MiB/s
  (higher than sequential here; file may still be partially cache-resident
  despite F_NOCACHE — treat as provisional, needs `purge`-controlled retest)

Cached pilot (page-cache resident, NOT SSD numbers, for contrast):

- `dd` 512 MiB sequential read immediately after write: ~15875 MB/s
- Python random reads on cached file: 4 KiB ~2030 MiB/s, 64 KiB ~10000 MiB/s,
  1 MiB ~13769 MiB/s — clearly cache hits, recorded only to show why
  uncached methodology matters.

TODO (§5): `sudo purge`-controlled cold reads, mmap/page-fault latency,
async multi-outstanding reads, ≥several-minute sustained reads,
throttling check, expert-object-sized reads after upstream layout inspection.
`sudo` steps need explicit user confirmation per session policy.

## 5. GPU / Metal baseline

- Static: 32-core M1 Max GPU, Metal 4 — confirmed via `SPDisplaysDataType`.
- Kernel numbers (FP16/BF16 GEMM, FP8/FP4 decode, GEMV batch-1,
  batches 2/4/8/16, grouped GEMM, fused SwiGLU, launch overhead,
  command-buffer overlap, sustained runs): pending full Xcode + Metal
  toolchain and DeepSeek expert-shape verification (§6).

## 6. CPU control plane / accelerators / DeepSeek baselines

Pending implementation + reference runs (§7–§9): routing-trace rate,
transition-table update rate, prediction latency, cache decision latency,
async I/O overhead, scheduler overhead/token, Core ML/accelerator comparison,
unoptimized DeepSeek prompt/decode tokens/s with full reporting
(hardware, OS, revisions, context, batch/speculation, warmup, median+range).

## 7. Minimum-profile status (against §15)

- [x] exact M1 Max CPU/GPU configuration
- [x] 64 GB Unified Memory confirmed
- [x] macOS + kernel version
- [ ] Xcode / Metal environment — CLT only; full Xcode + `xcrun metal` pending
- [ ] effective stable resident-memory budget — pending load testing
- [ ] sustained SSD read bandwidth — first single-run numbers only
- [ ] effective Unified Memory bandwidth — numpy CPU sanity only
- [ ] batch-1 GEMV baseline — pending
- [ ] batched / grouped GEMM baseline — pending
- [ ] unoptimized DeepSeek decode tokens/s — pending
- [ ] unoptimized DeepSeek prompt tokens/s — pending
- [x] upstream DeepSeek commit SHA — resolved `dba1be0...90277`; full fetch manifest pending

Next actions: install full Xcode → `xcrun metal -v` → run
`scripts/fetch_upstream.py` (records immutable manifest) → sustained
memory/SSD/GPU baselines with medians, ranges, buffer sizes, iteration
counts, thermal notes.
