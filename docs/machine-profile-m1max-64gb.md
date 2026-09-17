# Machine Profile — M1 Max 64GB (target machine, sanitized)

Date: 2026-09-18 (JST), updated same day with upstream fetch + GPU/CPU/SSD baselines
DeepSeeker commit: `f1c6fb0` at time of update (`docs: add sanitized M1 Max 64GB machine profile`)
Upstream `deepseek-ai/DeepSeek-V4.1-Flash` revision (`main`):
`dba1be0a40aa45a94ad051997016db3960a90277`
`python scripts/fetch_upstream.py`: DONE 2026-09-18, code-only (18 files),
weights excluded, manifest at `upstream/DEEPSEEKER_UPSTREAM.json` (git-ignored).

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
  Status 2026-09-18: still unavailable. Full Xcode cannot be installed from
  the CLI — `softwareupdate --list` offers no Xcode (App Store distribution
  only), `mas` is not installed, `/Applications/Xcode.app` is absent.
  Full-Xcode install requires a manual App Store step by the machine owner.
  GPU kernel baselines below therefore use MLX (prebuilt metallib, no Xcode
  needed) instead of hand-written Metal kernels — custom Metal tuning stays
  blocked on the App Store install.
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

## 5. GPU baseline via MLX 0.32.2 (no Xcode required)

Method: `.venv` with `pip install mlx`, `mx.default_device()=Device(gpu, 0)`,
`mx.eval()`-synchronized timing, warmup + 20 measured iterations,
median/min/max reported. Shapes use the real upstream expert dims
(`hidden_size=5120`, `moe_intermediate_size=2304`, fp16).

Upstream expert dims (from fetched `config.json`, `text_config`):
`hidden_size=5120`, `moe_intermediate_size=2304`, `n_routed_experts=384`,
`num_experts_per_tok=6`, `n_shared_experts=1`, `num_hidden_layers=40`,
base `dtype=bfloat16`, experts `fp4` (`quantization_config.expert_dtype`).

### 5a. Unified Memory bandwidth (GPU STREAM triad, fp16, 256 MiB)

`c = a + b*2.0`, traffic = 3×256 MiB, 20 iters: med 4.62 ms
(min 4.09, max 18.54) → ~162.2 GiB/s. Single-stream figure, not peak;
CPU+GPU simultaneous and multi-minute sustained runs still TODO (§4).
(An earlier `astype` no-op micro-test read 2354 GiB/s — discarded as a
fused-away no-op, kept here as a warning against lazy-eval artifacts.)

### 5b. MoE gate GEMM 5120×2304 fp16 (per expert)

| batch | med ms | min | max | GFLOPS(med) |
|---|---|---|---|---|
| 1 | 0.597 | 0.526 | 2.006 | 39.5 |
| 2 | 0.604 | 0.503 | 1.860 | 78.2 |
| 4 | 0.610 | 0.487 | 1.765 | 154.7 |
| 8 | 0.554 | 0.500 | 2.077 | 340.4 |
| 16 | 0.578 | 0.503 | 1.749 | 653.0 |

### 5c. MoE down GEMM 2304×5120 fp16

| batch | med ms | min | max | GFLOPS(med) |
|---|---|---|---|---|
| 6 (one token × 6 experts) | 0.799 | 0.683 | 2.398 | 177.2 |
| 12 | 0.733 | 0.672 | 1.849 | 386.2 |

### 5d. SwiGLU elementwise 6×2304 fp16 (`gate*silu(up)`)

med 0.487 ms (min 0.413, max 1.742). Unfused separate-ops reference;
a fused kernel is future work.

Reading: batch-1 latency is flat (~0.55–0.60 ms) across batches 1–16 —
launch-overhead dominated, expected for decode. Wide min/max spread shows
first-iteration/GC effects; medians are the comparison baseline.

## 6. CPU baselines (numpy 2.5.3, fp32, same expert shapes)

| op | batch | med ms | min | max | GFLOPS(med) |
|---|---|---|---|---|---|
| gate 5120×2304 | 1 | 1.38 | 1.23 | 1.64 | 17.1 |
| gate 5120×2304 | 2 | 1.77 | 1.68 | 2.16 | 26.7 |
| gate 5120×2304 | 4 | 1.82 | 1.70 | 2.09 | 51.9 |
| gate 5120×2304 | 8 | 0.98 | 0.94 | 1.22 | 192.7 |
| gate 5120×2304 | 16 | 1.04 | 0.90 | 1.90 | 361.3 |
| down 2304×5120 | 6 | 3.82 | 3.58 | 4.16 | 37.0 |

CPU SwiGLU elementwise 6×2304 fp32: med 0.034 ms. Small-batch inversions
(e.g. batch-8 med < batch-4) are BLAS threading/tiling effects, reported
as-is over 10 iters. CPU stays the control plane; these numbers exist to
prove GPU is required for the data plane, not to qualify CPU compute.

## 7. SSD sustained (2 GiB disposable file, F_NOCACHE, 4 KB blocks seq)

180 s sequential re-read loop, 30 s buckets (MiB/s):
5182 / 5247 / 5165 / 5090 / 5050 / ~5129 avg (901.6 GiB total).
No throttling signature across buckets in the tested path.
60 s random 1 MiB re-reads: 287552 ops, ~4793 MiB/s, ~209 us avg latency.

Caveat (honest): `F_NOCACHE` is advisory and the 2 GiB file fits in the
64 GiB page cache, so a cache-resident component cannot be excluded —
compare the earlier 1 GiB single-run read (2193 MiB/s) vs this run
(5129 MiB/s); methodology differs, both provisional. True cold-storage
verification needs `sudo purge` + retest, which requires interactive sudo
confirmation and was NOT run here. Pending: purge-controlled cold reads,
mmap/page-fault latency, async multi-outstanding, expert-object-sized
reads after upstream layout inspection.

## 8. CPU control plane / accelerators / DeepSeek baselines

Pending implementation + reference runs (§7–§9): routing-trace rate,
transition-table update rate, prediction latency, cache decision latency,
async I/O overhead, scheduler overhead/token, Core ML/accelerator comparison,
unoptimized DeepSeek prompt/decode tokens/s with full reporting
(hardware, OS, revisions, context, batch/speculation, warmup, median+range).

## 9. Minimum-profile status (against §15)

- [x] exact M1 Max CPU/GPU configuration
- [x] 64 GB Unified Memory confirmed
- [x] macOS + kernel version
- [ ] Xcode / Metal environment — CLT only; full Xcode needs manual App Store install (CLI impossible, verified 2026-09-18); MLX GPU path works without it
- [ ] effective stable resident-memory budget — pending load testing
- [~] sustained SSD read bandwidth — 180 s loop stable ~5.1 GB/s but cache-residency caveat; purge-controlled retest pending
- [~] effective Unified Memory bandwidth — GPU STREAM triad ~162 GiB/s single-stream; simultaneous CPU+GPU + sustained pending
- [x] batch-1 GEMV baseline — MLX fp16 gate 0.597 ms med (§5b), numpy fp32 1.38 ms (§6)
- [x] batched / grouped GEMM baseline — batches 2/4/8/16 done; grouped GEMM + FP4/FP8 decode paths pending
- [ ] unoptimized DeepSeek decode tokens/s — pending
- [ ] unoptimized DeepSeek prompt tokens/s — pending
- [x] upstream DeepSeek commit SHA — `dba1be0a40aa45a94ad051997016db3960a90277`, fetch manifest recorded

Next actions: owner installs full Xcode from App Store → `xcrun metal -v` →
`sudo purge`-controlled SSD retest (needs sudo confirmation) → sustained
CPU+GPU bandwidth → unoptimized DeepSeek reference runs (prompt/decode
tokens/s separated) → routing-trace instrumentation.
