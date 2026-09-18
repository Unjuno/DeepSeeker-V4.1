# Target-machine probe (automated baseline)

One-command, privacy-safe probe that reproduces the machine/software state
and baseline microbenchmarks. Manual hand-collected numbers live in
`docs/machine-profile-m1max-64gb.md`; this probe is the repeatable source
of truth going forward. Where they disagree, the probe output wins for
metadata and the manual doc wins only for measurements the probe does not
cover yet (multi-minute sustained runs, numpy CPU GEMM tables).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # huggingface_hub + pytest + ruff
pip install mlx numpy     # optional; missing deps become structured
                          # `unavailable` results, never a crash
python scripts/probe_machine.py
```

Output (default `--out-dir profiles/m1-max-64gb`):

```text
profiles/
└── m1-max-64gb/
    ├── machine.json    # machine + software + memory (sanitized)
    ├── benchmark.json  # microbenchmarks with full conditions
    └── README.md       # generated human-readable summary
```

Only these three generated files are committed (see `.gitignore`
exceptions). Per-run copies under `profiles/runs/` stay local.

## Privacy model

- Allowlisted commands only; narrow regex parsers extract numeric/version
  fields. Raw `system_profiler` output is never stored.
- The probe self-scans its output for hostname, username, home path,
  `/Users/*`, UUIDs, and serial/UUID keywords, and refuses to write on a
  hit (exit 2).
- `tests/test_probe.py` covers parser behavior and the privacy scanner.

## Schema

- `deepseeker.machine/v1`: soc, cpu_total, cpu_performance_cores,
  cpu_efficiency_cores, cpu_p/e_l2_bytes, unified_memory_bytes,
  gpu_chipset, gpu_cores, metal_support, storage_model, storage_capacity,
  root_capacity_kb, root_available_kb, powermode.
- `deepseeker.software/v1`: macos_version/build, kernel_release/revision,
  arch, python_version, clang_version, developer_dir, xcode_app_installed,
  metal_compiler_available, mlx{available,device}, deepseeker_commit,
  upstream{available,repo_id,requested/resolved_revision,
  weights_downloaded}.
- `deepseeker.memory/v1`: physical_bytes, page_size_bytes, curated
  `vm_stat` pages, swap_total/used/free, memory_free_percent.
- `deepseeker.benchmark/v1`: probe_version, collected_at_utc,
  deepseeker_commit, upstream_resolved_revision, cpu_streaming, mlx_gpu,
  storage. Every numeric entry carries warmup_iters, measured_iters,
  median/min/max, units, op/shape, dtype, device/backend.
- Storage entries carry `method`, `cache_control`, and
  `cold_storage_proven` (false until a purge-controlled run proves it).

## Repeatability evidence (2026-09-18, 4 consecutive runs)

Machine/software fields identical across all runs
(M1 Max, 10 CPU, 32 GPU, 64 GiB, macOS 26.6.2/25G83, kernel 25.6.0,
upstream `dba1be0…`, swap 0, compression 0).

| metric (median) | run1 | run2 | run3 | canon | spread |
|---|---|---|---|---|---|
| CPU copy 64 MiB | 1.61ms | 1.59ms | 1.50ms | 1.60ms | ~7% |
| CPU copy 256 MiB | 6.76ms | 6.47ms | 6.91ms | 6.63ms | ~6% |
| GPU triad | 150GiB/s | 154GiB/s | 147GiB/s | 133GiB/s | ~14% |
| gate b1 | 0.382ms | 0.378ms | 0.418ms | 0.422ms | ~11% |
| gate b16 | 0.314ms | 0.349ms | 0.308ms | 0.374ms | ~19% |
| SSD write | 2819 | 3882 | 3887 | 6557 MiB/s | high |
| SSD read | 2877 | 3118 | 3534 | 3182 MiB/s | ~20% |

GEMM min values cluster ~0.30 ms with max spikes 1.3–2.4 ms
(GC/scheduling); the regime is launch-overhead dominated, so absolute
dispersion (~0.05–0.25 ms) matters more than relative. SSD write spread
confirms the `advisory-uncached` / `cold_storage_proven: false` labeling:
page cache dominates, methodology honestly reported, no cold claim made.

## Blocked / pending (recorded, not expanded)

- Full Xcode / `xcrun metal`: App Store manual install required; custom
  Metal kernels blocked. MLX GPU path used instead.
- Purge-controlled cold SSD: needs interactive `sudo purge`; not run.
- Stable resident-model budget, simultaneous CPU+GPU bandwidth,
  DeepSeek prompt/decode baselines: need weights/long runs; pending.
