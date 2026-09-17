# Target Machine Profiling Checklist

DeepSeeker V4.1 is optimized against a specific target machine. Before serious optimization work begins, capture both **static machine information** and **measured performance limits**.

The first target is:

- Apple M1 Max
- 64 GB Unified Memory
- macOS
- DeepSeek-V4.1-Flash

This document defines the minimum machine profile that should be recorded before performance work.

> Do **not** commit raw machine identifiers, serial numbers, hardware UUIDs, usernames, hostnames, account names, or other personally identifying information to the public repository.

---

## 1. Static hardware information

Record:

- [ ] Apple SoC model
- [ ] CPU core count
- [ ] GPU core count
- [ ] Neural Engine availability
- [ ] Unified Memory capacity
- [ ] storage device model
- [ ] storage capacity
- [ ] display configuration only if it materially affects available GPU resources
- [ ] power mode / Low Power Mode state when benchmarking

Suggested local collection commands:

```bash
system_profiler SPHardwareDataType > hardware.raw.txt
system_profiler SPDisplaysDataType > gpu.raw.txt
system_profiler SPNVMeDataType > nvme.raw.txt
```

These files may contain identifying information. Sanitize them before committing anything derived from them.

---

## 2. Software environment

Record:

- [ ] macOS version
- [ ] Darwin kernel version
- [ ] Xcode version
- [ ] Metal toolchain version
- [ ] Python version
- [ ] MLX version, if used
- [ ] PyTorch / MPS version, if used
- [ ] compiler version
- [ ] DeepSeeker commit
- [ ] upstream DeepSeek-V4.1-Flash revision

Suggested commands:

```bash
sw_vers
uname -a
xcodebuild -version 2>/dev/null || true
xcrun metal -v 2>/dev/null || true
python3 --version
clang --version
```

For DeepSeek upstream, record the immutable revision written by:

```bash
python scripts/fetch_upstream.py
```

---

## 3. Memory limits and pressure

The nominal 64 GB Unified Memory capacity is not the usable resident-model budget.

Measure:

- [ ] total physical memory
- [ ] baseline memory usage after boot / normal login
- [ ] memory pressure before inference
- [ ] compressed memory
- [ ] swap usage
- [ ] maximum stable resident allocation before compression or swap becomes material
- [ ] safety margin required for macOS and runtime allocations
- [ ] memory footprint of the common non-expert model state
- [ ] practical expert-resident pool size
- [ ] KV / index memory footprint at representative context lengths
- [ ] Engram working-set memory footprint

Suggested commands:

```bash
sysctl hw.memsize
vm_stat
vm_stat 1
sysctl vm.swapusage
memory_pressure
```

Do not use "free bytes" alone as the capacity metric. macOS aggressively uses cache and compressed memory; DeepSeeker should identify the point where sustained inference begins to cause harmful memory pressure or swap activity.

---

## 4. Unified Memory bandwidth

Measure actual bandwidth on this exact machine rather than relying only on the advertised peak.

Required tests:

- [ ] sequential read bandwidth
- [ ] sequential write bandwidth
- [ ] read + write mixed bandwidth
- [ ] CPU-only bandwidth
- [ ] GPU-only bandwidth
- [ ] CPU + GPU simultaneous bandwidth
- [ ] bandwidth with representative FP4/FP8 decode kernels
- [ ] sustained bandwidth under multi-minute load

Report:

- median
- range / dispersion
- buffer size
- number of iterations
- CPU worker count
- GPU workload
- thermal state if measurable

The effective memory bandwidth under simultaneous CPU/GPU activity is more important than the theoretical peak.

---

## 5. Internal SSD performance

The cold-expert path depends on sustained storage behavior, not just short synthetic peaks.

Measure:

- [ ] large sequential read bandwidth
- [ ] random read bandwidth
- [ ] 4 KiB random read latency
- [ ] larger expert-sized random reads
- [ ] reads near one routed-expert object size
- [ ] mmap/page-fault latency
- [ ] multiple outstanding asynchronous reads
- [ ] sustained read performance for at least several minutes
- [ ] storage throttling / thermal degradation if present

Use a disposable benchmark file. Do not benchmark against model-weight files until the storage test procedure is understood.

Representative read sizes should include values around:

- 4 KiB
- 64 KiB
- 1 MiB
- 4 MiB
- ~16–32 MiB

The final values should be chosen after the actual DeepSeek expert storage layout is inspected.

---

## 6. GPU / Metal compute baseline

Before changing DeepSeek, establish the machine's kernel baseline.

Measure:

- [ ] FP16 GEMM
- [ ] BF16 GEMM where supported
- [ ] FP8-relevant decode/conversion path if implemented
- [ ] FP4 decode/dequantization path
- [ ] GEMV / batch-1 matrix-vector throughput
- [ ] GEMM with token batches 2, 4, 8, 16
- [ ] grouped GEMM
- [ ] fused SwiGLU expert path
- [ ] kernel launch overhead
- [ ] command-buffer overlap
- [ ] sustained GPU performance under long runs

Use DeepSeek-relevant shapes, especially the real MoE expert dimensions after verifying them against the pinned upstream revision.

For every benchmark record:

- matrix shapes
- dtype / quantization
- batch size
- number of warmup iterations
- number of measured iterations
- median latency
- achieved throughput
- memory bandwidth if measurable

---

## 7. CPU control-plane baseline

DeepSeeker intends to keep the CPU primarily on the control plane rather than using it as the primary model compute engine.

Measure:

- [ ] routing-trace processing rate
- [ ] online transition-table update rate
- [ ] future-expert prediction latency
- [ ] cache admission / eviction decision latency
- [ ] async I/O submission overhead
- [ ] scheduler overhead per layer
- [ ] scheduler overhead per accepted output token
- [ ] CPU utilization while GPU inference is active

The scheduler must remain comfortably below the target token budget.

---

## 8. Optional accelerator baseline

Do not use the Neural Engine merely because it exists.

Measure separately:

- [ ] Core ML model invocation latency
- [ ] small predictor latency on CPU
- [ ] same predictor latency using available accelerator placement
- [ ] synchronization cost with the main Metal execution path
- [ ] effect on end-to-end accepted tokens/s

Enable accelerator offload only if it improves end-to-end performance or materially reduces CPU contention.

---

## 9. DeepSeek baseline before optimization

Run the closest available reference implementation before changing scheduling or kernels.

Record:

- [ ] upstream DeepSeek revision
- [ ] checkpoint / weight representation
- [ ] context length
- [ ] prompt length
- [ ] generated token count
- [ ] sampling parameters
- [ ] batch size
- [ ] speculative decoding state
- [ ] prompt-processing tokens/s
- [ ] accepted output tokens/s
- [ ] time-to-first-token
- [ ] peak resident memory
- [ ] swap activity
- [ ] GPU utilization
- [ ] CPU utilization
- [ ] storage read rate
- [ ] Unified Memory bandwidth if measurable

Prompt-processing throughput and decode throughput must be reported separately.

---

## 10. DeepSeeker-specific MoE measurements

Once routing instrumentation exists, record:

- [ ] authoritative Top-k expert IDs per layer / token
- [ ] router scores if available
- [ ] next-layer expert-set overlap
- [ ] next-token expert-set overlap
- [ ] multi-layer routing motifs
- [ ] expert reuse across speculative token blocks
- [ ] unique expert count per multi-token block
- [ ] resident-set Recall@K
- [ ] cache hit rate
- [ ] cache churn
- [ ] false-positive prefetch bytes
- [ ] false-negative / late expert events
- [ ] expert-induced GPU stall time

Prediction quality should be evaluated primarily by whether all required experts are resident before their deadline, not by simple classification accuracy.

---

## 11. KV / sparse-index / Engram measurements

Record independently:

- [ ] KV bytes read per accepted output token
- [ ] KV bytes written per accepted output token
- [ ] sparse-index traffic
- [ ] Engram bytes fetched per token
- [ ] cache reuse
- [ ] scheduling deadline misses
- [ ] stall time attributed to each subsystem

The long-term goal is a unified scheduler, so measurements must make it possible to distinguish:

- MoE traffic
- KV/index traffic
- Engram traffic
- other model traffic

---

## 12. Thermal and sustained-performance behavior

Short runs are not enough for machine-specific optimization.

Measure:

- [ ] cold-start performance
- [ ] performance after 1 minute
- [ ] performance after 5 minutes
- [ ] performance after 15+ minutes
- [ ] CPU/GPU power if accessible
- [ ] thermal throttling indicators
- [ ] fan behavior if relevant
- [ ] throughput degradation over time

Where possible:

```bash
sudo powermetrics --samplers cpu_power,gpu_power -i 1000
```

Availability and sampler names may vary with macOS version.

---

## 13. Public profile: privacy rules

Never commit raw values for:

- [ ] serial number
- [ ] hardware UUID
- [ ] device UUID
- [ ] hostname
- [ ] username
- [ ] home-directory path
- [ ] Apple ID / account data
- [ ] network addresses
- [ ] unrelated attached-device identifiers

The public profile should contain only performance-relevant machine data.

---

## 14. Recommended repository output

The eventual sanitized profile should look roughly like:

```text
profiles/
└── m1-max-64gb/
    ├── machine.json
    ├── software.json
    ├── memory.json
    ├── storage.json
    ├── metal.json
    ├── baseline-deepseek.json
    └── README.md
```

Raw local outputs should remain outside Git or under ignored paths.

---

## 15. Minimum profile before optimization begins

The minimum acceptable starting profile is:

- [ ] exact M1 Max CPU/GPU configuration
- [ ] 64 GB Unified Memory confirmed
- [ ] macOS + kernel version
- [ ] Xcode / Metal environment
- [ ] effective stable resident-memory budget
- [ ] sustained SSD read bandwidth
- [ ] effective Unified Memory bandwidth
- [ ] batch-1 GEMV baseline
- [ ] batched / grouped GEMM baseline
- [ ] unoptimized DeepSeek decode tokens/s
- [ ] unoptimized DeepSeek prompt tokens/s
- [ ] upstream DeepSeek commit SHA

Optimization work should be compared against this baseline, not against advertised hardware specifications.

---

## Benchmark reporting rule

Any performance result intended for the public repository should include:

- hardware
- clocks / power mode if relevant
- OS
- software revisions
- context length
- batch / speculative settings
- measured token count
- warmup method
- repeated runs
- median and range
- accepted output tokens/s

A single best run is not sufficient evidence.
