# Architecture

DeepSeeker V4.1 is a machine-specific inference acceleration project for DeepSeek-V4.1-Flash.

## Objective

Maximize **accepted output tokens/s** on the target machine without intentionally reducing target-model quality.

The first target is an Apple M1 Max with 64 GB Unified Memory.

## Quality invariant

The default execution path should preserve the official target model's weights and authoritative routing decisions.

Allowed optimization classes include:

- reordering independent work
- asynchronous prefetch
- cache/residency changes
- mathematically equivalent kernel fusion
- grouped execution / batching
- speculative execution when verified by the target model
- runtime autotuning

Approximate routing, extra quantization beyond the chosen official checkpoint representation, expert dropping, or other quality-changing methods must not be silently enabled in the default benchmark path.

## System split

### Control plane

Primarily CPU, with optional accelerator use when measured beneficial:

- routing trace collection
- online transition / motif learning
- future expert-set prediction
- cache admission / eviction policy
- KV/index/Engram deadline scheduling
- asynchronous I/O submission
- machine-specific autotuning
- runtime telemetry

### Data plane

Primarily Metal GPU / MLX during prototyping:

- attention
- MoE expert computation
- grouped GEMM
- FP4/FP8 decode/dequantization
- KV/index operations
- verification work for speculative execution
- fused kernels

## Memory hierarchy

Conceptual hierarchy:

```text
cold storage
    ↓
Unified Memory resident pools
    ├── common / always-needed weights
    ├── routed expert resident set
    ├── KV + sparse index state
    ├── Engram working set
    └── prefetch / staging buffers
            ↓
        GPU execution
```

The scheduler should optimize the hierarchy globally rather than independently optimizing MoE, KV, and Engram.

## MoE predictor boundary

DeepSeeker's predictor is not the model router.

```text
official router -> authoritative expert choice -> computation
       ↑
future-route predictor -> memory preparation only
```

Prediction misses may cause stalls or wasted reads, but should not change which experts the model actually executes in quality-preserving mode.

## Online adaptation lifecycle

1. **Observe** real routing and memory behavior.
2. **Shadow predict** without changing memory policy.
3. Measure recall, wasted prefetch bytes, cache churn, and stall cost.
4. Enable predictive residency only after thresholds are met.
5. Disable or reduce aggressiveness under distribution shift.
6. Continue learning online.

## Scheduler priorities

A useful initial ordering is:

1. data with a known near deadline (actual routed expert, required KV/index)
2. high-confidence future expert groups
3. deterministic Engram / known future state
4. lower-confidence speculative prefetch when bandwidth is available

The final policy must be selected by target-machine measurement, not assumed.

## Performance accounting

Every benchmark should report at least:

- accepted output tokens/s
- prompt processing tokens/s separately
- model revision
- target hardware and OS
- context length
- batch/speculation configuration
- memory footprint
- Unified Memory bandwidth if measurable
- storage read bandwidth
- expert resident hit / recall
- wasted prefetch bytes
- GPU idle / stall time
- output-quality equivalence checks
