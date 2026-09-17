# DeepSeeker V4.1

Experimental, unofficial inference-acceleration runtime for **DeepSeek-V4.1-Flash** on Apple Silicon.

The primary target is a **64 GB M1 Max**. The project goal is simple:

> Maximize accepted output tokens per second without intentionally reducing model quality.

DeepSeeker V4.1 is not affiliated with or endorsed by DeepSeek.

## Status

Bootstrap / research phase. No performance claim in this repository should be treated as validated until it is accompanied by a reproducible benchmark.

## Design principles

1. **Quality is the invariant.** The default path keeps the official DeepSeek-V4.1-Flash weights and routing semantics. Approximate modes, if explored later, must be opt-in and reported separately.
2. **Optimize the whole machine.** CPU, GPU, Unified Memory, storage I/O, and optional accelerators are scheduled as one system.
3. **Move data before it is needed.** Predictive MoE residency, KV/index scheduling, Engram prefetching, and asynchronous I/O are treated as one memory-scheduling problem.
4. **Measure on the target machine.** Startup autotuning and runtime adaptation are preferred over generic fixed heuristics.
5. **Do not confuse prediction with routing.** The official model router remains authoritative. Predictors may prepare memory and execution work ahead of time, but must not silently change the model's routing decisions.

## Target platform

Initial optimization target:

- Apple M1 Max
- 64 GB Unified Memory
- macOS / Apple Silicon
- Metal / MLX experimentation where appropriate

Other Apple Silicon systems may work later, but portability is not the first objective.

Before optimization work, collect the target-machine baseline defined in [docs/MACHINE_PROFILE.md](docs/MACHINE_PROFILE.md).

## Planned optimization surface

DeepSeeker is intentionally broader than a MoE cache experiment. Candidate work includes:

- online future-expert prediction and resident expert groups
- layer/token lookahead and asynchronous prefetch
- KV-cache / sparse-index scheduling
- Engram working-set prefetch
- expert-aware grouped GEMM and multi-token weight reuse
- speculative execution with target-model verification
- FP4/FP8 decode and operator fusion
- Metal kernel tuning
- CPU/GPU/accelerator work partitioning
- Unified Memory placement and cache policy
- machine-specific autotuning and runtime adaptation

## Upstream DeepSeek reference code

The official DeepSeek-V4.1-Flash reference implementation is currently hosted in the model repository:

- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- reference local-inference code: `inference/`

This repository does **not** commit model weights. It also does not mix upstream files into DeepSeeker source code by default.

Fetch the current official reference code and metadata into the ignored `upstream/` working directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/fetch_upstream.py
```

The fetch script resolves and records the exact Hugging Face revision so experiments remain reproducible. It downloads only reference code / configuration / license metadata, not weight shards.

See [docs/UPSTREAM.md](docs/UPSTREAM.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Repository layout

```text
.
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MACHINE_PROFILE.md
│   ├── ROADMAP.md
│   └── UPSTREAM.md
├── scripts/
│   └── fetch_upstream.py
├── src/deepseeker/
│   └── __init__.py
├── upstream/
│   └── README.md
├── .gitignore
├── LICENSE
├── THIRD_PARTY_NOTICES.md
└── pyproject.toml
```

## First research milestone

Before implementing aggressive kernels, collect the real routing trace and measure:

- actual Top-k experts per layer and token
- next-layer and next-token expert-set predictability
- resident-set recall at fixed memory budgets
- cache churn and wasted prefetch bytes
- expert reuse across speculative / multi-token blocks
- KV/index/Engram traffic per accepted output token
- GPU idle time and Unified Memory bandwidth

The first Go/No-Go criterion is not a headline tokens/s number. It is whether the observed routing and memory-access structure provides enough reuse and predictability to hide storage latency and reduce DRAM traffic.

## Performance claims

The long-term research target is very high decode throughput on the target M1 Max, but **400 tok/s is a stretch target, not a current result**. All reported numbers should include hardware, OS version, model revision, context length, sampling configuration, batch/speculation settings, and whether the number is accepted output tokens/s.

## License

DeepSeeker V4.1 original code is licensed under the MIT License. See [LICENSE](LICENSE).

DeepSeek-V4.1-Flash upstream code and model weights are separately licensed by DeepSeek under MIT. When upstream code is copied or modified, its original copyright and license notice must be preserved. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
