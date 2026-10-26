# DeepSeeker V4.1

Experimental, unofficial inference-acceleration runtime for **DeepSeek-V4.1-Flash** on Apple Silicon.

The primary target is a **64 GB M1 Max**. The project goal is simple:

> Maximize accepted output tokens per second without intentionally reducing model quality.

DeepSeeker V4.1 is not affiliated with or endorsed by DeepSeek.

## Status: research verdict (2026-10)

This was a **time-boxed feasibility probe** for aggressive local inference of
DeepSeek-V4.1-Flash on one 64 GB M1 Max. It is concluded. The honest summary:

**What works (delivered, tested):**
- A corrected MLX/CPU streaming inference engine for the official FP4 weights,
  including a fixed `kv_norm` bug that had silently degraded attention quality
  (cos 0.83 → 0.999996 vs the torch reference); the engine now produces
  coherent end-to-end text.
- A reproducible quality harness (`scripts/quality_q3.py`) that replays
  generation teacher-forced against an fp32 torch reference and reports
  top-1/top-5/rank statistics; the repo carries 264 hermetic tests (green)
  plus CPU-oracle debugging tooling.
- A validated 3-bit pilot for routed experts L20-39 at
  **92.7% top-1 match** with the FP4 reference (`n=1,471` token replay),
  saving **-11.6% bytes per token** (150.8 vs 170.6 GiB/token).
- Full per-experiment writeups in `docs/QUANT_RESEARCH.md` and
  `docs/ROADMAP.md`.

**What does not work (the headline finding):**
- **The 400 tok/s goal is physically out of reach on this hardware** with
  any lossless or near-lossless configuration. Decode of routed experts alone
  streams ~3.7 GiB per token; the SSD caps sustained reads at ~6 GiB/s.
  The measured B=1 steady state (1.63-1.67 tok/s) is already at the
  bandwidth ceiling, not short of it.
- Speculative multi-token decode (MTP) was re-measured post-fix:
  acceptance 27-37%, but throughput 0.86-0.93x (draft is cheap, verification
  serializes). Not a win.
- Hot-expert pinning improved latency (+23%) but cannot beat the
  bandwidth bound; cache-policy sweeps (LRU/LFU, pin sizes) all plateau at
  the same physical limit.

**Bottom line:** on a 64 GB machine streaming a 231 GB payload, tokens/s is
governed by `SSD bandwidth / bytes-per-token`. Any further gain requires
either much lower bits-per-weight (with real quality loss), a bigger unified
memory machine, or a different storage tier. This repository stands as a
negative-but-thorough result with a working baseline engine.


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
├── docs/            # ARCHITECTURE, MACHINE_PROFILE, ROADMAP (session logs), QUANT_RESEARCH, UPSTREAM
├── scripts/         # fetch_upstream, probe_machine, bench_*, quality_q3, requant_experts_3bit, ...
├── src/deepseeker/  # streaming MLX inference engine + CPU reference + quant/cache tooling
├── tests/           # 264 hermetic tests (run under DEEPSEEKER_VARIANT=1017 for the real weights)
├── upstream/        # ignored; official DeepSeek reference code fetched by scripts/fetch_upstream.py
├── models/          # ignored; local checkpoint + generated quantized payloads (not committed)
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

All numbers quoted in `docs/` include hardware, OS version, model revision,
context length, sampling configuration, and batch/speculation settings. The
historical 400 tok/s stretch target is **concluded as not viable** on this
hardware class; see the Status section above and `docs/ROADMAP.md` for the
measured physics.

## License

DeepSeeker V4.1 original code is licensed under the MIT License. See [LICENSE](LICENSE).

DeepSeek-V4.1-Flash upstream code and model weights are separately licensed by DeepSeek under MIT. When upstream code is copied or modified, its original copyright and license notice must be preserved. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
