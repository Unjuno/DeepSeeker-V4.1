# Analytic decode flow-map (throughput maximization)

Closed form, zero experiments: `tok/s = B·(1+a) / (bytes_fwd/BW + L)`.
Reproduces the MLX bench (109.9ms predicted vs 109.8ms measured).

## The 11 tok/s number was not the maximum

It assumed batch 1, dense fixed, no sparsity, no MTP. The flow map:

| combo (B, a, s) | tok/s | GiB/forward |
|---|---|---|
| (1, 0, 0) baseline | 10.7 | 11.7 |
| (1, 0, 0.7) sparsity only | 14.1 | 8.8 |
| (1, 2, 0) MTP only | 32.2 | 11.7 |
| (8, 2, 0.5) | 132.3 | 23.4 |
| (32, 2, 0.5) | 207.5 | 60.7 |
| (128, 2, 0.7) | 652.5 | 77.4 |

Solver: 400 tok/s needs B=125 (a=2, s=0.5) or s=0.78 (@B=32, a=2).
2000 tok/s needs B=446 (a=2, s=0.7) — where KV×B hits memory walls first.

## Why sparsity alone stalls at batch 1

dense_fixed (8.1GB: attention+shared+head) is paid regardless. Skipping
70% of expert neurons only removes ~3GB of 11.7GB. The floor is dense,
not MoE — the unexpected half of the bottleneck.

## Cheat inventory (what each trick actually buys)

- **MTP (built-in, 3 heads in weights)**: up to 3x tokens per forward at
  ~zero extra traffic. Largest single-token lever. OPEN: verify cost of
  the mtp-expert pass per draft (7.2GB in manifest — if run per draft,
  the 3x shrinks; measure next).
- **Neuron sparsity** (row-select gate/up, column-select down): needs a
  neuron predictor; our router-prediction machinery extends naturally.
  Pays mostly at batch ≥ 8 where MoE dominates the union.
- **Batch**: sublinear (occupancy union saturates to all 384 experts by
  B≈128: 333/layer) but still the strongest lever to 400+.
- **Prefix KV sharing**: free tokens for shared prompt prefixes (not in
  the equation yet — next term to add with serving data).
- **Residency** (P4 pool): converts SSD traffic to zero for hot experts;
  modeled as byte multiplier, default 1.0 (streaming assumption).

## OR framing (standing problem)

maximize tok/s(B, s, a, R) s.t. resident+KV×B ≤ 64GB, forward latency ≤
SLA, router authoritative, MTP verify cost measured. Batch is the only
lever reaching 400+; sparsity+MTP decide how small B can stay (latency).
2000 needs B≈446 → infeasible on latency+KV grounds: cap the ambition
at 400 unless prefix-sharing changes the equation.
