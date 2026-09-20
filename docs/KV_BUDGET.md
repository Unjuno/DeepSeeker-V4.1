# KV/index memory model + unified budget (Issue #15, roadmap P5 input)

Element-exact cache geometry read off `inference/model.py`, replacing the
HIGH-uncertainty arch-estimates in the Issue #5 resident plan.

## Geometry (batch B, context S)

- `window`: all 40 layers × [B, 128, 512] fp8 — context-independent ring.
- `compress`: layers [2, 8, 14, 20] only × [B, S//ratio, 512] fp4,
  ratios {2:2, 8:2, 14:2, 20:1} (comment at model.py:759 confirms fp4).
- `index`: same 4 owning layers × [B, S//ratio, 128] fp4.

Excluded and noted: quant scales (+~3–6%), MTP-layer state, fp32
compressor working state (not cache). Single request assumed (B=1).

## Headline (B=1)

| ctx | window | compress | index | total |
|-----|--------|----------|-------|-------|
| 4k | 2.5MB | 2.5MB | 0.6MB | 5.6MB |
| 32k | 2.5MB | 20MB | 5MB | 27.5MB |
| 131k | 2.5MB | 80MB | 20MB | 102.5MB |

The resident plan estimated index-state alone at ~537MB @4k
(`8 layers x 32 x 128 x 2B x ctx x 2.0`); the code-derived value for the
whole KV subsystem is 5.6MB — three orders of magnitude lower. Only 4
layers own index keys, sized S//ratio in fp4.

## Unified table (@131k ctx, 12 slots/layer)

common-model 9.1GB + dense-engram 0.3GB + expert-pool 8.6GB + kv 0.1GB
+ runtime 1GB = 19.1GB vs 45.1GB operating → **25.9GB headroom**.

The expert pool, not KV, is the resident-memory question on this
machine. P5 scheduling complexity belongs there first.

## Usage

    python scripts/size_kv.py --batch 1 --seqs 4096,32768,131072 --slots 12
