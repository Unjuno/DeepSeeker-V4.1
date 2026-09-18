# Canonical tensor manifest (metadata only, no weights)

## Run

```bash
python scripts/build_model_manifest.py
```

Reads the pinned revision from `upstream/DEEPSEEKER_UPSTREAM.json`
(currently `dba1be0a40aa45a94ad051997016db3960a90277`) and refuses any
other revision unless `--allow-other-revision` is passed explicitly.

Output:

```text
profiles/
└── deepseek-v4.1-flash/
    └── <revision>/
        ├── model-manifest.json  # full inventory (~40 MB, committed)
        └── storage-summary.md   # aggregate report (generated)
```

Transport: safetensors index (one small JSON) + HTTP Range header-only
fetch per shard (8-byte prefix + JSON header). Full 510 GB map cost
~10.7 MB of network traffic; zero weight bytes downloaded or committed.

## Schema (`deepseeker.model-manifest/v1`)

- `upstream{repo_id, revision}`, `generated_at_utc`, `deepseeker_commit`
- `architecture`: values extracted from pinned `config.json`
  (hidden 5120, 40 layers, 384 routed experts, top-6, 1 shared,
  MoE intermediate 2304, 3 next-token layers, FP4 experts)
- Per tensor: name, shape, dtype, elements, bytes (from
  `data_offsets`, authoritative), shard, `data_range` (relative to
  data-section start), `file_range` (absolute), classification
  `{subsystem, layer, expert, routed, detail}`
- `totals{by_subsystem}`, `per_layer`, `per_expert_bytes`,
  `expert_stats`, `validation`, `shards`, `transport`

## Key figures (pinned revision)

- 96,085 tensors / 510,286,023,000 bytes (475.24 GiB) across 48 shards;
  tensor bytes match the index `total_size` exactly; container overhead 0.
- Routed experts: 15,360 (40 layers × 384), uniform 18,800,640 bytes
  each (w1/w2/w3 weight+scale), 268.95 GiB total — streamable pool.
- Always-needed: 199.57 GiB. Notable: Engram is 12 tensors but
  189.13 GiB — dominates the always-needed side; residency planning must
  treat it separately from backbone weights.
- MTP: 3 layers × 128 experts (2,304 tensors, 6.72 GiB) + routers/shared.
- Unknown: 258 tensors (0.16 GiB), all `hc_*` auxiliaries (40 backbone +
  3 MTP layers × 6); semantics unconfirmed, kept visible, none dropped.
- Dtypes: BF16 520, F32 387, F8_E8M0 47,589 (FP4 scales, ue8m0),
  F8_E4M3 357, I8 47,232 (packed FP4 weights).
- Validation: 0 missing / 0 extra vs `weight_map`; buckets reconcile
  exactly; two consecutive runs identical modulo timestamp.

## Query example

```python
m = json.load(open("profiles/deepseek-v4.1-flash/<rev>/model-manifest.json"))
ts = [t for t in m["tensors"] if t.get("layer") == 12
      and t.get("expert") == 305 and t["subsystem"] == "routed-expert"]
# 6 tensors, one shard, each with data_range + file_range for random access
```

## Residency rule (documented in manifest)

Streamable = `routed-expert` + `mtp-expert` only; everything else is
always-needed for planning. Vision/Engram conditionality is a planning
concern for the next issue, not encoded here.
