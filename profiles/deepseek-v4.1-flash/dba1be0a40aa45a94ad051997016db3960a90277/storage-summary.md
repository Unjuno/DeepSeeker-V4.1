# Storage summary — DeepSeek-V4.1-Flash

Revision: `dba1be0a40aa45a94ad051997016db3960a90277`
Schema: `deepseeker.model-manifest/v1`
Generated: 2026-09-18T00:33:27.673819+00:00 (DeepSeeker ce68765ceccf34f8587a0fe0330b83ad270b3e37)

## Totals

- Tensors: 96085
- Tensor bytes: 510286023000 (475.24 GiB)
- Streamable (routed experts): 295997276160 (275.67 GiB)
- Always-needed: 214288746840 (199.57 GiB)

## By subsystem

- attention: 520 tensors, 5069721600 bytes (4.72 GiB)
- embedding: 4 tensors, 1323857920 bytes (1.23 GiB)
- engram: 12 tensors, 203073076240 bytes (189.13 GiB)
- kv-index: 43 tensors, 81835008 bytes (0.08 GiB)
- lm-head: 1 tensors, 1323827200 bytes (1.23 GiB)
- mtp: 52 tensors, 591424512 bytes (0.55 GiB)
- mtp-expert: 2304 tensors, 7219445760 bytes (6.72 GiB)
- mtp-router: 9 tensors, 3935232 bytes (0.00 GiB)
- mtp-shared: 18 tensors, 106272000 bytes (0.10 GiB)
- norm: 81 tensors, 829440 bytes (0.00 GiB)
- routed-expert: 92160 tensors, 288777830400 bytes (268.95 GiB)
- router: 120 tensors, 157409280 bytes (0.15 GiB)
- shared-expert: 240 tensors, 1416960000 bytes (1.32 GiB)
- unknown: 258 tensors, 169092168 bytes (0.16 GiB)
- vision: 263 tensors, 970506240 bytes (0.90 GiB)

## Routed experts

- Experts: 15360
- Min/median/max bytes: 18800640 / 18800640 / 18800640

## Validation

- weight_map entries: 96085
- header entries: 96085
- missing from headers: 0
- extra in headers: 0
- accounting reconciled: True

## Transport

- Shards: 48
- Network bytes (headers+index only): 10685312
- Weights downloaded: none

## Offset convention

data_range is [start, end) relative to the safetensors data-section start; file_range is absolute from file byte 0 (8-byte header length + JSON header + relative offset).

## Unknown tensors

- 258 tensors, 169092168 bytes. Enumerated in the manifest with full names; none dropped.
