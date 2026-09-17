# Upstream workspace

This directory is reserved for locally fetched DeepSeek-V4.1-Flash reference code and metadata.

Everything under `upstream/` is ignored by Git except this file.

Run:

```bash
python scripts/fetch_upstream.py
```

The default destination is:

```text
upstream/DeepSeek-V4.1-Flash/
```

The fetcher downloads only selected reference code/configuration/license files and records the exact resolved upstream revision. It deliberately excludes model weight shards.

Do not remove upstream copyright or license notices when copying or modifying upstream code.
