# Upstream integration

## Canonical source

DeepSeeker currently treats the official Hugging Face model repository as the canonical DeepSeek-V4.1-Flash reference source:

- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- local reference implementation: `inference/`

At project bootstrap time the upstream repository is licensed under MIT.

## Why upstream code is fetched instead of vendored by default

DeepSeeker needs the official implementation for:

- architecture/config inspection
- trace instrumentation
- correctness/reference comparisons
- weight conversion behavior
- routing / KV / Engram integration points

But mixing upstream files directly into `src/deepseeker/` would make ownership and update tracking harder.

The default workflow therefore keeps:

```text
src/deepseeker/                 # DeepSeeker-owned implementation
upstream/DeepSeek-V4.1-Flash/  # fetched official reference snapshot, ignored by Git
```

separate.

## Fetching

```bash
pip install -e .
python scripts/fetch_upstream.py
```

To pin a known upstream commit explicitly:

```bash
python scripts/fetch_upstream.py --revision <commit-sha>
```

The fetcher first resolves the requested revision to an immutable Hugging Face commit SHA, downloads only selected code/configuration/license files, and writes:

```text
upstream/DeepSeek-V4.1-Flash/DEEPSEEKER_UPSTREAM.json
```

Weights are excluded.

## If upstream code is later vendored or patched

If a future DeepSeeker optimization requires checked-in modifications of DeepSeek code:

1. place them under an explicit third-party or patch boundary;
2. preserve DeepSeek's copyright and MIT license notice;
3. record the exact upstream revision;
4. keep modifications reviewable as patches where practical;
5. do not silently move copied upstream code into DeepSeeker-owned modules.

A preferred pattern is:

```text
patches/deepseek-v4.1/<upstream-sha>/
```

with small patches applied to a fetched upstream snapshot during development.

## Weights

DeepSeeker does not redistribute DeepSeek model weights. Weight storage, conversion, and local caching belong outside Git history.
