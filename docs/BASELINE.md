# Baseline reference verification (Issue #7)

Verifies the vendored DeepSeek-V4.1-Flash reference tree locally without
requiring model weights, so it runs before the Issue #6 checkpoint finishes.

## Finding fixed by this issue

`inference/generate.py` does `from encoding import ...`, but the `encoding/`
package lives at the upstream repo root, outside the `inference/` subtree
that `fetch_upstream.py` collected. The vendored tree was therefore missing
`encoding/encoding.py` and no reference entry point could import.

Fix: `fetch_upstream.py` allow-patterns now include `encoding/*` and
`encoding/**/*`; the pinned subtree was fetched (13 files, weights excluded).

## Harness

    python scripts/verify_reference.py

Blocking checks (exit 1 on failure):

- `pin_consistency` — `DEEPSEEKER_UPSTREAM.json` revision equals the Issue #2
  manifest revision (`dba1be0a...`).
- `reference_files` — all expected `inference/` + `encoding/` files present.
- `cross_module_imports` — AST check that every sibling import resolves to a
  vendored file. This is the regression test for the missing-`encoding` bug.
- `arch_reconciliation` — `inference/config.json` values
  (`dim=5120`, `n_layers=40`, `n_routed_experts=384`,
  `n_activated_experts=6`, `moe_inter_dim=2304`, `n_shared_experts=1`,
  `n_mtp_layers=3`, `dtype=fp8`, `expert_dtype=fp4`) equal the manifest.
- `import_smoke` — real import of the torch-free `encoding.encoding` module.

Advisory checks (reported, never fail):

- `requirements` — `torch`, `Pillow`, `tilelang==0.1.8`, etc. Full
  `generate.py` needs `pip install -r upstream/DeepSeek-V4.1-Flash/inference/requirements.txt`.
  Note: `tilelang` has no prebuilt macOS/ARM wheel; expect a source build.
- `checkpoint_readiness` — Issue #6 progress; a real baseline run needs
  48/48 verified shards plus the four small files, then
  `convert.py --model-parallel 8` to build the TP8 checkpoint `run.sh` wants.

## Full baseline run (once ready)

1. `python scripts/manage_checkpoint.py verify` → 48/48 PASS.
2. `pip install -r upstream/DeepSeek-V4.1-Flash/inference/requirements.txt`
3. `convert.py --hf-ckpt-path models/DeepSeek-V4.1-Flash --save-path <TP8> --model-parallel 8`
4. `INPUT_FILE=inference/examples/example_harmony.json CKPT_PATH=<TP8> ./inference/run.sh`

Record results per the benchmark rule in `docs/ROADMAP.md`. No predictive
optimization is in scope here; the router stays authoritative (P1 ground
truth comes next).
