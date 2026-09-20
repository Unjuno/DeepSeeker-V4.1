"""Issue #7: local reference-code verification and baseline readiness.

Static checks over the vendored DeepSeek-V4.1-Flash reference tree plus
environment/revision reconciliation. Nothing here needs model weights, so
the harness runs before the Issue #6 checkpoint finishes downloading.

Checklist:
  pin_consistency      - DEEPSEEKER_UPSTREAM.json revision == Issue #2 manifest
  reference_files      - expected inference/ + encoding/ files exist
  cross_module_imports - every sibling `from X import` resolves to a
                         vendored file (AST check; would have caught the
                         missing encoding/ package)
  requirements         - importlib find_spec per requirements.txt module
  arch_reconciliation  - inference/config.json values == Issue #2 manifest
  checkpoint_readiness - Issue #6 state counts + small-file presence
  import_smoke         - real import of the torch-free reference modules
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

# Files the reference runtime needs, relative to the upstream snapshot root.
REFERENCE_FILES = (
    "inference/model.py",
    "inference/generate.py",
    "inference/convert.py",
    "inference/config.json",
    "inference/engram.py",
    "inference/kernel.py",
    "inference/vision.py",
    "inference/image_processor.py",
    "inference/run.sh",
    "inference/requirements.txt",
    "inference/examples/example_harmony.json",
    "encoding/encoding.py",
    "encoding/test_encoding.py",
    "encoding/README.md",
)

# requirements.txt module -> distribution hint shown when missing.
REQUIREMENT_MODULES = {
    "torch": "torch>=2.10.0",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "safetensors": "safetensors>=0.7.0",
    "numpy": "numpy",
    "sympy": "sympy",
    "PIL": "Pillow",
    "tilelang": "tilelang==0.1.8",
    "tqdm": "tqdm",
    "pytest": "pytest",
}

# inference/config.json key -> Issue #2 manifest architecture key.
ARCH_MAP = (
    ("dim", "hidden_size"),
    ("n_layers", "num_hidden_layers"),
    ("n_routed_experts", "n_routed_experts"),
    ("n_activated_experts", "num_experts_per_tok"),
    ("moe_inter_dim", "moe_intermediate_size"),
    ("n_shared_experts", "n_shared_experts"),
    ("n_mtp_layers", "num_nextn_predict_layers"),
    # NOTE: manifest "dtype" comes from the root transformers-style
    # config.json (stored weight dtype, bfloat16), while inference/
    # config.json "dtype" is the runtime compute dtype (fp8). Different
    # domains: compared against manifest quantization instead (below).
)

SMALL_FILES = (
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
)

# Torch-free modules safe for a real import smoke test.
SMOKE_MODULES = ("encoding.encoding",)


def check_pin(upstream_meta: dict, manifest: dict) -> dict:
    """Upstream pin must match the manifest's recorded revision."""
    pinned = upstream_meta.get("resolved_revision")
    recorded = (manifest.get("upstream") or {}).get("revision")
    ok = bool(pinned) and pinned == recorded
    return {
        "name": "pin_consistency",
        "ok": ok,
        "pinned": pinned,
        "manifest_revision": recorded,
        "problems": [] if ok else [f"pin {pinned!r} != manifest {recorded!r}"],
    }


def check_reference_files(snapshot_root: Path) -> dict:
    """Every expected reference file must exist with nonzero size."""
    missing = [
        name
        for name in REFERENCE_FILES
        if not (snapshot_root / name).is_file()
        or (snapshot_root / name).stat().st_size == 0
    ]
    return {
        "name": "reference_files",
        "ok": not missing,
        "checked": len(REFERENCE_FILES),
        "problems": [f"missing: {m}" for m in missing],
    }


def _sibling_imports(path: Path) -> set[str]:
    """Top-level module names imported in a file (AST, no execution)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def check_cross_module_imports(snapshot_root: Path) -> dict:
    """Sibling imports inside inference/ + encoding/ must resolve locally.

    Third-party imports are ignored here (covered by `requirements`).
    """
    inference = snapshot_root / "inference"
    encoding_dir = snapshot_root / "encoding"
    local_modules = set()
    for directory in (inference, encoding_dir):
        if directory.is_dir():
            local_modules.update(
                p.stem for p in directory.glob("*.py") if p.name != "__init__.py"
            )
    third_party = set(REQUIREMENT_MODULES) | {
        "math",
        "json",
        "os",
        "re",
        "sys",
        "copy",
        "shutil",
        "glob",
        "pathlib",
        "typing",
        "argparse",
        "dataclasses",
        "functools",
        "contextlib",
        "struct",
    }
    problems = []
    checked = 0
    for directory in (inference, encoding_dir):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("test_"):
                continue
            checked += 1
            for name in sorted(_sibling_imports(path)):
                if name in third_party or name in sys.stdlib_module_names:
                    continue
                if name not in local_modules:
                    problems.append(f"{path.parent.name}/{path.name}: unresolved `{name}`")
    return {
        "name": "cross_module_imports",
        "ok": not problems,
        "checked": checked,
        "problems": problems,
    }


def check_requirements() -> dict:
    """Every requirements.txt module must be importable."""
    missing = {
        mod: hint
        for mod, hint in REQUIREMENT_MODULES.items()
        if importlib.util.find_spec(mod) is None
    }
    return {
        "name": "requirements",
        "ok": not missing,
        "checked": len(REQUIREMENT_MODULES),
        "missing": missing,
        "problems": [f"missing module `{m}` (pip: {h})" for m, h in missing.items()],
    }


def check_arch(manifest: dict, config: dict) -> dict:
    """Reference config values must equal the Issue #2 manifest architecture."""
    arch = manifest.get("architecture", {})
    problems = []
    compared = 0
    for cfg_key, arch_key in ARCH_MAP:
        if cfg_key not in config or arch_key not in arch:
            problems.append(f"key absent: config[{cfg_key!r}] or arch[{arch_key!r}]")
            continue
        compared += 1
        if config[cfg_key] != arch[arch_key]:
            problems.append(
                f"{cfg_key}: config={config[cfg_key]!r} != manifest={arch[arch_key]!r}"
            )
    quant = manifest.get("architecture", {}).get("quantization") or {}
    for cfg_key, quant_key in (("expert_dtype", "expert_dtype"), ("dtype", "quant_method")):
        if cfg_key in config and quant.get(quant_key) is not None:
            compared += 1
            if config[cfg_key] != quant[quant_key]:
                problems.append(
                    f"{cfg_key}: config={config[cfg_key]!r} != manifest.{quant_key}={quant[quant_key]!r}"
                )
    return {
        "name": "arch_reconciliation",
        "ok": not problems,
        "compared": compared,
        "problems": problems,
    }


def check_checkpoint_readiness(model_root: Path, state: dict) -> dict:
    """Report Issue #6 progress; ready only at 48/48 verified + small files."""
    files = state.get("files", {}) if isinstance(state, dict) else {}
    shards = {k: v for k, v in files.items() if k.endswith(".safetensors")}
    verified = sum(1 for v in shards.values() if v.get("status") == "verified")
    downloaded = sum(
        1 for v in shards.values() if v.get("status") == "downloaded-unverified"
    )
    small = {
        name: (model_root / name).is_file() and (model_root / name).stat().st_size > 0
        for name in SMALL_FILES
    }
    ready = verified == 48 and all(small.values())
    problems = []
    if verified != 48:
        problems.append(f"shards verified {verified}/48 (downloaded: {downloaded})")
    for name, present in small.items():
        if not present:
            problems.append(f"small file missing: {name}")
    return {
        "name": "checkpoint_readiness",
        "ok": ready,
        "verified": verified,
        "downloaded_unverified": downloaded,
        "small_files": small,
        "problems": problems,
    }


def import_smoke(snapshot_root: Path) -> dict:
    """Really import the torch-free reference modules (catches syntax errors)."""
    saved = list(sys.path)
    # snapshot_root exposes the `encoding` namespace package; inference/
    # exposes the flat `model`, `kernel`, ... modules. (The encoding/
    # directory itself must NOT be on sys.path: its encoding.py would
    # shadow the `encoding` package.)
    sys.path.insert(0, str(snapshot_root / "inference"))
    sys.path.insert(0, str(snapshot_root))
    problems = []
    imported = []
    try:
        for name in SMOKE_MODULES:
            try:
                sys.modules.pop(name, None)
                __import__(name)
                imported.append(name)
            except Exception as exc:  # noqa: BLE001 - report, don't raise
                problems.append(f"import {name}: {type(exc).__name__}: {exc}")
    finally:
        sys.path[:] = saved
        for name in SMOKE_MODULES:
            sys.modules.pop(name, None)
    return {
        "name": "import_smoke",
        "ok": not problems,
        "imported": imported,
        "problems": problems,
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_root(repo_root: Path, override: Path | None = None) -> Path:
    """Model root honoring --model-root, then DEEPSEEKER_MODEL_ROOT.

    An unset *or empty* environment variable falls back to the default;
    never silently resolve to the current directory.
    """
    import os

    if override is not None:
        return override
    env = os.environ.get("DEEPSEEKER_MODEL_ROOT")
    if env:
        return Path(env)
    return repo_root / "models" / "DeepSeek-V4.1-Flash"
