#!/usr/bin/env python3
"""Issue #7: verify the vendored DeepSeek-V4.1-Flash reference locally.

    python scripts/verify_reference.py
    python scripts/verify_reference.py --model-root models/DeepSeek-V4.1-Flash

Static checks run without weights. Exit 0 only when every blocking check
passes; advisory checks (runtime deps, checkpoint completion) are reported
but never fail the run, since they depend on environment/download state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import (
    check_arch,
    check_checkpoint_readiness,
    check_cross_module_imports,
    check_pin,
    check_reference_files,
    check_requirements,
    import_smoke,
    load_json,
    resolve_model_root,
)

SNAPSHOT = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash"


def pinned_revision() -> str | None:
    path = SNAPSHOT / "DEEPSEEKER_UPSTREAM.json"
    if not path.exists():
        return None
    try:
        return load_json(path).get("resolved_revision")
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify vendored reference code.")
    parser.add_argument("--model-root", type=Path, default=None)
    args = parser.parse_args()

    revision = pinned_revision()
    if revision is None:
        print("FAIL pin_consistency: pinned revision unknown (run fetch_upstream first)")
        return 1
    manifest_path = (
        REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / revision / "model-manifest.json"
    )
    if not manifest_path.exists():
        print(f"FAIL: manifest not found: {manifest_path}")
        return 1

    manifest = load_json(manifest_path)
    upstream_meta = load_json(SNAPSHOT / "DEEPSEEKER_UPSTREAM.json")
    config = load_json(SNAPSHOT / "inference" / "config.json")

    blocking = [
        check_pin(upstream_meta, manifest),
        check_reference_files(SNAPSHOT),
        check_cross_module_imports(SNAPSHOT),
        check_arch(manifest, config),
        import_smoke(SNAPSHOT),
    ]
    advisory = [check_requirements()]

    model_root = resolve_model_root(REPO_ROOT, args.model_root)
    state_path = model_root / ".deepseeker-checkpoint.json"
    state = load_json(state_path) if state_path.exists() else {}
    advisory.append(check_checkpoint_readiness(model_root, state))

    failed = False
    for result in blocking:
        status = "PASS" if result["ok"] else "FAIL"
        print(f"{status} {result['name']}", flush=True)
        for problem in result.get("problems", []):
            print(f"  - {problem}")
        failed = failed or not result["ok"]
    for result in advisory:
        status = "ok" if result["ok"] else "ADVISORY"
        print(f"{status} {result['name']} (non-blocking)", flush=True)
        for problem in result.get("problems", []):
            print(f"  - {problem}")
    if failed:
        return 1
    print("reference verification: PASS (blocking checks)")
    print("note: full generate.py run awaits checkpoint completion + runtime deps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
