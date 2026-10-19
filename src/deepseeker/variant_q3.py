"""Variant-aware manifest/model-root selection for the Q3 pilot.

DEEPSEEKER_VARIANT can be:
  "1017"       - the MXFP4 baseline checkpoint (verbatim)
  "q3" / "1017-q3" - the Q3 pilot: routed experts layers 20-39 converted
                   to int3 affine (see requant_experts_3bit.py); manifest
                   includes .bias tensors and U32_Q3_AFFINE weight dtypes
"""
from __future__ import annotations

import os
from pathlib import Path


def is_q3(variant: str | None = None) -> bool:
    v = variant if variant is not None else os.environ.get(
        "DEEPSEEKER_VARIANT", "")
    return v.lower().endswith("q3")


def paths_for_variant(revision: str, repo_root: Path, snapshot_name: str,
                      variant: str | None = None) -> tuple[Path, Path]:
    """(model_root, manifest_path) for the active variant."""
    profile = repo_root / "profiles" / "deepseek-v4.1-flash" / revision
    if is_q3(variant):
        manifest_name = os.environ.get("DEEPSEEKER_Q3_MANIFEST",
                                       "model-manifest-q3.json")
        root = repo_root / "models" / f"{snapshot_name}-Q3"
    else:
        manifest_name = "model-manifest.json"
        root = repo_root / "models" / snapshot_name
    fallback = repo_root / "models" / "DeepSeek-V4.1-Flash"
    return (root if root.exists() else fallback,
            profile / manifest_name)
