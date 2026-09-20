"""Tests for Issue #7 reference verification (committed artifacts only)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.baseline import (
    ARCH_MAP,
    REFERENCE_FILES,
    SMALL_FILES,
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
REVISION = load_json(SNAPSHOT / "DEEPSEEKER_UPSTREAM.json")["resolved_revision"]
MANIFEST = load_json(
    REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / REVISION / "model-manifest.json"
)
CONFIG = load_json(SNAPSHOT / "inference" / "config.json")


def test_pin_matches_manifest():
    meta = load_json(SNAPSHOT / "DEEPSEEKER_UPSTREAM.json")
    result = check_pin(meta, MANIFEST)
    assert result["ok"], result["problems"]
    assert result["pinned"] == "dba1be0a40aa45a94ad051997016db3960a90277"


def test_pin_mismatch_detected():
    result = check_pin({"resolved_revision": "deadbeef"}, MANIFEST)
    assert not result["ok"]
    assert result["problems"]


def test_reference_files_present():
    result = check_reference_files(SNAPSHOT)
    assert result["ok"], result["problems"]
    assert result["checked"] == len(REFERENCE_FILES)
    assert "encoding/encoding.py" in REFERENCE_FILES


def test_reference_files_missing_detected(tmp_path):
    result = check_reference_files(tmp_path)
    assert not result["ok"]
    assert len(result["problems"]) == len(REFERENCE_FILES)


def test_cross_module_imports_resolve():
    result = check_cross_module_imports(SNAPSHOT)
    assert result["ok"], result["problems"]
    assert result["checked"] > 0


def test_cross_module_imports_catch_missing(tmp_path):
    (tmp_path / "inference").mkdir()
    (tmp_path / "inference" / "generate.py").write_text(
        "from nonexistent_module import thing\n", encoding="utf-8"
    )
    result = check_cross_module_imports(tmp_path)
    assert not result["ok"]
    assert any("nonexistent_module" in p for p in result["problems"])


def test_arch_reconciliation():
    result = check_arch(MANIFEST, CONFIG)
    assert result["ok"], result["problems"]
    assert result["compared"] == len(ARCH_MAP) + 2  # + expert_dtype, quant_method


def test_arch_mismatch_detected():
    bad = dict(CONFIG, dim=1)
    result = check_arch(MANIFEST, bad)
    assert not result["ok"]
    assert any("dim" in p for p in result["problems"])


def test_requirements_shape():
    result = check_requirements()
    assert result["checked"] > 0
    assert isinstance(result["missing"], dict)
    # torch is installed in this environment; its absence would be fatal.
    assert "torch" not in result["missing"]


def test_import_smoke():
    result = import_smoke(SNAPSHOT)
    assert result["ok"], result["problems"]
    assert "encoding.encoding" in result["imported"]


def test_checkpoint_readiness_reports_progress(tmp_path):
    state = {
        "files": {
            "model-00001-of-00048.safetensors": {"status": "verified"},
            "model-00002-of-00048.safetensors": {"status": "downloaded-unverified"},
        }
    }
    result = check_checkpoint_readiness(tmp_path, state)
    assert not result["ok"]  # not 48/48 yet
    assert result["verified"] == 1
    assert result["downloaded_unverified"] == 1
    assert len(result["small_files"]) == len(SMALL_FILES)


def test_resolve_model_root_never_cwd(tmp_path, monkeypatch):
    default = REPO_ROOT / "models" / "DeepSeek-V4.1-Flash"
    monkeypatch.delenv("DEEPSEEKER_MODEL_ROOT", raising=False)
    assert resolve_model_root(REPO_ROOT) == default
    monkeypatch.setenv("DEEPSEEKER_MODEL_ROOT", "")
    assert resolve_model_root(REPO_ROOT) == default
    monkeypatch.setenv("DEEPSEEKER_MODEL_ROOT", str(tmp_path))
    assert resolve_model_root(REPO_ROOT) == tmp_path
    assert resolve_model_root(REPO_ROOT, tmp_path / "x") == tmp_path / "x"
