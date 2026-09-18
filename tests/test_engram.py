"""Tests for Engram access semantics and row mapping.

Test A needs torch/transformers and the pinned upstream reference code;
it is skipped when those are unavailable. All other tests are hermetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepseeker.engram import (
    EngramGeometry,
    amplified_bytes,
    build_multipliers,
    build_offsets,
    build_primes,
    coalesce,
    hash_positions,
    lru_hit_rate,
    row_ranges,
    table_refs_from_manifest,
)

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

UPSTREAM_INFERENCE = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "inference"
if not (UPSTREAM_INFERENCE / "engram.py").exists():
    pytest.skip("pinned upstream reference code not fetched", allow_module_level=True)

sys.path.insert(0, str(UPSTREAM_INFERENCE))
import engram as official

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
CACHE_DIR = REPO_ROOT / ".cache" / "engram-analysis"


def _geometry_and_map():
    from transformers import AutoTokenizer

    analyzer_spec = importlib.util.spec_from_file_location(
        "analyze_engram_access",
        REPO_ROOT / "scripts" / "analyze_engram_access.py",
    )
    assert analyzer_spec is not None and analyzer_spec.loader is not None
    analyzer = importlib.util.module_from_spec(analyzer_spec)
    analyzer_spec.loader.exec_module(analyzer)
    analyzer.load_tokenizer(CACHE_DIR)  # ensure pinned files are cached
    manifest_path = REPO_ROOT / "upstream" / "DeepSeek-V4.1-Flash" / "DEEPSEEKER_UPSTREAM.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["resolved_revision"] == REVISION
    tok = AutoTokenizer.from_pretrained(
        str(CACHE_DIR), local_files_only=True, trust_remote_code=False
    )
    token_map, compressed_vocab = official.build_compressed_token_map(tok)
    assert compressed_vocab == 99092
    geometry = EngramGeometry(
        layer_ids=(1, 14),
        max_ngram_size=4,
        n_heads=8,
        head_dim=256,
        num_embeddings=(384006168, 384016682),
        vocab_size=16000000,
        compressed_vocab_size=compressed_vocab,
        pad_id=token_map[2],
    )
    return tok, token_map, geometry


@pytest.fixture(scope="module")
def official_state():
    from types import SimpleNamespace

    tok, token_map, geometry = _geometry_and_map()
    args = SimpleNamespace(
        engram_layer_ids=(1, 14),
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_head_dim=256,
        num_embeddings=(384006168, 384016682),
        engram_num_embeddings=(384006168, 384016682),
        engram_vocab_size=16000000,
        engram_compressed_vocab_size=99092,
        engram_pad_id=2,
        max_batch_size=2,
        max_seq_len=4096,
    )
    layout = official.EngramLayout.from_args(args)
    state = official.NgramHashState(args, layout, tok)
    multipliers = build_multipliers(
        geometry.layer_ids,
        geometry.max_ngram_size,
        geometry.compressed_vocab_size,
    )
    primes = build_primes(geometry)
    # Prime allocation must match the official layout exactly.
    assert primes == layout.primes, "prime sequence diverged from upstream"
    offsets = build_offsets(primes)
    assert offsets == [list(map(int, row)) for row in state.offsets.tolist()]
    assert multipliers.tolist() == state.multipliers.detach().cpu().tolist()
    return tok, token_map, geometry, state


def _official_hashes(state, ids, dead_positions=()):
    mask = torch.ones(1, len(ids), dtype=torch.bool)
    for pos in dead_positions:
        mask[0, pos] = False
    with torch.inference_mode():
        return state(torch.tensor([ids]), 0, mask).tolist()


def _mine(token_map, geometry, ids, dead=()):
    multipliers = build_multipliers(
        geometry.layer_ids,
        geometry.max_ngram_size,
        geometry.compressed_vocab_size,
    )
    primes = build_primes(geometry)
    offsets = build_offsets(primes)
    compressed = [token_map[i] for i in ids]
    return hash_positions(compressed, geometry, multipliers, primes, offsets, set(dead))


def test_hash_equivalence_plain(official_state):
    tok, token_map, geometry, state = official_state
    ids = tok.encode("The harbor lights flickered across the water.")
    expected = _mine(token_map, geometry, ids)
    got = _official_hashes(state, ids)
    assert got[0] == expected


def test_hash_equivalence_with_dead_tokens(official_state):
    tok, token_map, geometry, state = official_state
    ids = tok.encode("one two three four five six seven eight")
    dead = {2, 5}
    assert _official_hashes(state, ids, dead)[0] == _mine(token_map, geometry, ids, dead)


def test_hash_equivalence_short_and_single_token(official_state):
    tok, token_map, geometry, state = official_state
    for text in ("Hi", "Hello, world!"):
        ids = tok.encode(text)
        assert _official_hashes(state, ids)[0] == _mine(token_map, geometry, ids)


def test_official_prefill_decode_consistency(official_state):
    tok, _token_map, _geometry, state = official_state
    ids = tok.encode("Mara counted the masts from the stone pier.")
    full = _official_hashes(state, ids)[0]
    # Prefill first 4, then decode one step at a time through the cache.
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.inference_mode():
        state(torch.tensor([ids[:4]]), 0, mask)
        for k in range(4, len(ids)):
            m = torch.ones(1, 1, dtype=torch.bool)
            step = state(torch.tensor([[ids[k]]]), k, m).tolist()[0][0]
            assert step == full[k]


# --------------------------------------------------------------------------
# Hermetic geometry / mapping / accounting tests


def _manifest():
    path = REPO_ROOT / "profiles" / "deepseek-v4.1-flash" / REVISION / "model-manifest.json"
    if not path.exists():
        pytest.skip("tensor manifest not generated yet")
    return json.loads(path.read_text(encoding="utf-8"))


def test_row_geometry_from_manifest():
    manifest = _manifest()
    refs = table_refs_from_manifest(manifest)
    assert sorted(refs) == [1, 14]
    assert refs[1].num_rows == 384006168
    assert refs[14].num_rows == 384016682
    for ref in refs.values():
        assert ref.weight_row_bytes == 256
        assert ref.scale_row_bytes == 8


def test_row_range_boundaries():
    manifest = _manifest()
    refs = table_refs_from_manifest(manifest)
    ref = refs[1]
    for row in (0, ref.num_rows // 2, ref.num_rows - 1):
        rr = row_ranges(ref, row)
        for part in ("weight", "scale"):
            s, e = rr[part]["range"]
            lo, hi = ref.weight_file_range if part == "weight" else ref.scale_file_range
            assert lo <= s < e <= hi
    with pytest.raises(ValueError):
        row_ranges(ref, ref.num_rows)
    with pytest.raises(ValueError):
        row_ranges(ref, -1)


def test_amplified_bytes_and_coalesce_exact():
    ranges = [(0, 264), (4096, 4360), (8192, 8456)]
    pages, total = amplified_bytes(ranges, 4096)
    assert (pages, total) == (3, 3 * 4096)
    merged = coalesce([(0, 100), (150, 200), (10000, 10100)], 100)
    assert merged == [(0, 200), (10000, 10100)]
    assert coalesce([(0, 100)], 0) == [(0, 100)]


def test_lru_cache_monotonic_on_repetitive_sequence():
    requests = (list(range(50)) * 20)[:1000]
    hits = [lru_hit_rate(requests, cap) for cap in (10, 50, 200)]
    assert hits == sorted(hits)
    assert hits[-1] == 0.95  # only the first 50 fills miss
    assert lru_hit_rate([], 10) == 0.0
    assert lru_hit_rate([1, 2], 0) == 0.0


def test_hash_deterministic():
    geometry = EngramGeometry(
        layer_ids=(1, 14),
        max_ngram_size=4,
        n_heads=8,
        head_dim=256,
        num_embeddings=(384006168, 384016682),
        vocab_size=16000000,
        compressed_vocab_size=99092,
        pad_id=7,
    )
    multipliers = build_multipliers((1, 14), 4, 99092)
    primes = build_primes(geometry)
    offsets = build_offsets(primes)
    seq = [5, 100, 7000, 42, 42, 9, 100000, 3]
    first = hash_positions(seq, geometry, multipliers, primes, offsets)
    second = hash_positions(seq, geometry, multipliers, primes, offsets)
    assert first == second
    assert len(first) == len(seq)
    assert all(len(pos) == 2 and len(pos[0]) == 24 for pos in first)
