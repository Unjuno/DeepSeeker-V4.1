"""Issue #14: routing motifs + speculative-block working sets (roadmap P2/P6).

Two offline analyses over Issue #8 traces:

  layer bigrams  - frequent ((layer, expert) -> (layer+1, expert)) pairs
                   within a token: the multi-layer routing motifs P6
                   block verification would exploit.
  block sets     - unique experts per (request, block of B tokens, layer):
                   sublinear growth in B is the weight-reuse opportunity
                   for speculative multi-token blocks.
"""

from __future__ import annotations

from collections import Counter


def layer_bigrams(records: list[dict]) -> Counter[tuple[tuple[int, int], tuple[int, int]]]:
    """Count ((layer, expert) -> (layer+1, expert)) co-occurrences in tokens."""
    by_token: dict[tuple[str, int], dict[int, list[int]]] = {}
    for record in records:
        by_token.setdefault((record["request_id"], record["token_pos"]), {})[
            record["layer"]
        ] = record["experts"]
    pairs: Counter[tuple[tuple[int, int], tuple[int, int]]] = Counter()
    for layers in by_token.values():
        for layer in sorted(layers):
            if layer + 1 not in layers:
                continue
            for src in layers[layer]:
                for dst in layers[layer + 1]:
                    pairs[((layer, src), (layer + 1, dst))] += 1
    return pairs


def motif_coverage(
    records: list[dict], motifs: set[tuple[tuple[int, int], tuple[int, int]]]
) -> float:
    """Fraction of adjacent-layer expert pairs covered by the motif set."""
    covered = 0
    total = 0
    by_token: dict[tuple[str, int], dict[int, list[int]]] = {}
    for record in records:
        by_token.setdefault((record["request_id"], record["token_pos"]), {})[
            record["layer"]
        ] = record["experts"]
    for layers in by_token.values():
        for layer in sorted(layers):
            if layer + 1 not in layers:
                continue
            for src in layers[layer]:
                for dst in layers[layer + 1]:
                    total += 1
                    if ((layer, src), (layer + 1, dst)) in motifs:
                        covered += 1
    return covered / total if total else 0.0


def block_working_sets(records: list[dict], block_size: int) -> list[dict]:
    """Unique expert sets per (request, token-block, layer)."""
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size!r}")
    blocks: dict[tuple[str, int, int], set[int]] = {}
    for record in records:
        key = (record["request_id"], record["token_pos"] // block_size, record["layer"])
        blocks.setdefault(key, set()).update(record["experts"])
    return [
        {
            "request_id": request,
            "block": block,
            "layer": layer,
            "unique_experts": sorted(experts),
            "unique_count": len(experts),
        }
        for (request, block, layer), experts in sorted(blocks.items())
    ]


def block_summary(records: list[dict], block_sizes: list[int], top_k: int) -> list[dict]:
    """Mean unique experts per block for each block size (reuse curve).

    reuse_ratio = mean_unique / (block_size * top_k): 1.0 means every
    token brings all-new experts; lower means block-level weight reuse.
    """
    rows = []
    for size in block_sizes:
        sets = block_working_sets(records, size)
        mean_unique = sum(s["unique_count"] for s in sets) / len(sets) if sets else 0.0
        rows.append(
            {
                "block_size": size,
                "blocks": len(sets),
                "mean_unique_experts": mean_unique,
                "reuse_ratio": mean_unique / (size * top_k) if size * top_k else 0.0,
            }
        )
    return rows
