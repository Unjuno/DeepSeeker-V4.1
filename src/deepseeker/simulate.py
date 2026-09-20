"""Issue #9: P2 offline routing + cache simulator (roadmap P2).

Compares simple next-set predictors on Issue #8 traces under fixed
per-layer memory budgets. No weights needed; runs on synthetic traces
until online traces land.

Predictor protocol: `update(request_id, token_pos, layer, experts)` feeds
ground truth in causal order; `predict(request_id, token_pos, layer, k)`
returns up to k ranked expert IDs guessed BEFORE the update.

Metrics per (predictor, budget K), averaged over records:
  recall_at_k   mean |actual ∩ predicted[:K]| / top_k
  exact_set     fraction with predicted[:top_k] == actual set
  false_pos     mean |predicted[:K] - actual| (wasted prefetches)
  wasted_bytes  false_pos * expert_bytes
"""

from __future__ import annotations

from collections import OrderedDict

PREDICTORS = (
    "persistence",
    "lru",
    "lfu",
    "ema",
    "token_transition",
    "layer_transition",
    "bundle",
)


class _Base:
    name = "base"

    def update(self, request_id: str, token_pos: int, layer: int, experts: list[int]) -> None:
        raise NotImplementedError

    def predict(self, request_id: str, token_pos: int, layer: int, k: int) -> list[int]:
        raise NotImplementedError

    @staticmethod
    def _top(scores: dict[int, float], k: int) -> list[int]:
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [expert for expert, _ in ranked[:k]]


class Persistence(_Base):
    """Repeat the previous token's set at the same layer/request."""

    name = "persistence"

    def __init__(self) -> None:
        self._last: dict[tuple[str, int], list[int]] = {}

    def update(self, request_id: str, token_pos: int, layer: int, experts: list[int]) -> None:
        self._last[(request_id, layer)] = list(experts)

    def predict(self, request_id: str, token_pos: int, layer: int, k: int) -> list[int]:
        return list(self._last.get((request_id, layer), []))[:k]


class LRU(_Base):
    """Per-layer most-recently-seen experts."""

    name = "lru"

    def __init__(self) -> None:
        self._order: dict[int, OrderedDict[int, None]] = {}

    def update(self, request_id: str, token_pos: int, layer: int, experts: list[int]) -> None:
        bucket = self._order.setdefault(layer, OrderedDict())
        for expert in experts:
            bucket[expert] = None
            bucket.move_to_end(expert)

    def predict(self, request_id: str, token_pos: int, layer: int, k: int) -> list[int]:
        bucket = self._order.get(layer, OrderedDict())
        return list(reversed(bucket))[:k]


class LFU(_Base):
    """Per-layer most-frequent experts."""

    name = "lfu"

    def __init__(self) -> None:
        self._counts: dict[int, dict[int, int]] = {}

    def update(self, request_id: str, token_pos: int, layer: int, experts: list[int]) -> None:
        bucket = self._counts.setdefault(layer, {})
        for expert in experts:
            bucket[expert] = bucket.get(expert, 0) + 1

    def predict(self, request_id: str, token_pos: int, layer: int, k: int) -> list[int]:
        return self._top({e: float(c) for e, c in self._counts.get(layer, {}).items()}, k)


class EMA(_Base):
    """Per-layer exponentially-decayed frequency (alpha per observation)."""

    name = "ema"

    def __init__(self, alpha: float = 0.2) -> None:
        self._alpha = alpha
        self._scores: dict[int, dict[int, float]] = {}

    def update(self, request_id: str, token_pos: int, layer: int, experts: list[int]) -> None:
        bucket = self._scores.setdefault(layer, {})
        for expert in set(bucket) | set(experts):
            prior = bucket.get(expert, 0.0)
            bucket[expert] = (1 - self._alpha) * prior + (1.0 if expert in experts else 0.0)

    def predict(self, request_id: str, token_pos: int, layer: int, k: int) -> list[int]:
        return self._top(self._scores.get(layer, {}), k)


class TokenTransition(_Base):
    """First-order token transition: P(set_t | set_{t-1}) per layer.

    Transition mass from every expert in the previous token's set is
    spread uniformly over the next set, then summed per candidate.
    """

    name = "token_transition"

    def __init__(self) -> None:
        self._trans: dict[int, dict[int, dict[int, float]]] = {}
        self._prev: dict[tuple[str, int], list[int]] = {}

    def update(self, request_id: str, token_pos: int, layer: int, experts: list[int]) -> None:
        key = (request_id, layer)
        prev = self._prev.get(key)
        if prev:
            table = self._trans.setdefault(layer, {})
            share = 1.0 / len(prev)
            for src in prev:
                row = table.setdefault(src, {})
                for dst in experts:
                    row[dst] = row.get(dst, 0.0) + share
        self._prev[key] = list(experts)

    def predict(self, request_id: str, token_pos: int, layer: int, k: int) -> list[int]:
        prev = self._prev.get((request_id, layer), [])
        table = self._trans.get(layer, {})
        scores: dict[int, float] = {}
        for src in prev:
            for dst, mass in table.get(src, {}).items():
                scores[dst] = scores.get(dst, 0.0) + mass
        return self._top(scores, k)


class LayerTransition(_Base):
    """Layer-aware transition: P(set_layer | set_{layer-1}) within a token."""

    name = "layer_transition"

    def __init__(self) -> None:
        self._trans: dict[int, dict[int, dict[int, float]]] = {}
        self._prev_layer: dict[tuple[str, int], list[int]] = {}

    def update(self, request_id: str, token_pos: int, layer: int, experts: list[int]) -> None:
        key = (request_id, token_pos)
        prev = self._prev_layer.get(key)
        if prev is not None and layer > 0:
            table = self._trans.setdefault(layer, {})
            share = 1.0 / len(prev)
            for src in prev:
                row = table.setdefault(src, {})
                for dst in experts:
                    row[dst] = row.get(dst, 0.0) + share
        self._prev_layer[key] = list(experts)

    def predict(self, request_id: str, token_pos: int, layer: int, k: int) -> list[int]:
        if layer == 0:
            return []
        prev = self._prev_layer.get((request_id, token_pos), [])
        table = self._trans.get(layer, {})
        scores: dict[int, float] = {}
        for src in prev:
            for dst, mass in table.get(src, {}).items():
                scores[dst] = scores.get(dst, 0.0) + mass
        return self._top(scores, k)


class Bundle(_Base):
    """Windowed bundle: frequency over the previous W tokens, same layer.

    Union-of-recent-sets with frequency ranking: recalls experts that
    recur within a short horizon even when the single previous token
    differs (covers bursty reuse persistence misses).
    """

    name = "bundle"

    def __init__(self, window: int = 4) -> None:
        self._window = window
        self._history: dict[tuple[str, int], list[list[int]]] = {}

    def update(self, request_id: str, token_pos: int, layer: int, experts: list[int]) -> None:
        key = (request_id, layer)
        hist = self._history.setdefault(key, [])
        hist.append(list(experts))
        del hist[: max(0, len(hist) - self._window)]

    def predict(self, request_id: str, token_pos: int, layer: int, k: int) -> list[int]:
        counts: dict[int, int] = {}
        for seen in self._history.get((request_id, layer), []):
            for expert in seen:
                counts[expert] = counts.get(expert, 0) + 1
        return self._top({e: float(c) for e, c in counts.items()}, k)


def make_predictor(name: str, **kwargs) -> _Base:
    """Instantiate a predictor by roadmap baseline name."""
    kinds: dict[str, type[_Base]] = {
        "persistence": Persistence,
        "lru": LRU,
        "lfu": LFU,
        "ema": EMA,
        "token_transition": TokenTransition,
        "layer_transition": LayerTransition,
        "bundle": Bundle,
    }
    if name not in kinds:
        raise ValueError(f"unknown predictor {name!r}; choose from {sorted(kinds)}")
    return kinds[name](**kwargs)


def evaluate(
    records: list[dict],
    predictor: _Base,
    budget: int,
    top_k: int,
    expert_bytes: int = 0,
) -> dict:
    """Causal simulation: predict each record before observing it."""
    ordered = sorted(records, key=lambda r: (r["request_id"], r["token_pos"], r["layer"]))
    recall_sum = 0.0
    exact = 0
    fp_sum = 0
    n = 0
    for record in ordered:
        actual = record["experts"]
        guessed = predictor.predict(
            record["request_id"], record["token_pos"], record["layer"], budget
        )
        hit = len(set(actual) & set(guessed))
        recall_sum += hit / top_k
        exact += set(guessed[:top_k]) == set(actual)
        fp_sum += len(set(guessed) - set(actual))
        n += 1
        predictor.update(record["request_id"], record["token_pos"], record["layer"], actual)
    return {
        "predictor": predictor.name,
        "budget": budget,
        "records": n,
        "recall_at_k": recall_sum / n if n else 0.0,
        "exact_set": exact / n if n else 0.0,
        "false_pos": fp_sum / n if n else 0.0,
        "wasted_bytes": (fp_sum / n * expert_bytes) if n else 0.0,
    }


def compare(
    records: list[dict],
    budgets: list[int],
    top_k: int,
    expert_bytes: int = 0,
    predictors: tuple[str, ...] = PREDICTORS,
) -> list[dict]:
    """Evaluate every predictor at every budget; fresh state each run."""
    rows = []
    for name in predictors:
        for budget in budgets:
            rows.append(evaluate(records, make_predictor(name), budget, top_k, expert_bytes))
    return rows
