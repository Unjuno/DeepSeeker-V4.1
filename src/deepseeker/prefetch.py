"""Issue #25: gated predictive prefetch over staging + pool (roadmap P4).

Wires an Issue #9 predictor to the Issue #24 staging engine feeding the
Issue #23 pool. The authoritative router (here: trace records) is never
modified: every serve goes through pool lookup with demand fallback, so
outputs are byte-identical with prefetch ON or OFF.

Gate: prefetch submits only while enabled AND the rolling agreement
(predicted set == actual set, trailing window) clears min_agreement.
A distribution shift auto-disables back to demand-only. A NO-GO verdict
is a first-class outcome (the issue's decision rule).
"""

from __future__ import annotations

from deepseeker.simulate import make_predictor
from deepseeker.stage import StagingError

SCHEMA = "deepseeker.prefetch/v1"


class PrefetchController:
    """Lookahead prefetch with gate, fallback, and waste accounting."""

    def __init__(
        self,
        pool,
        engine,
        predictor_name: str = "persistence",
        budget: int = 6,
        horizon: int = 1,
        enabled: bool = False,
        min_agreement: float = 0.5,
        window: int = 20,
        n_layers: int = 40,
    ) -> None:
        self._pool = pool
        self._engine = engine
        self._predictor = make_predictor(predictor_name)
        self._budget = budget
        self._horizon = horizon
        self.enabled = enabled
        self._min_agreement = min_agreement
        self._window = window
        self._n_layers = n_layers
        self._history: list[tuple[str, int]] = []  # (request, token) fully observed
        self._records: dict[tuple[str, int], list[dict]] = {}
        self._agreements: list[float] = []  # per-token set-match rate
        self._prefetched: dict[tuple[int, int], int] = {}  # (layer,expert) -> token idx
        self._outstanding: list = []
        self._wasted_bytes = 0
        self._served_prefetched = 0
        self.auto_disabled = False
        self.prefetch_submits = 0

    def observe_token(self, records: list[dict]) -> None:
        """Score pre-update predictions vs actuals, then absorb the token.

        Agreement compares what the predictor said BEFORE seeing this
        token (causal); only then is the predictor updated. A trailing
        window below min_agreement auto-disables to demand-only.
        """
        key = (records[0]["request_id"], records[0]["token_pos"])
        hits = 0.0
        for record in records:
            guessed = self._predictor.predict(
                record["request_id"], record["token_pos"], record["layer"], self._budget
            )
            actual = record["experts"]
            hits += len(set(actual) & set(guessed)) / len(actual) if actual else 0.0
        self._agreements.append(hits / len(records) if records else 0.0)
        for record in records:
            self._predictor.update(
                record["request_id"], record["token_pos"], record["layer"], record["experts"]
            )
        self._records[key] = records
        self._history.append(key)
        if (
            self.enabled
            and len(self._agreements) >= self._window
            and sum(self._agreements[-self._window:]) / self._window < self._min_agreement
        ):
            self.enabled = False
            self.auto_disabled = True

    @staticmethod
    def _coords(ticket) -> tuple[int, int]:
        if hasattr(ticket, "layer"):
            return ticket.layer, ticket.expert
        return tuple(ticket)

    def drain(self) -> int:
        """Admit completed staged fetches without blocking; returns count."""
        admitted = 0
        pending = []
        for ticket in self._outstanding:
            try:
                blob = self._engine.wait(ticket, timeout=0)
            except TimeoutError:
                pending.append(ticket)
                continue
            except (KeyError, StagingError):
                continue
            layer, expert = self._coords(ticket)
            try:
                self._pool.admit(layer, expert, blob)
                admitted += 1
            except (RuntimeError, ValueError):
                continue
        self._outstanding = pending
        return admitted

    def prefetch_ahead(self) -> int:
        """Submit predictions for the upcoming horizon; returns submits."""
        if not self.enabled or self.auto_disabled:
            return 0
        if not self._history:
            return 0
        self.drain()
        request_id, token_pos = self._history[-1]
        target = token_pos + self._horizon
        submitted = 0
        current_idx = len(self._history)
        for layer in range(self._n_layers):
            guessed = self._predictor.predict(request_id, target, layer, self._budget)
            for expert in guessed:
                if self._pool.contains(layer, expert):
                    continue
                try:
                    ticket = self._engine.submit(layer, expert)
                except RuntimeError:
                    continue
                self._outstanding.append(ticket)
                self._prefetched[(layer, expert)] = current_idx
                submitted += 1
        self.prefetch_submits += submitted
        return submitted

    def serve(self, layer: int, expert: int, loader, expert_bytes: int = 0) -> bytes:
        """Serve one expert use: resident hit, staged hit, or demand fallback."""
        hit = self._pool.lookup(layer, expert)
        if hit is not None:
            if (layer, expert) in self._prefetched:
                self._served_prefetched += 1
                del self._prefetched[(layer, expert)]
            return bytes(hit)
        ticket = None
        try:
            ticket = self._engine.submit(layer, expert)
            blob = self._engine.wait(ticket)
            self._pool.admit(layer, expert, blob)
        except (RuntimeError, KeyError, TimeoutError, StagingError):
            blob = loader(layer, expert)
            self._pool.admit(layer, expert, blob)
        return bytes(blob)

    def collect_waste(self, current_idx: int, horizon_grace: int = 4) -> int:
        """Charge staged-but-unused predictions older than grace tokens."""
        stale = [
            key for key, idx in self._prefetched.items() if current_idx - idx > horizon_grace
        ]
        return len(stale)

    def report(self, expert_bytes: int = 0) -> dict:
        return {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "auto_disabled": self.auto_disabled,
            "prefetch_submits": self.prefetch_submits,
            "served_prefetched": self._served_prefetched,
            "stale_prefetched": len(self._prefetched),
            "wasted_bytes": len(self._prefetched) * expert_bytes,
            "trailing_agreement": (
                sum(self._agreements[-self._window:]) / min(len(self._agreements), self._window)
                if self._agreements else None
            ),
        }
