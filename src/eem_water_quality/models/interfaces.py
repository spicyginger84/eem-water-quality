"""Model interface used by classical benchmark candidates."""

from typing import Protocol


class Regressor(Protocol):
    """Minimal estimator contract required by the evaluator."""

    def fit(self, x, y): ...

    def predict(self, x): ...

