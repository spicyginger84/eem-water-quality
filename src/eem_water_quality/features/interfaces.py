"""Small interfaces shared by feature transformers."""

from typing import Protocol


class FeatureTransformer(Protocol):
    """Train-only feature transformer contract."""

    def fit_transform(self, eem, samples): ...

    def transform(self, eem, samples): ...

