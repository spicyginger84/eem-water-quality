"""PCA representation facade.

The fold-aware implementation remains in :class:`FeatureBuilder`; this module
provides a stable name for future standalone transformers.
"""

from .builder import FeatureBuilder

__all__ = ["FeatureBuilder"]

