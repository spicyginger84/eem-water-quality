"""Evaluation protocols for classical regression benchmarks."""

from .baselines import baseline_predictions
from .cross_validation import grouped_cv_splits, leave_one_group_out_splits, random_cv_splits
from .metrics import regression_metrics
from .protocols import build_evaluation_splits, resolve_protocol

__all__ = [
    "baseline_predictions",
    "build_evaluation_splits",
    "grouped_cv_splits",
    "leave_one_group_out_splits",
    "random_cv_splits",
    "regression_metrics",
    "resolve_protocol",
]
