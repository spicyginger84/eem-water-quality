"""Leakage-safe splitters used by the classical benchmark pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut


def _validate_fold_count(n_samples: int, n_folds: int) -> None:
    if n_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if n_samples < n_folds:
        raise ValueError("cv_folds cannot exceed the number of samples")


def random_cv_splits(n_samples: int, n_folds: int = 5, seed: int = 42):
    """Return shuffled sample-level KFold splits as absolute indices."""
    _validate_fold_count(n_samples, n_folds)
    indices = np.arange(n_samples)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        {"fold": fold, "train": indices[train], "test": indices[test]}
        for fold, (train, test) in enumerate(splitter.split(indices), start=1)
    ]


def grouped_cv_splits(
    samples: pd.DataFrame,
    group_col: str = "Point",
    n_folds: int = 5,
    seed: int = 42,
):
    """Return shuffled GroupKFold splits, keeping each group in one fold."""
    if group_col not in samples or samples[group_col].isna().any():
        raise ValueError(f"Grouped CV requires a nonmissing {group_col!r} column")
    groups = samples[group_col].to_numpy()
    unique_groups = np.unique(groups)
    _validate_fold_count(len(unique_groups), n_folds)
    indices = np.arange(len(samples))
    # ``shuffle`` was added to GroupKFold after the minimum sklearn version
    # supported by this repository.  Keep the deterministic no-shuffle fallback
    # so the protocol remains usable with sklearn 1.4/1.5 as well.
    try:
        splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    except TypeError:  # pragma: no cover - exercised only on older sklearn
        splitter = GroupKFold(n_splits=n_folds)
    return [
        {
            "fold": fold,
            "train": indices[train],
            "test": indices[test],
            "heldout_group": tuple(pd.unique(groups[test]).tolist()),
        }
        for fold, (train, test) in enumerate(splitter.split(indices, groups=groups), start=1)
    ]


def leave_one_group_out_splits(samples: pd.DataFrame, group_col: str):
    """Return deterministic leave-one-group-out splits."""
    if group_col not in samples or samples[group_col].isna().any():
        raise ValueError(f"Leave-one-group-out requires a nonmissing {group_col!r} column")
    indices = np.arange(len(samples))
    groups = samples[group_col].to_numpy()
    splitter = LeaveOneGroupOut()
    result = []
    for fold, (train, test) in enumerate(splitter.split(indices, groups=groups), start=1):
        heldout = pd.unique(groups[test]).tolist()
        result.append(
            {
                "fold": fold,
                "train": indices[train],
                "test": indices[test],
                "heldout_group": heldout[0] if len(heldout) == 1 else tuple(heldout),
            }
        )
    return result
