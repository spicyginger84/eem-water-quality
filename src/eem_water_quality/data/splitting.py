"""Random, grouped and two-way split utilities."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


def split_indices(
    samples,
    mode="group",
    group_col="Point",
    seed=42,
    test_size=0.2,
    val_size=0.2,
    secondary_group_col="Month",
):
    """Create one train/validation/test split before target filtering."""
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test_size and val_size must lie strictly between 0 and 1")
    indices = np.arange(len(samples))
    if mode == "group":
        if group_col not in samples or samples[group_col].isna().any():
            raise ValueError(f"Grouped splitting requires a nonmissing {group_col!r} column")
        groups = samples[group_col].to_numpy()
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        rest, test = next(outer.split(indices, groups=groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_rel, val_rel = next(inner.split(rest, groups=groups[rest]))
        train, val = rest[train_rel], rest[val_rel]
    elif mode == "random":
        rest, test = train_test_split(indices, test_size=test_size, random_state=seed)
        train, val = train_test_split(rest, test_size=val_size, random_state=seed)
    elif mode == "two_way":
        for column in (group_col, secondary_group_col):
            if column not in samples or samples[column].isna().any():
                raise ValueError(f"Two-way splitting requires a nonmissing {column!r} column")
        station_values = samples[group_col].drop_duplicates().to_numpy()
        month_values = samples[secondary_group_col].drop_duplicates().to_numpy()
        station_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, held_station_rel = next(station_split.split(station_values, groups=station_values))
        month_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + 1)
        _, held_month_rel = next(month_split.split(month_values, groups=month_values))
        held_stations = station_values[held_station_rel]
        held_months = month_values[held_month_rel]
        station = samples[group_col].to_numpy()
        month = samples[secondary_group_col].to_numpy()
        test_mask = np.isin(station, held_stations) & np.isin(month, held_months)
        eligible_mask = ~np.isin(station, held_stations) & ~np.isin(month, held_months)
        test = indices[test_mask]
        eligible = indices[eligible_mask]
        if len(test) < 2 or len(eligible) < 4:
            raise ValueError("Two-way holdout leaves too few train/test samples")
        inner = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_rel, val_rel = next(
            inner.split(eligible, groups=samples.iloc[eligible][group_col].to_numpy())
        )
        train, val = eligible[train_rel], eligible[val_rel]
        used = np.concatenate([train, val, test])
        excluded = np.setdiff1d(indices, used, assume_unique=False)
        return {"train": train, "validation": val, "test": test, "excluded": excluded}
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return {"train": train, "validation": val, "test": test}


def group_kfold_indices(samples, indices, n_splits=5, group_col="Point"):
    """Yield absolute train/validation indices for compatibility callers."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    indices = np.asarray(indices, dtype=int)
    if group_col not in samples:
        raise ValueError(f"Grouped CV requires a {group_col!r} column")
    groups = samples.iloc[indices][group_col].to_numpy()
    if pd.isna(groups).any() or np.unique(groups).size < n_splits:
        raise ValueError(f"Need at least {n_splits} nonmissing groups for grouped CV")
    splitter = GroupKFold(n_splits=n_splits)
    for train_rel, validation_rel in splitter.split(indices, groups=groups):
        yield indices[train_rel], indices[validation_rel]


def target_partitions(samples, target, splits):
    """Remove nonfinite target rows from each split."""
    y = pd.to_numeric(samples[target], errors="coerce").to_numpy(dtype=float)
    parts = {name: indices[np.isfinite(y[indices])] for name, indices in splits.items()}
    for name, indices in parts.items():
        if name != "excluded" and len(indices) < 2:
            raise ValueError(f"{target!r}: {name} needs at least two finite targets")
    return y, parts

