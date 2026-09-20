"""Load aligned processed arrays; all indexing downstream is positional."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

BOD = "BOD\n(0.0)"
COD = "COD\n(0.0)"
TOC = "TOC\n(0.0)"
SS = "SS\n(0.0)"
EC = "EC\n(0)"
ALIASES = {"BOD": BOD, "COD": COD, "TOC": TOC, "SS": SS, "EC": EC, "BOD_COD": "BOD_COD"}


def resolve_column(name):
    return ALIASES.get(name, name)


def load_data(directory):
    directory = Path(directory)
    eem = np.load(directory / "eem.npy", allow_pickle=False).astype(np.float32)
    samples = pd.read_parquet(directory / "samples.parquet").reset_index(drop=True)
    if eem.ndim != 3 or len(eem) != len(samples):
        raise ValueError("Expected aligned EEM (samples, excitation, emission) and metadata rows")
    if not np.isfinite(eem).all():
        raise ValueError(
            "EEM contains NaN/Inf; repair the processed data before running experiments"
        )
    if BOD in samples and COD in samples:
        denominator = pd.to_numeric(samples[COD], errors="coerce").replace(0, np.nan)
        samples["BOD_COD"] = pd.to_numeric(samples[BOD], errors="coerce") / denominator
    wavelengths = {}
    path = directory / "wavelengths.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            wavelengths = {key: archive[key] for key in archive.files}
    return eem, samples, wavelengths


def split_indices(samples, mode="group", group_col="Point", seed=42, test_size=0.2, val_size=0.2):
    """Split once before target filtering. val_size is a fraction of non-test rows/groups."""
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
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return {"train": train, "validation": val, "test": test}


def target_partitions(samples, target, splits):
    y = pd.to_numeric(samples[target], errors="coerce").to_numpy(dtype=float)
    parts = {name: indices[np.isfinite(y[indices])] for name, indices in splits.items()}
    for name, indices in parts.items():
        if len(indices) < 2:
            raise ValueError(f"{target!r}: {name} needs at least two finite targets")
    return y, parts
