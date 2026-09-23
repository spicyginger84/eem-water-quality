"""Leakage-safe global and group mean baselines."""

import numpy as np
import pandas as pd


def baseline_predictions(samples, y, fit_indices, eval_indices, group_col="Point"):
    """Return global and group means fitted only on ``fit_indices``."""
    global_mean = float(np.mean(y[fit_indices]))
    global_prediction = np.full(len(eval_indices), global_mean, dtype=float)
    if group_col not in samples:
        return global_prediction, global_prediction.copy()
    grouped = pd.DataFrame(
        {"group": samples.iloc[fit_indices][group_col].to_numpy(), "target": y[fit_indices]}
    )
    group_means = grouped.groupby("group")["target"].mean()
    eval_groups = pd.Series(samples.iloc[eval_indices][group_col].to_numpy())
    group_prediction = eval_groups.map(group_means).fillna(global_mean).to_numpy(dtype=float)
    return global_prediction, group_prediction

