"""Regression metrics in mg/L and the station-paired bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y, p, station, brackish=None):
    e = p - y
    out = {
        "n": len(y),
        "r2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
        "rmse": np.sqrt(np.mean(e ** 2)),
        "mae": np.mean(np.abs(e)),
        "station_mae": pd.Series(np.abs(e)).groupby(station).mean().mean(),
        "mae_top10": np.mean(np.abs(e[y >= np.quantile(y, 0.9)])),
    }
    if brackish is not None and brackish.any():
        out["mae_brackish"] = np.mean(np.abs(e[brackish]))
        out["mae_fresh"] = np.mean(np.abs(e[~brackish]))
    return out


def paired_station_bootstrap(y, pa, pb, station, n=2000, seed=1701):
    """Station-MAE(b) - station-MAE(a) with a 95% station-bootstrap interval."""
    st = np.unique(station)
    da = pd.Series(np.abs(pa - y)).groupby(station).mean().loc[st].to_numpy()
    db = pd.Series(np.abs(pb - y)).groupby(station).mean().loc[st].to_numpy()
    d = db - da
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi
