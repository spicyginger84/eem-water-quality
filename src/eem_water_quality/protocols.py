"""Outer splits and few-shot calibration designs.

Outer schemes:
  random                 5-fold KFold over samples (what most papers report)
  station                frozen 5-fold station split (samples.csv station_cv_fold)
  station_gkf5_seed*     alternative 5-fold station splits stored in samples.csv
  loso                   leave-one-station-out
Few-shot (station schemes only): a station-level log offset on top of the zero-shot
prediction, Bayes-shrunk: b = sum_support (log y - log yhat) / (n_support + prior).
Support designs, by date order within each station:
  early     the station's first K samples; query = the rest
  rolling   for each sample, the K labelled samples just before it (one step ahead);
            query = samples with at least K earlier samples
  random    K random samples (uses future labels; reference only)
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from .data import Dataset

SEED = 11


def outer_splits(D: Dataset, scheme: str):
    if scheme == "random":
        return list(KFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(D))))
    labels = D.station if scheme == "loso" else D.folds[scheme]
    return [(np.flatnonzero(labels != f), np.flatnonzero(labels == f)) for f in np.unique(labels)]


def is_station_scheme(scheme: str) -> bool:
    return scheme != "random"


def calibrate(D: Dataset, log_pred: np.ndarray, design: str, k=3, prior=3.0):
    """Few-shot log predictions and the query mask for one support design."""
    ly = np.log(D.y)
    rank = D.station_rank
    out = log_pred.copy()
    if design == "rolling":
        query = rank >= k
        order = {}
        for s in np.unique(D.station):
            ids = np.flatnonzero(D.station == s)
            order[s] = ids[np.argsort(rank[ids])]
        for i in np.flatnonzero(query):
            sup = order[D.station[i]][rank[i] - k:rank[i]]
            out[i] = log_pred[i] + (ly[sup] - log_pred[sup]).sum(0) / (k + prior)
        return out, query
    if design == "early":
        support = rank < k
    elif design == "random":
        rng = np.random.default_rng(SEED)
        support = np.zeros(len(D), bool)
        for s in np.unique(D.station):
            ids = np.flatnonzero(D.station == s)
            support[rng.choice(ids, min(k, len(ids)), replace=False)] = True
    else:
        raise ValueError(f"Unknown support design {design!r}")
    for s in np.unique(D.station):
        sup = np.flatnonzero(support & (D.station == s))
        rows = D.station == s
        out[rows] = log_pred[rows] + (ly[sup] - log_pred[sup]).sum(0) / (len(sup) + prior)
    return out, ~support
