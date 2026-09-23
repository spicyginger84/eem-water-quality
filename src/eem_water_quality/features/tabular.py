"""Tabular feature extraction and train-only scaling helpers."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def numeric_tabular(samples, columns):
    return (
        samples[list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=float)
    )


def tabular_scaler():
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    )

__all__ = ["numeric_tabular", "tabular_scaler"]
