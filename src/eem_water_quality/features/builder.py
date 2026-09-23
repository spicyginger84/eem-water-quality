"""Training-only spectral and tabular feature estimators."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .tabular import numeric_tabular, tabular_scaler


class FeatureBuilder:
    def __init__(
        self,
        experiment,
        target,
        pca_components=30,
        seed=42,
    ):
        self.experiment = experiment
        self.target = target
        self.pca_components = pca_components
        self.seed = seed
        self.tabular_columns = tuple(c for c in experiment.tabular if c != target)

    def fit_transform(self, eem, samples):
        parts = []
        if self.experiment.eem != "none":
            flat = eem.reshape(len(eem), -1).astype(np.float64)
            self.eem_shape_ = tuple(eem.shape[1:])
            # The processed EEM grid contains a fixed Rayleigh/scatter mask:
            # columns that are zero for every training sample carry no signal
            # and must not be allowed to dominate scaling or PCA.
            self.eem_feature_mask_ = np.any(np.abs(flat) > 0, axis=0)
            if not np.any(self.eem_feature_mask_):
                raise ValueError("EEM contains no nonzero training features")
            flat = flat[:, self.eem_feature_mask_]
            if self.experiment.eem == "pca":
                rank = min(self.pca_components, *flat.shape)
                self.eem_pipeline_ = make_pipeline(
                    StandardScaler(), PCA(n_components=rank, random_state=self.seed)
                )
                flat = self.eem_pipeline_.fit_transform(flat)
            parts.append(flat)
        if self.tabular_columns:
            self.tab_pipeline_ = tabular_scaler()
            parts.append(
                self.tab_pipeline_.fit_transform(numeric_tabular(samples, self.tabular_columns))
            )
        if not parts:
            raise ValueError("No features remain after removing the target column")
        return np.hstack(parts)

    def transform(self, eem, samples):
        parts = []
        if self.experiment.eem != "none":
            if tuple(eem.shape[1:]) != self.eem_shape_:
                raise ValueError("EEM wavelength dimensions differ from fitted features")
            flat = eem.reshape(len(eem), -1).astype(np.float64)
            flat = flat[:, self.eem_feature_mask_]
            if self.experiment.eem == "pca":
                flat = self.eem_pipeline_.transform(flat)
            parts.append(flat)
        if self.tabular_columns:
            parts.append(
                self.tab_pipeline_.transform(numeric_tabular(samples, self.tabular_columns))
            )
        return np.hstack(parts)
