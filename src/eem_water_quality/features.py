"""Training-only spectral and tabular feature estimators."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorly.decomposition import non_negative_parafac, parafac

from .data import EC, SS, TOC


@dataclass(frozen=True)
class Experiment:
    name: str
    eem: str = "none"
    parafac: bool = False
    tabular: tuple = ()


def experiment_catalog():
    experiments = [
        Experiment("TOC", tabular=(TOC,)),
        Experiment("TOC_SS_EC", tabular=(TOC, SS, EC)),
    ]
    for prefix, eem, pf in [
        ("PARAFAC", "none", True),
        ("EEM", "raw", False),
        ("EEM_PARAFAC", "raw", True),
        ("EEMpca", "pca", False),
        ("EEMpca_PARAFAC", "pca", True),
    ]:
        for suffix, tab in [
            ("", ()),
            ("_TOC", (TOC,)),
            ("_SS_EC", (SS, EC)),
            ("_TOC_SS_EC", (TOC, SS, EC)),
        ]:
            experiments.append(Experiment(prefix + suffix, eem, pf, tab))
    return {exp.name: exp for exp in experiments}


class ParafacFeatures:
    def __init__(self, rank=4, nonnegative=False, max_iter=1000, tol=1e-7, seed=42):
        self.rank = rank
        self.nonnegative = nonnegative
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def _prepare(self, eem):
        x = np.asarray(eem, dtype=np.float64)
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError("PARAFAC requires finite 3D EEM data")
        return np.clip(x, 0, None) if self.nonnegative else x

    def fit_transform(self, eem):
        x = self._prepare(eem)
        if not 1 <= self.rank <= min(x.shape):
            raise ValueError("PARAFAC rank must be between 1 and the smallest tensor dimension")
        fit = non_negative_parafac if self.nonnegative else parafac
        self.cp_model_ = fit(
            x,
            rank=self.rank,
            init="svd",
            n_iter_max=self.max_iter,
            tol=self.tol,
            random_state=self.seed,
            normalize_factors=False,
        )
        weights, (scores, self.ex_loadings_, self.em_loadings_) = self.cp_model_
        self.basis_ = np.column_stack(
            [
                np.outer(self.ex_loadings_[:, r], self.em_loadings_[:, r]).ravel()
                for r in range(self.rank)
            ]
        )
        self.shape_ = x.shape[1:]
        scores = np.asarray(scores) * (1 if weights is None else np.asarray(weights)[None, :])
        self.train_relative_error_ = self.relative_error(x, scores)
        return scores

    def transform(self, eem):
        x = self._prepare(eem)
        if x.shape[1:] != self.shape_:
            raise ValueError("EEM wavelength dimensions differ from fitted PARAFAC")
        flat = x.reshape(len(x), -1)
        if self.nonnegative:
            return np.asarray([nnls(self.basis_, row)[0] for row in flat])
        return np.linalg.lstsq(self.basis_, flat.T, rcond=None)[0].T

    def relative_error(self, eem, scores):
        flat = self._prepare(eem).reshape(len(eem), -1)
        norm = np.linalg.norm(flat)
        return float(np.linalg.norm(flat - scores @ self.basis_.T) / norm) if norm else 0.0


def numeric_tabular(samples, columns):
    import pandas as pd

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


class FeatureBuilder:
    def __init__(
        self,
        experiment,
        target,
        pca_components=30,
        pf_rank=4,
        pf_nonnegative=False,
        pf_max_iter=1000,
        seed=42,
    ):
        self.experiment = experiment
        self.target = target
        self.pca_components = pca_components
        self.pf_rank = pf_rank
        self.pf_nonnegative = pf_nonnegative
        self.pf_max_iter = pf_max_iter
        self.seed = seed
        self.tabular_columns = tuple(c for c in experiment.tabular if c != target)

    def fit_transform(self, eem, samples, parafac_cache=None):
        parts = []
        if self.experiment.eem != "none":
            flat = eem.reshape(len(eem), -1).astype(np.float64)
            if self.experiment.eem == "pca":
                rank = min(self.pca_components, *flat.shape)
                self.eem_pipeline_ = make_pipeline(
                    StandardScaler(), PCA(n_components=rank, random_state=self.seed)
                )
                flat = self.eem_pipeline_.fit_transform(flat)
            parts.append(flat)
        if self.experiment.parafac:
            if parafac_cache is None:
                self.parafac_ = ParafacFeatures(
                    self.pf_rank, self.pf_nonnegative, self.pf_max_iter, seed=self.seed
                )
                scores = self.parafac_.fit_transform(eem)
            else:
                self.parafac_, scores = parafac_cache
            self.pf_scaler_ = StandardScaler()
            parts.append(self.pf_scaler_.fit_transform(scores))
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
            flat = eem.reshape(len(eem), -1).astype(np.float64)
            if self.experiment.eem == "pca":
                flat = self.eem_pipeline_.transform(flat)
            parts.append(flat)
        if self.experiment.parafac:
            parts.append(self.pf_scaler_.transform(self.parafac_.transform(eem)))
        if self.tabular_columns:
            parts.append(
                self.tab_pipeline_.transform(numeric_tabular(samples, self.tabular_columns))
            )
        return np.hstack(parts)
