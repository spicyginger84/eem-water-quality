"""Benchmark models. Interface: est.fit(Z, ly, groups) -> est; est.predict(Z) -> log(y).

`groups` (station per train row, or None) drives the inner CV that picks
hyperparameters; it mirrors the outer protocol. Nonlinear models fit log(y). The linear
chemometric models (mlr, upls, npls) fit y in mg/L, as in the literature: a linear map
from intensity to log(y) becomes exponential after back-transform and explodes on
extrapolated samples. Predictions are clipped to [train min / 2, train max * 2].

Literature baselines (input = the feature blocks in MODEL_SPECS):
  mlr        ordinary least squares (Hur & Cho 2012; Yang et al. 2015)
  rf         random forest, 500 trees (Zhang et al. 2025)
  xgb        XGBoost, 500 trees, depth 4, lr 0.05
  upls       PLS1 (NIPALS) on the unfolded EEM, 1-15 components (Peer et al. 2022)
  npls       trilinear PLS1 (Bro 1996), 1-15 components
  pca_svr    20 PCA scores of the unfolded EEM -> RBF SVR
Kernel models on the tabular blocks:
  svr        RBF SVR, C by inner CV
  gard_svr   group-ARD SVR: per feature-group weight w_g in {0, .5, 1, 2} inside the
             kernel, k = exp(-gamma sum_g w_g^2 ||x_g - x'_g||^2), weights and C by
             coordinate descent on inner CV
  gard_matern  as gard_svr with a Matern-5/2 kernel and a length-scale multiplier
Tree models on the tabular blocks (log target, fixed settings, no inner CV):
  lgbm       LightGBM (Ke et al. 2017), 500 trees, 15 leaves, lr 0.03
  lgbm_oxy   lgbm + the oxygen block (DO deficit, DO saturation); best zero-shot BOD model
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEED = 11
MAX_COMPONENTS = 15
C_GRID = (1.0, 3.0, 10.0)
GROUP_LEVELS = (0.0, 0.5, 1.0, 2.0)


def inner_splits(n, groups, k=3):
    if groups is not None and len(np.unique(groups)) >= k:
        return list(GroupKFold(k).split(np.zeros(n), groups=groups))
    return list(KFold(k, shuffle=True, random_state=SEED).split(np.zeros(n)))


class Estimator:
    """Base: target scale (log or mg/L), clipping to the train label range, inner-CV helper."""

    raw = False

    def fit(self, Z, ly, groups=None):
        self.lo, self.hi = ly.min() - np.log(2), ly.max() + np.log(2)
        self._fit(Z, np.exp(ly) if self.raw else ly, groups)
        return self

    def predict(self, Z):
        p = self._predict(Z)
        if self.raw:
            return np.log(np.clip(p, np.exp(self.lo), np.exp(self.hi)))
        return np.clip(p, self.lo, self.hi)

    @staticmethod
    def cv_error(make, Z, ly, splits):
        return np.mean([np.mean(np.abs(make().fit(Z[a], ly[a]).predict(Z[b]) - ly[b])) for a, b in splits])


class Sklearn(Estimator):
    def __init__(self, make, raw=False):
        self.make, self.raw = make, raw

    def _fit(self, Z, ly, groups):
        self.model = self.make().fit(Z, ly)

    def _predict(self, Z):
        return self.model.predict(Z)


def mlr():
    return Sklearn(LinearRegression, raw=True)


def rf():
    return Sklearn(lambda: RandomForestRegressor(500, random_state=SEED, n_jobs=1))


def xgb():
    from xgboost import XGBRegressor

    return Sklearn(lambda: XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                                        colsample_bytree=0.8, random_state=SEED, n_jobs=1))


def lgbm():
    from lightgbm import LGBMRegressor

    return Sklearn(lambda: LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                         random_state=SEED, n_jobs=1, verbose=-1))


class GridSVR(Estimator):
    """Standardise -> [PCA] -> RBF SVR; C chosen on inner CV."""

    def __init__(self, pca=None):
        self.pca = pca

    def make(self, C, rows):
        pca = [PCA(min(self.pca, rows - 1), random_state=SEED)] if self.pca else []
        return make_pipeline(StandardScaler(), *pca, SVR(C=C, gamma="scale", epsilon=0.1))

    def _fit(self, Z, ly, groups):
        splits = inner_splits(len(Z), groups)
        rows = min(len(a) for a, _ in splits)
        C = min(C_GRID, key=lambda c: self.cv_error(lambda: self.make(c, rows), Z, ly, splits))
        self.model = self.make(C, rows).fit(Z, ly)

    def _predict(self, Z):
        return self.model.predict(Z)


class PLS1(Estimator):
    """NIPALS PLS1 on centred data; staged predictions give every component count at once."""

    raw = True

    @staticmethod
    def nipals(X, y, a_max):
        X, y = X.copy(), y.copy()
        W, P, q = [], [], []
        for _ in range(min(a_max, X.shape[1], len(X) - 1)):
            w = X.T @ y
            w /= np.linalg.norm(w) + 1e-12
            t = X @ w
            tt = t @ t + 1e-12
            p, c = X.T @ t / tt, (y @ t) / tt
            X -= np.outer(t, p)
            y -= c * t
            W.append(w), P.append(p), q.append(c)
        W, P, q = np.array(W).T, np.array(P).T, np.array(q)
        # regression vector for the first a components: W_a (P_a' W_a)^-1 q_a
        return [W[:, :a] @ np.linalg.solve(P[:, :a].T @ W[:, :a], q[:a]) for a in range(1, len(q) + 1)]

    def _fit(self, Z, ly, groups):
        errors = []
        for a, b in inner_splits(len(Z), groups):
            mx, my = Z[a].mean(0), ly[a].mean()
            coefs = self.nipals(Z[a] - mx, ly[a] - my, MAX_COMPONENTS)
            errors.append([np.mean(np.abs((Z[b] - mx) @ c + my - ly[b])) for c in coefs])
        n = min(len(e) for e in errors)
        self.components = int(np.argmin(np.mean([e[:n] for e in errors], axis=0))) + 1
        self.mx, self.my = Z.mean(0), ly.mean()
        self.coef = self.nipals(Z - self.mx, ly - self.my, self.components)[-1]

    def _predict(self, Z):
        return (Z - self.mx) @ self.coef + self.my


class NPLS1(Estimator):
    """Trilinear PLS1 (Bro 1996) on an n x J x K cube."""

    raw = True

    @staticmethod
    def components(X, y, a_max):
        Xr, yr, Ws, T = X.copy(), y.copy(), [], []
        for _ in range(a_max):
            u, _, vt = np.linalg.svd(np.einsum("i,ijk->jk", yr, Xr), full_matrices=False)
            wj, wk = u[:, 0], vt[0]
            t = np.einsum("ijk,j,k->i", Xr, wj, wk)
            Xr -= t[:, None, None] * np.outer(wj, wk)[None]
            T.append(t), Ws.append((wj, wk))
            Tm = np.column_stack(T)
            yr = y - Tm @ np.linalg.lstsq(Tm, y, rcond=None)[0]
        return Ws, np.column_stack(T)

    @staticmethod
    def scores(X, Ws):
        Xr, T = X.copy(), []
        for wj, wk in Ws:
            t = np.einsum("ijk,j,k->i", Xr, wj, wk)
            Xr -= t[:, None, None] * np.outer(wj, wk)[None]
            T.append(t)
        return np.column_stack(T)

    def _fit(self, Z, ly, groups):
        errors = []
        for a, b in inner_splits(len(Z), groups):
            mx, my = Z[a].mean(0), ly[a].mean()
            Ws, T = self.components(Z[a] - mx, ly[a] - my, MAX_COMPONENTS)
            Tb = self.scores(Z[b] - mx, Ws)
            errors.append([np.mean(np.abs(Tb[:, :c] @ np.linalg.lstsq(T[:, :c], ly[a] - my, rcond=None)[0] + my - ly[b]))
                           for c in range(1, MAX_COMPONENTS + 1)])
        self.n = int(np.argmin(np.mean(errors, axis=0))) + 1
        self.mx, self.my = Z.mean(0), ly.mean()
        self.Ws, T = self.components(Z - self.mx, ly - self.my, self.n)
        self.b = np.linalg.lstsq(T, ly - self.my, rcond=None)[0]

    def _predict(self, Z):
        return self.scores(Z - self.mx, self.Ws) @ self.b + self.my


def rbf(A, B, scale):
    return np.exp(-cdist(A, B, "sqeuclidean") / scale)


def matern52(A, B, scale):
    r = np.sqrt(5.0) * cdist(A, B) / scale
    return (1 + r + r * r / 3) * np.exp(-r)


class GroupARDSVR(Estimator):
    """SVR on a precomputed kernel with per-feature-group weights chosen on inner CV."""

    def __init__(self, groups_of_columns, kernel="rbf", multipliers=(1.0,)):
        self.blocks, self.kernel, self.multipliers = groups_of_columns, kernel, multipliers

    def _kernel(self, A, B, m):
        d = A.shape[1]
        if self.kernel == "rbf":
            return rbf(A, B, d * m)
        return matern52(A, B, np.sqrt(d) * m)

    def _weighted(self, Z, w):
        s = np.ones(Z.shape[1])
        for cols, wg in zip(self.blocks, w):
            s[cols] = wg
        return Z * s

    def _fit(self, Z, ly, groups):
        self.scaler = StandardScaler().fit(Z)
        S = self.scaler.transform(Z)
        splits = inner_splits(len(S), groups)

        def err(w, C, m):
            Sw = self._weighted(S, w)
            return np.mean([np.mean(np.abs(
                SVR(kernel="precomputed", C=C, epsilon=0.1).fit(self._kernel(Sw[a], Sw[a], m), ly[a])
                .predict(self._kernel(Sw[b], Sw[a], m)) - ly[b])) for a, b in splits])

        w, C, m = [1.0] * len(self.blocks), 3.0, self.multipliers[len(self.multipliers) // 2]
        best = err(w, C, m)
        for _ in range(2):
            for g in range(len(self.blocks)):
                for v in GROUP_LEVELS:
                    cand = w[:g] + [v] + w[g + 1:]
                    if any(cand):
                        e = err(cand, C, m)
                        if e < best:
                            best, w = e, cand
            for c in C_GRID:
                for mm in self.multipliers:
                    e = err(w, c, mm)
                    if e < best:
                        best, C, m = e, c, mm
        self.w, self.C, self.m = w, C, m
        self.train = self._weighted(S, w)
        self.model = SVR(kernel="precomputed", C=C, epsilon=0.1).fit(self._kernel(self.train, self.train, m), ly)

    def _predict(self, Z):
        S = self._weighted(self.scaler.transform(Z), self.w)
        return self.model.predict(self._kernel(S, self.train, self.m))


# name: (feature blocks or input kind, estimator factory taking the column groups)
TABULAR = ["peaks", "fri", "field", "season"]
MODEL_SPECS = {
    "peaks_mlr": (["peaks"], lambda g: mlr()),
    "fri_mlr": (["fri"], lambda g: mlr()),
    "parafac_mlr": (["parafac"], lambda g: mlr()),
    "parafac_rf": (["parafac"], lambda g: rf()),
    "parafac_xgb": (["parafac"], lambda g: xgb()),
    "upls": ("unfolded", lambda g: PLS1()),
    "npls": ("cube", lambda g: NPLS1()),
    "pca_svr": ("unfolded", lambda g: GridSVR(pca=20)),
    "xgb": (TABULAR, lambda g: xgb()),
    "svr": (TABULAR, lambda g: GridSVR()),
    "gard_svr": (TABULAR, lambda g: GroupARDSVR(g, "rbf")),
    "gard_matern": (TABULAR, lambda g: GroupARDSVR(g, "matern", multipliers=(0.5, 1.0, 2.0))),
    "lgbm": (TABULAR, lambda g: lgbm()),
    "lgbm_oxy": (TABULAR + ["oxygen"], lambda g: lgbm()),
    "cnn": ("image", None),
}
FIELD_VARIANTS = ("peaks_mlr", "fri_mlr", "parafac_mlr", "parafac_rf", "parafac_xgb", "upls", "pca_svr", "cnn")


def resolve(name):
    """'parafac_xgb+field' -> (inputs, factory, with_field)."""
    base, plus_field = (name[:-6], True) if name.endswith("+field") else (name, False)
    if base not in MODEL_SPECS:
        raise KeyError(f"Unknown model {name!r}; known: {sorted(MODEL_SPECS)} (+field for {FIELD_VARIANTS})")
    if plus_field and base not in FIELD_VARIANTS:
        raise KeyError(f"{base} has no +field variant")
    inputs, factory = MODEL_SPECS[base]
    if plus_field and isinstance(inputs, list):
        inputs = inputs + ["field"]
    return inputs, factory, plus_field
