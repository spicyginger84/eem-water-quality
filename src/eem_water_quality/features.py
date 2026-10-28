"""Feature blocks. Fixed blocks are functions of one sample; PARAFAC is fitted on train rows.

Fixed blocks (no fitted state, so no leakage):
  peaks    Coble (1996) peaks B, T, A, M, C as log1p window maxima, HIX (Ohno 2002),
           log BIX (Huguet 2009), log FI (Cory & McKnight 2005)
  fri      fluorescence regional integration (Chen 2003): log1p area-normalised volumes
  field    Temp, pH, DO, log1p SS, log1p EC
  season   sin / cos of day of year
  oxygen   DO deficit (DOsat - DO), log(DO / DOsat), DOsat, deficit x (pH < 8.5); DOsat from
           Benson & Krause (1984) with salinity ~ 0.64 EC (mS/cm), clipped to [0, 40]
Fitted block:
  parafac  non-negative PARAFAC (HALS, masked cells imputed by the model each sweep),
           log Fmax of each component; new samples are projected onto the train loadings
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .data import FIELD, Dataset

PEAK_WINDOWS = {  # name: (ex_lo, ex_hi, em_lo, em_hi)
    "B": (270, 280, 300, 320), "T": (270, 280, 330, 350), "A": (255, 265, 380, 460),
    "M": (305, 320, 380, 420), "C": (345, 355, 420, 480),
}
FRI_REGIONS = [((220, 250), (280, 330)), ((220, 250), (330, 380)), ((220, 250), (380, 550)),
               ((250, 500), (280, 380)), ((250, 500), (380, 550))]


def _window(D: Dataset, ex_lo, ex_hi, em_lo, em_hi, op):
    sel = D.x[:, (D.em >= em_lo) & (D.em <= em_hi)][:, :, (D.ex >= ex_lo) & (D.ex <= ex_hi)]
    return op(sel.reshape(len(D), -1), axis=1)


def peaks(D: Dataset):
    cols = [np.log1p(_window(D, *w, np.max)) for w in PEAK_WINDOWS.values()]
    hi = _window(D, 250, 255, 435, 480, np.sum)
    lo = _window(D, 250, 255, 300, 345, np.sum)
    bix = _window(D, 310, 310, 378, 382, np.mean) / (_window(D, 310, 310, 428, 432, np.mean) + 1e-9)
    fi = _window(D, 370, 370, 468, 472, np.mean) / (_window(D, 370, 370, 518, 522, np.mean) + 1e-9)
    cols += [hi / (hi + lo + 1e-9), np.log(bix + 1e-6), np.log(fi + 1e-6)]
    return np.column_stack(cols), [f"peak_{p}" for p in PEAK_WINDOWS] + ["HIX", "log_BIX", "log_FI"]


def fri(D: Dataset):
    total = D.mask.sum()
    cols = []
    for (a, b), (c, d) in FRI_REGIONS:
        r = ((D.em >= c) & (D.em < d))[:, None] & ((D.ex >= a) & (D.ex < b))[None] & D.mask
        cols.append(np.log1p(D.x[:, r].sum(1) * total / max(r.sum(), 1)))
    return np.column_stack(cols), [f"FRI_{i + 1}" for i in range(5)]


def field_block(D: Dataset):
    names = [f"log1p_{f}" if f in ("SS", "EC") else f for f in FIELD]
    return D.field.astype(float), names


def season(D: Dataset):
    doy = (D.date - D.date.astype("datetime64[Y]")).astype(int) + 1
    angle = 2 * np.pi * doy / 365.25
    return np.column_stack([np.sin(angle), np.cos(angle)]), ["season_sin", "season_cos"]


def oxygen(D: Dataset):
    temp, ph, do = (D.field[:, FIELD.index(f)].astype(float) for f in ("Temp", "pH", "DO"))
    salinity = np.clip(np.expm1(D.field[:, FIELD.index("EC")].astype(float)) / 1000 * 0.64, 0, 40)
    tk = temp + 273.15
    do_sat = np.exp(-139.34411 + 1.575701e5 / tk - 6.642308e7 / tk ** 2 + 1.2438e10 / tk ** 3
                    - 8.621949e11 / tk ** 4 - salinity * (1.7674e-2 - 1.0754e1 / tk + 2.1407e3 / tk ** 2))
    deficit = do_sat - do
    cols = [deficit, np.log(np.clip(do, 0.1, None) / do_sat), do_sat, deficit * (ph < 8.5)]
    return np.column_stack(cols), ["DO_deficit", "log_DO_sat_ratio", "DO_sat", "DO_deficit_pH_lt_8.5"]


FIXED_BLOCKS = {"peaks": peaks, "fri": fri, "field": field_block, "season": season, "oxygen": oxygen}


class Parafac:
    """Non-negative CP of an n x J x K cube by HALS; masked cells are EM-imputed."""

    def __init__(self, rank=4, iters=200, seed=11):
        self.rank, self.iters, self.seed = rank, iters, seed

    def fit(self, x, mask):
        rng = np.random.default_rng(self.seed)
        n, J, K = x.shape
        A, B, C = (rng.random((d, self.rank)) + 0.1 for d in (n, J, K))
        x = x.astype(np.float64)
        miss = ~mask
        for _ in range(self.iters):
            if miss.any():
                rec = np.einsum("ir,jr,kr->ijk", A, B, C, optimize=True)
                x[:, miss] = rec[:, miss]
            for mode in range(3):
                if mode == 0:
                    M, G = np.einsum("ijk,jr,kr->ir", x, B, C, optimize=True), (B.T @ B) * (C.T @ C)
                    F = A
                elif mode == 1:
                    M, G = np.einsum("ijk,ir,kr->jr", x, A, C, optimize=True), (A.T @ A) * (C.T @ C)
                    F = B
                else:
                    M, G = np.einsum("ijk,ir,jr->kr", x, A, B, optimize=True), (A.T @ A) * (B.T @ B)
                    F = C
                for r in range(self.rank):
                    F[:, r] = np.maximum(F[:, r] + (M[:, r] - F @ G[:, r]) / max(G[r, r], 1e-12), 1e-12)
            for F in (B, C):
                norm = np.linalg.norm(F, axis=0)
                F /= norm
                A *= norm
        self.mask, self.B, self.C = mask, B, C
        self.design = np.einsum("jr,kr->jkr", B, C)[mask]         # valid cells x rank
        self.scale = B.max(0) * C.max(0)
        return self

    def transform(self, x):
        scores = np.array([nnls(self.design, xi[self.mask])[0] for xi in x.astype(np.float64)])
        return np.log(scores * self.scale + 1e-4)                  # log Fmax


class FeatureBuilder:
    """Builds named feature blocks; fixed blocks once, fitted blocks once per train set."""

    def __init__(self, D: Dataset, parafac_rank=4, parafac_iters=200, seed=11):
        self.D = D
        self.parafac_args = {"rank": parafac_rank, "iters": parafac_iters, "seed": seed}
        self.fixed = {name: fn(D) for name, fn in FIXED_BLOCKS.items()}
        self._fitted = {}

    def fit_block(self, name, key, train):
        """Fitted block for all rows, using only `train` rows to fit; cached under `key`."""
        if (name, key) not in self._fitted:
            if name != "parafac":
                raise KeyError(name)
            model = Parafac(**self.parafac_args).fit(self.D.x[train], self.D.mask)
            self._fitted[(name, key)] = (model.transform(self.D.x), [f"PARAFAC_C{r + 1}" for r in range(model.rank)])
        return self._fitted[(name, key)]

    def table(self, blocks, key=None, train=None):
        """Concatenate blocks; returns matrix, column names and column index per block."""
        mats, names, groups, start = [], [], [], 0
        for b in blocks:
            m, n = self.fixed[b] if b in self.fixed else self.fit_block(b, key, train)
            mats.append(m)
            names += n
            groups.append(np.arange(start, start + m.shape[1]))
            start += m.shape[1]
        return np.column_stack(mats).astype(np.float64), names, groups

    def unfolded(self):
        return self.D.x[:, self.D.mask].astype(np.float64)
