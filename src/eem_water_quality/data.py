"""Processed dataset: EEM cube, labels, field parameters, stations, dates and fold schemes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("BOD", "COD", "TOC")
FIELD = ("Temp", "pH", "DO", "SS", "EC")
LOG1P_FIELD = ("SS", "EC")


def column(frame: pd.DataFrame, name: str) -> str:
    """Resolve a short name ("COD") to the workbook column ("COD\\n(0.0)")."""
    if name in frame:
        return name
    short = lambda c: re.sub(r"\s*\(.*$", "", c.split("\n")[0]).strip().casefold()
    hits = [c for c in frame if short(c) == name.casefold()]
    if len(hits) != 1:
        raise KeyError(f"{name!r} must match exactly one column, found {hits}")
    return hits[0]


@dataclass
class Dataset:
    x: np.ndarray            # n x Em x Ex, clipped at 0, masked cells set to 0
    mask: np.ndarray         # Em x Ex, True = valid (non-scatter) cell
    em: np.ndarray
    ex: np.ndarray
    y: np.ndarray            # n x targets, mg/L
    targets: tuple
    field: np.ndarray        # n x 5, SS and EC as log1p
    station: np.ndarray
    date: np.ndarray         # datetime64[D]
    sample_id: np.ndarray
    folds: dict = field(default_factory=dict)   # scheme -> fold label per sample
    brackish: np.ndarray | None = None

    def __len__(self):
        return len(self.y)

    @property
    def station_rank(self) -> np.ndarray:
        """Order of each sample within its station by date (0 = first)."""
        frame = pd.DataFrame({"s": self.station, "d": self.date, "i": np.arange(len(self))})
        return frame.sort_values(["s", "d", "i"]).groupby("s").cumcount().reindex(frame.index).to_numpy()

    def subset(self, rows) -> Dataset:
        rows = np.asarray(rows)
        return Dataset(self.x[rows], self.mask, self.em, self.ex, self.y[rows], self.targets, self.field[rows],
                       self.station[rows], self.date[rows], self.sample_id[rows],
                       {k: v[rows] for k, v in self.folds.items()},
                       None if self.brackish is None else self.brackish[rows])


def load_dataset(path="data/processed", em_step=2, targets=TARGETS, brackish_ec=2000.0) -> Dataset:
    path = Path(path)
    s = pd.read_csv(path / "samples.csv")
    x = np.load(path / "eem.npy")[:, ::em_step, :]
    axes = np.load(path / "wavelengths.npz")
    mask = np.load(path / "eem_fixed_mask.npy")[::em_step, :]
    if x.shape[1:] != mask.shape or len(x) != len(s):
        raise ValueError("eem.npy, eem_fixed_mask.npy and samples.csv do not align")
    x = np.where(mask, np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0), 0.0).astype(np.float32)
    y = np.column_stack([pd.to_numeric(s[column(s, t)], errors="coerce") for t in targets]).astype(float)
    fld = np.column_stack([pd.to_numeric(s[column(s, f)], errors="coerce") for f in FIELD]).astype(float)
    for i, name in enumerate(FIELD):
        if name in LOG1P_FIELD:
            fld[:, i] = np.log1p(np.clip(fld[:, i], 0, None))
    station = s.Point.astype(str).to_numpy()
    date = pd.to_datetime(s["Sampling date"].astype(str).str.replace(".", "-", regex=False)).to_numpy("datetime64[D]")
    folds = {"station": s.station_cv_fold.astype(str).to_numpy()}
    for c in s.columns:
        if c.startswith("station_cv_fold_gkf5_"):
            folds["station_" + c.removeprefix("station_cv_fold_")] = s[c].astype(str).to_numpy()
    ec = np.expm1(fld[:, FIELD.index("EC")])
    brackish = pd.Series(station).map(pd.Series(ec).groupby(station).median() > brackish_ec).to_numpy()
    return Dataset(x, mask, axes["emission"][::em_step], axes["excitation"], y, tuple(targets), fld.astype(np.float32),
                   station, date, s.sample_id.to_numpy(), folds, brackish)
