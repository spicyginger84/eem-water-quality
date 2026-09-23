"""Names and lightweight validation for the processed dataset contract."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

BOD = "BOD\n(0.0)"
COD = "COD\n(0.0)"
TOC = "TOC\n(0.0)"
SS = "SS\n(0.0)"
EC = "EC\n(0)"
TEMP = "Temp(0.0)"
PH = "pH(0.0)"

ALIASES = {
    "BOD": BOD,
    "COD": COD,
    "TOC": TOC,
    "SS": SS,
    "EC": EC,
    "TEMP": TEMP,
    "Temp": TEMP,
    "PH": PH,
    "pH": PH,
    "BOD_COD": "BOD_COD",
}


def resolve_column(name):
    """Resolve a user-facing target name to its processed-table column."""
    return ALIASES.get(name, name)


@dataclass(frozen=True)
class ProcessedDatasetSchema:
    """Shape and key requirements for aligned processed EEM data."""

    eem_ndim: int = 3
    station_column: str = "Point"
    month_column: str = "Month"

    def validate(self, eem, samples: pd.DataFrame) -> None:
        if np.asarray(eem).ndim != self.eem_ndim:
            raise ValueError("Expected EEM with shape (samples, emission, excitation)")
        if len(eem) != len(samples):
            raise ValueError("EEM and metadata must contain the same number of samples")
        if not np.isfinite(eem).all():
            raise ValueError("EEM contains NaN/Inf; repair processed data first")

