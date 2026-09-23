"""Adapters for the processed EEM dataset on disk."""

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import BOD, COD, ProcessedDatasetSchema


def load_processed_dataset(directory):
    """Load aligned EEM, sample metadata and wavelength arrays."""
    directory = Path(directory)
    eem = np.load(directory / "eem.npy", allow_pickle=False).astype(np.float32)
    samples = pd.read_parquet(directory / "samples.parquet").reset_index(drop=True)
    ProcessedDatasetSchema().validate(eem, samples)
    if BOD in samples and COD in samples:
        denominator = pd.to_numeric(samples[COD], errors="coerce").replace(0, np.nan)
        samples["BOD_COD"] = pd.to_numeric(samples[BOD], errors="coerce") / denominator
    wavelengths = {}
    path = directory / "wavelengths.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            wavelengths = {key: archive[key] for key in archive.files}
        if "emission" in wavelengths and len(wavelengths["emission"]) != eem.shape[1]:
            raise ValueError("emission wavelength count does not match EEM row axis")
        if "excitation" in wavelengths and len(wavelengths["excitation"]) != eem.shape[2]:
            raise ValueError("excitation wavelength count does not match EEM column axis")
    return eem, samples, wavelengths


# Compatibility name used by the original pipeline.
load_data = load_processed_dataset

