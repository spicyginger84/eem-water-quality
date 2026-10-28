"""Synthetic dataset in the data/processed layout: Gaussian fluorophores on the real grid."""

import numpy as np
import pandas as pd
import pytest

EM = np.arange(280, 551, 1.0)
EX = np.arange(220, 501, 5.0)


def blob(ex0, em0, sx=20, se=30):
    return np.exp(-0.5 * (((EX[None] - ex0) / sx) ** 2 + ((EM[:, None] - em0) / se) ** 2))


@pytest.fixture(scope="session")
def processed(tmp_path_factory):
    rng = np.random.default_rng(0)
    stations, per = [f"S{i}" for i in range(6)], 8
    n = len(stations) * per
    humic, protein = rng.lognormal(0, 0.4, n), rng.lognormal(-1, 0.4, n)
    x = humic[:, None, None] * blob(340, 440) + protein[:, None, None] * blob(275, 340)
    mask = EM[:, None] >= EX[None] + 20
    x = np.where(mask, x, 0.0) + rng.normal(0, 0.002, x.shape)
    station = np.repeat(stations, per)
    level = np.repeat(rng.lognormal(0, 0.3, len(stations)), per)
    toc = 2 * humic * level + 0.5
    s = pd.DataFrame({
        "sample_id": np.arange(n), "Point": station,
        "Sampling date": [f"2023.{m:02d}.{d:02d}" for m, d in zip(np.tile(np.arange(1, per + 1), len(stations)), np.repeat(np.arange(1, 7), per))],
        "Temp(0.0)": rng.normal(15, 5, n), "pH(0.0)": rng.normal(7.5, 0.3, n), "DO\n(0.0)": rng.normal(9, 1, n),
        "SS\n(0.0)": rng.lognormal(1.5, 0.5, n), "EC\n(0)": np.where(np.isin(station, ["S5"]), 9000.0, 400.0),
        "BOD\n(0.0)": 0.5 * toc + protein, "COD\n(0.0)": 1.5 * toc, "TOC\n(0.0)": toc,
        "station_cv_fold": pd.Series(station).map({s: i % 3 + 1 for i, s in enumerate(stations)}),
        "station_cv_fold_gkf5_seed11": pd.Series(station).map({s: (i + 1) % 3 + 1 for i, s in enumerate(stations)}),
    })
    path = tmp_path_factory.mktemp("processed")
    s.to_csv(path / "samples.csv", index=False)
    np.save(path / "eem.npy", x.astype(np.float32))
    np.save(path / "eem_fixed_mask.npy", mask)
    np.savez(path / "wavelengths.npz", excitation=EX, emission=EM)
    return path
