import json

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression

from eem_water_quality.data import load_dataset
from eem_water_quality.features import FeatureBuilder, Parafac
from eem_water_quality.metrics import paired_station_bootstrap, regression_metrics
from eem_water_quality.models import MODEL_SPECS, NPLS1, PLS1, resolve
from eem_water_quality.protocols import calibrate, outer_splits
from eem_water_quality.runner import Benchmark, Config


def test_load_dataset(processed):
    D = load_dataset(processed)
    assert D.x.shape == (48, 136, 57) and D.mask.shape == (136, 57)
    assert (D.x[:, ~D.mask] == 0).all() and (D.x >= 0).all()
    assert np.allclose(np.expm1(D.field[:, 4]), np.where(D.station == "S5", 9000, 400))
    assert set(D.folds) == {"station", "station_gkf5_seed11"}
    assert D.brackish[D.station == "S5"].all() and not D.brackish[D.station != "S5"].any()
    rank = D.station_rank
    for s in np.unique(D.station):
        ids = np.flatnonzero(D.station == s)
        assert (np.argsort(rank[ids]) == np.argsort(D.date[ids], kind="stable")).all()


def test_fixed_blocks_finite(processed):
    fb = FeatureBuilder(load_dataset(processed))
    for name, (m, names) in fb.fixed.items():
        assert m.shape[1] == len(names) and np.isfinite(m).all(), name


def test_oxygen_saturation_reference(processed):
    D = load_dataset(processed).subset([0])
    D.field[0, :3] = [20.0, 7.0, 9.092]  # Temp, pH, DO; fresh water at 20 C saturates at 9.09 mg/L
    D.field[0, 4] = np.log1p(0.0)
    m, names = FeatureBuilder(D).fixed["oxygen"]
    assert abs(m[0, names.index("DO_sat")] - 9.092) < 0.01 and abs(m[0, names.index("DO_deficit")]) < 0.01


def test_parafac_recovers_low_rank():
    rng = np.random.default_rng(1)
    A, B, C = rng.random((40, 2)) + 0.1, rng.random((30, 2)), rng.random((20, 2))
    x = np.einsum("ir,jr,kr->ijk", A, B, C)
    mask = np.ones((30, 20), bool)
    mask[:5, :5] = False
    p = Parafac(rank=2, iters=300).fit(np.where(mask, x, 0.0), mask)
    rec = np.einsum("ir,jr,kr->ijk", np.exp(p.transform(x)) / p.scale, p.B, p.C)
    assert np.linalg.norm((rec - x)[:, mask]) / np.linalg.norm(x[:, mask]) < 0.02


def test_pls1_matches_sklearn():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(60, 12)), rng.normal(size=60)
    coefs = PLS1.nipals(X - X.mean(0), y - y.mean(), 4)
    for a in (1, 4):
        ref = PLSRegression(a, scale=False).fit(X, y).predict(X).ravel()
        assert np.allclose((X - X.mean(0)) @ coefs[a - 1] + y.mean(), ref, atol=1e-6)


def test_every_estimator_fits_and_clips(processed):
    D = load_dataset(processed)
    fb = FeatureBuilder(D, parafac_iters=20)
    tr = np.flatnonzero(D.folds["station"] != "1")
    ly = np.log(D.y[:, 2])
    for name, (inputs, factory) in MODEL_SPECS.items():
        if factory is None:
            continue
        if isinstance(inputs, list):
            Z, _, groups = fb.table(inputs, "k", tr)
        else:
            Z, groups = (fb.unfolded() if inputs == "unfolded" else D.x.astype(float)), None
        est = factory(groups).fit(Z[tr], ly[tr], D.station[tr])
        p = est.predict(Z)
        assert np.isfinite(p).all() and p.min() >= ly[tr].min() - np.log(2) - 1e-9, name
    assert isinstance(NPLS1().fit(D.x[tr].astype(float), ly[tr], D.station[tr]).n, int)


def test_resolve_field_variants():
    assert resolve("parafac_xgb+field")[0] == ["parafac", "field"]
    assert resolve("upls+field")[2] is True
    with pytest.raises(KeyError):
        resolve("svr+field")


def test_station_splits_disjoint(processed):
    D = load_dataset(processed)
    for scheme in ("station", "station_gkf5_seed11", "loso"):
        for tr, te in outer_splits(D, scheme):
            assert not set(D.station[tr]) & set(D.station[te])
    assert sum(len(te) for _, te in outer_splits(D, "random")) == len(D)


def test_rolling_calibration_uses_only_earlier_labels(processed):
    D = load_dataset(processed)
    ly = np.log(D.y)
    zero = np.zeros_like(ly)
    out, query = calibrate(D, zero, "rolling", k=3, prior=3.0)
    rank = D.station_rank
    assert (query == (rank >= 3)).all()
    i = np.flatnonzero(query)[0]
    ids = np.flatnonzero(D.station == D.station[i])
    earlier = ids[(rank[ids] < rank[i]) & (rank[ids] >= rank[i] - 3)]
    assert np.allclose(out[i], ly[earlier].sum(0) / 6)
    # changing a later label must not change the calibrated prediction
    D.y[ids[rank[ids] > rank[i]]] *= 10
    assert np.allclose(calibrate(D, zero, "rolling", 3, 3.0)[0][i], out[i])


def test_early_calibration_query_excludes_support(processed):
    D = load_dataset(processed)
    _, query = calibrate(D, np.zeros_like(D.y), "early", k=3)
    assert (query == (D.station_rank >= 3)).all()


def test_metrics_and_bootstrap():
    y, p = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 5.0])
    st = np.array(["a", "a", "b", "b"])
    m = regression_metrics(y, p, st, np.array([False, False, True, True]))
    assert m["r2"] == pytest.approx(0.8) and m["station_mae"] == pytest.approx(0.25)
    d, lo, hi = paired_station_bootstrap(y, p, y, st, n=100)
    assert d == pytest.approx(-0.25) and lo <= d <= hi


def test_runner_end_to_end(processed, tmp_path):
    cfg = Config(data=str(processed), schemes=["random", "station"], models=["peaks_mlr", "parafac_mlr+field", "svr"],
                 reference="peaks_mlr", parafac_iters=20, bootstrap=50, workers=2)
    metrics, summary = Benchmark(cfg).run(tmp_path)
    for f in ("predictions.parquet", "metrics.csv", "summary.csv", "paired.csv", "report.md", "config.json"):
        assert (tmp_path / f).exists(), f
    zero = metrics[(metrics.regime == "zero-shot") & (metrics.subset == "all")]
    assert (zero.n == 48).all() and np.isfinite(zero.r2).all()
    assert {"few-shot early", "few-shot rolling", "few-shot random"} <= set(metrics.regime)
    assert "random" not in set(summary.columns) and summary.schemes.max() == 1
    preds = pd.read_parquet(tmp_path / "predictions.parquet")
    assert np.isfinite(preds.y_pred).all()
    assert json.loads((tmp_path / "config.json").read_text())["models"] == cfg.models
