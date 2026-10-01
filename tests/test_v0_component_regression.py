"""Check feature domain, held-out predictions, metrics and component selection."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

spec = importlib.util.spec_from_file_location(
    "v0_component_regression",
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/evaluate_v0_component_regression.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_component_max_ignores_unobserved_extrapolation():
    surfaces = np.array([[[1., 2.], [3., 999.]], [[2., 4.], [1., 1000.]]])
    mask = np.array([[True, True], [True, False]])
    amplitudes = np.array([[2., 3.], [0., 1.]])
    np.testing.assert_allclose(module.component_max_features(surfaces, amplitudes, mask),
                               [[6., 12.], [0., 4.]])


def test_oof_baseline_uses_only_training_targets_and_metrics_match():
    x = np.arange(23.)[:, None]
    y = np.arange(23.)**2
    scores, oof = module.evaluate_regressor(x, y, DummyRegressor())
    assert not oof.position.duplicated().any()
    assert set(oof.position) == set(range(23))
    expected = np.empty(23)
    fold_r2 = []
    for train, test in KFold(5, shuffle=True, random_state=42).split(x):
        expected[test] = y[train].mean()
        fold_r2.append(r2_score(y[test], expected[test]))
    np.testing.assert_allclose(oof.sort_values("position").predicted, expected)
    np.testing.assert_allclose(scores["R2_cv_mean"], np.mean(fold_r2))
    np.testing.assert_allclose(scores["R2_cv_std"], np.std(fold_r2, ddof=1))
    np.testing.assert_allclose(scores["R2_oof"], r2_score(y, expected))


def test_linear_recovery_preserves_original_positions_after_missing_values():
    x = np.arange(25.)[:, None]
    y = 3*x[:, 0]+2
    y[4] = np.nan
    scores, oof = module.evaluate_regressor(x, y, LinearRegression())
    assert 4 not in set(oof.position)
    assert scores["n"] == 24
    np.testing.assert_allclose([scores[k] for k in ["R2_train", "R2_cv_mean", "R2_oof"]], 1)


def test_best_single_component_excludes_all_and_failed_rows():
    base = {"variant": "v", "station": "s", "target": "TOC", "model": "linear", "status": "ok"}
    rows = [{**base, "component": name, "R2_cv_mean": score}
            for name, score in [("B", .1), ("T", .3), ("all_components", .8)]]
    rows.append({**base, "component": "C", "R2_cv_mean": .9, "status": "insufficient_or_constant"})
    assert module.best_components(pd.DataFrame(rows)).component.tolist() == ["T"]


def test_regressor_recovers_joint_signal_and_preserves_fold_assignment():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 3))
    y = x @ np.array([2., -3., 4.]) + 1
    scores, oof = module.evaluate_regressor(x, y, LinearRegression())
    _, single_oof = module.evaluate_regressor(x[:, :1], y, LinearRegression())
    np.testing.assert_allclose([scores["R2_train"], scores["R2_cv_mean"], scores["R2_oof"]], 1)
    pd.testing.assert_frame_equal(oof[["position", "fold"]], single_oof[["position", "fold"]])


def test_regressor_rejects_empty_feature_matrix():
    with pytest.raises(ValueError, match="one or more peak"):
        module.evaluate_regressor(np.empty((20, 0)), np.arange(20.), LinearRegression())


def test_decomposition_receives_only_train_rows_after_target_filtering():
    from types import SimpleNamespace

    eems = np.arange(25., dtype=float)[:, None, None] + 1
    ids = np.arange(25)[:, None]
    y = np.arange(25., dtype=float) ** 2
    y[4] = np.nan
    calls = []

    def fit(train):
        calls.append(train[:, 0, 0].astype(int) - 1)
        # Features depend on training data, making accidental global reuse visible.
        center = train.mean()
        return SimpleNamespace(
            full_surfaces=np.ones((1, 1, 1)), mask=np.ones((1, 1), bool),
            transform=lambda values: (values[:, 0, 0] - center)[:, None])

    features = module.cached_component_features(eems, fit)
    _, oof = module.evaluate_regressor(ids, y, LinearRegression(), feature_fitter=features)
    valid = np.flatnonzero(np.isfinite(y))
    assert len(calls) == 6  # descriptive full fit + five independent training fits
    np.testing.assert_array_equal(calls[0], valid)
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=42).split(valid)):
        np.testing.assert_array_equal(calls[fold+1], valid[train])
        assert set(calls[fold+1]).isdisjoint(valid[test])
    assert 4 not in oof.position.values
    module.evaluate_regressor(ids, y, DummyRegressor(), feature_fitter=features)
    assert len(calls) == 6  # reuse only identical membership across estimators


def test_heldout_only_signal_does_not_change_component_mask():
    from eem_water_quality.features.peak_decomposition import fit_peak_model

    rng = np.random.default_rng(7)
    raw = rng.uniform(1, 2, (12, 4, 4))
    raw[:10, 0, 0] = 0
    fits = []

    def fit(train):
        fitted = fit_peak_model(train, np.arange(4.) + 250, np.arange(4.) + 350,
                                rank=1, n_starts=1, max_steps=2, seed=42)
        fits.append(fitted)
        return fitted

    train, test = np.arange(10)[:, None], np.array([[10], [11]])
    first = module.cached_component_features(raw, fit)(train, test)
    changed = raw.copy()
    changed[10:, 0, 0] = 1e9
    second = module.cached_component_features(changed, fit)(train, test)
    assert not fits[0].mask[0, 0]
    np.testing.assert_allclose(fits[0].full_surfaces, fits[1].full_surfaces)
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])


def test_parafac_runner_uses_raw_eems_without_saved_features(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts/diagnostics'))
    import evaluate_parafac_all_components as parafac

    data = tmp_path / 'data/processed'
    data.mkdir(parents=True)
    output = tmp_path / 'output'
    output.mkdir()
    rng = np.random.default_rng(12)
    np.save(data / 'eem.npy', rng.uniform(1, 3, (10, 4, 4)))
    np.savez(data / 'wavelengths.npz', excitation=np.arange(4.)+250,
             emission=np.arange(4.)+350)
    pd.DataFrame({'Group': ['g']*10, 'BOD': np.arange(10.),
                  'COD': np.arange(10.)**2, 'TOC': np.arange(10.)+2}).to_csv(
        data / 'samples.csv', index=False)
    pd.DataFrame({'Group': ['g']}).to_csv(output / 'selected_top_groups.csv', index=False)
    monkeypatch.setattr(parafac, 'ROOT', tmp_path)
    scores = parafac.run_all_components(output, rank=1, max_steps=2,
                                       n_starts=1, include_single=True)
    assert len(scores) == 6
    assert scores.status.eq('ok').all()
    oof = pd.read_csv(output / 'group_all_components_oof_predictions.csv')
    assert len(oof) == 60
    assert oof.groupby(['target', 'model']).sample_index.nunique().eq(10).all()
    assert (output / 'group_best_component_linear_cv_r2.csv').exists()
