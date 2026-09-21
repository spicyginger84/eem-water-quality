import json

import numpy as np
import pandas as pd
import pytest

from eem_water_quality.cli import main
from eem_water_quality.data import (
    BOD,
    COD,
    EC,
    SS,
    TOC,
    load_data,
    split_indices,
    target_partitions,
)
from eem_water_quality.features import (
    Experiment,
    FeatureBuilder,
    ParafacFeatures,
    experiment_catalog,
)
from eem_water_quality.metrics import regression_metrics
from eem_water_quality.predict import predict_saved


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(8)
    eem = rng.uniform(0.1, 2, (60, 8, 9)).astype(np.float32)
    samples = pd.DataFrame(
        {
            "Point": np.repeat(np.arange(15), 4),
            BOD: eem.mean(axis=(1, 2)) * 2,
            COD: np.full(60, 4.0),
            TOC: rng.normal(size=60),
            EC: rng.normal(size=60),
            SS: rng.normal(size=60),
        }
    )
    samples.loc[3, SS] = np.nan
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "eem.npy", eem)
    samples.to_parquet(data / "samples.parquet")
    return data, eem, samples


def test_splits_reproducible_group_disjoint_and_target_filtering(dataset):
    _, _, samples = dataset
    splits = split_indices(samples)
    second = split_indices(samples)
    assert len(np.concatenate(list(splits.values()))) == len(samples)
    for key in splits:
        np.testing.assert_array_equal(splits[key], second[key])
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert set(samples.iloc[splits[a]].Point).isdisjoint(samples.iloc[splits[b]].Point)
    samples.loc[splits["train"][0], BOD] = np.inf
    _, filtered = target_partitions(samples, BOD, splits)
    assert len(filtered["train"]) == len(splits["train"]) - 1
    np.testing.assert_array_equal(filtered["test"], splits["test"])


def test_ratio_zero_denominator_and_nonrange_index(dataset):
    directory, _, samples = dataset
    samples.loc[0, COD] = 0
    samples.index = np.arange(100, 160)
    samples.to_parquet(directory / "samples.parquet")
    _, loaded, _ = load_data(directory)
    assert loaded.index.equals(pd.RangeIndex(60))
    assert np.isnan(loaded.loc[0, "BOD_COD"])


def test_pca_and_imputation_fit_train_only(dataset):
    _, eem, samples = dataset
    exp = Experiment("pca_tab", "pca", tabular=(EC, SS))
    builder = FeatureBuilder(exp, BOD, pca_components=3)
    train = builder.fit_transform(eem[:30], samples.iloc[:30])
    assert train.shape == (30, 5)
    np.testing.assert_allclose(
        builder.eem_pipeline_[0].mean_, eem[:30].reshape(30, -1).mean(axis=0, dtype=float)
    )
    mean = builder.eem_pipeline_[0].mean_.copy()
    components = builder.eem_pipeline_[1].components_.copy()
    shifted = samples.iloc[30:].copy()
    shifted[EC] = 1e8
    assert np.isfinite(builder.transform(eem[30:] + 1000, shifted)).all()
    np.testing.assert_array_equal(builder.eem_pipeline_[0].mean_, mean)
    np.testing.assert_array_equal(builder.eem_pipeline_[1].components_, components)
    assert experiment_catalog()["EEMpca"].eem == "pca"
    assert len(experiment_catalog()) == 22


@pytest.mark.parametrize("nonnegative", [False, True])
def test_parafac_projection_on_fixed_training_basis(nonnegative):
    rng = np.random.default_rng(3)
    a, b, c = rng.uniform(0.2, 2, (3, 10, 1))
    tensor = np.einsum("ir,jr,kr->ijk", a, b, c)
    pf = ParafacFeatures(rank=1, nonnegative=nonnegative, max_iter=100)
    train = pf.fit_transform(tensor[:6])
    basis = pf.basis_.copy()
    test = pf.transform(tensor[6:])
    np.testing.assert_allclose(test @ basis.T, tensor[6:].reshape(4, -1), atol=1e-5)
    assert pf.relative_error(tensor[:6], train) < 1e-5
    np.testing.assert_array_equal(pf.basis_, basis)


def test_metrics_units():
    metrics = regression_metrics([1, 2], [2, 2])
    assert metrics["MSE"] == 0.5
    assert metrics["MAPE"] == 0.5
    assert metrics["RMSE"] == np.sqrt(0.5)


@pytest.mark.parametrize("command", ["ml", "parafac"])
def test_cli_artifact_reload(dataset, tmp_path, command):
    data, eem, samples = dataset
    output = tmp_path / command
    args = [command, "--data", str(data), "--output", str(output), "--targets", "BOD"]
    if command == "ml":
        args += [
            "--models",
            "linear",
            "tree",
            "--experiments",
            "TOC",
            "EEMpca",
            "--pca-components",
            "3",
        ]
    else:
        args += ["--pf-rank", "1", "--pf-max-iter", "10"]
    main(args)
    destination = output / "BOD"
    predictions = pd.read_csv(destination / "test_predictions.csv")
    indices = predictions.row_position.to_numpy()
    reloaded = predict_saved(destination, eem[indices], samples.iloc[indices])
    np.testing.assert_allclose(reloaded, predictions.y_pred, rtol=1e-8)
    selected = json.loads((destination / "selected.json").read_text())
    validation = pd.read_csv(destination / "validation_metrics.csv")
    assert selected["validation"]["RMSE"] == pytest.approx(validation.RMSE.min())
    assert (output / "run.json").exists()


def test_target_not_reintroduced_as_feature(dataset):
    _, eem, samples = dataset
    builder = FeatureBuilder(Experiment("tab", tabular=(BOD, EC)), BOD)
    assert builder.fit_transform(eem, samples).shape == (len(eem), 1)
    assert builder.tabular_columns == (EC,)
