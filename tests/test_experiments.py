import numpy as np
import pandas as pd
import pytest

from eem_water_quality.cli import main
from eem_water_quality.data import (
    BOD,
    COD,
    EC,
    PH,
    SS,
    TEMP,
    TOC,
    group_kfold_indices,
    load_data,
    split_indices,
    target_partitions,
)
from eem_water_quality.evaluation.metrics import regression_metrics
from eem_water_quality.features import (
    Experiment,
    FeatureBuilder,
    experiment_catalog,
)


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


def test_two_way_holdout_and_grouped_cv(dataset):
    _, _, samples = dataset
    samples["Month"] = np.tile(np.arange(4), 15)
    splits = split_indices(samples, mode="two_way", seed=42)
    assert set(samples.iloc[splits["test"]].Point).isdisjoint(
        samples.iloc[splits["train"]].Point
    )
    assert set(samples.iloc[splits["test"]].Month).isdisjoint(
        samples.iloc[splits["train"]].Month
    )
    train_val = np.concatenate([splits["train"], splits["validation"]])
    folds = list(group_kfold_indices(samples, train_val, n_splits=5))
    assert len(folds) == 5
    assert set(np.concatenate([np.concatenate(fold) for fold in folds])).isdisjoint(
        splits["test"]
    )
    for train, validation in folds:
        assert set(samples.iloc[train].Point).isdisjoint(samples.iloc[validation].Point)


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
    assert experiment_catalog()["SS_EC"].tabular == (SS, EC)
    assert experiment_catalog()["Temp_pH"].tabular == (TEMP, PH)
    assert len(experiment_catalog()) == 29


def test_eem_zero_columns_are_masked_before_pca(dataset):
    _, eem, samples = dataset
    eem = eem.copy()
    eem[:, 0, 0] = 0
    builder = FeatureBuilder(Experiment("pca", "pca"), BOD, pca_components=3)
    features = builder.fit_transform(eem, samples)
    assert not builder.eem_feature_mask_[0]
    assert features.shape == (len(eem), 3)


def test_metrics_units():
    metrics = regression_metrics([1, 2], [2, 2])
    assert metrics["MSE"] == 0.5
    assert metrics["MAPE"] == 0.5
    assert metrics["RMSE"] == np.sqrt(0.5)


def test_cli_cv_artifacts(dataset, tmp_path):
    data, _, _ = dataset
    output = tmp_path / "ml_cv"
    args = [
        "ml",
        "--data",
        str(data),
        "--output",
        str(output),
        "--targets",
        "BOD",
        "--models",
        "linear",
        "tree",
        "--features",
        "EEMpca",
        "--pca-components",
        "3",
    ]
    main(args)
    destination = output / "BOD"
    predictions = pd.read_csv(destination / "cv_predictions.csv")
    metrics = pd.read_csv(destination / "cv_fold_metrics.csv")
    summary = pd.read_csv(destination / "cv_summary.csv")
    assert set(predictions.protocol) == {"cv"}
    assert set(metrics.fold.dropna().astype(int)) == {1, 2, 3, 4, 5}
    assert set(summary.fold) == {"mean", "pooled"}
    assert set(metrics.features.dropna()) >= {"EEMpca", "baseline_global_mean"}
    assert "delta_R2_vs_global" in metrics.columns
    summary_model = summary[summary.features == "EEMpca"]
    assert summary_model["delta_R2_vs_global"].notna().all()
    assert not (destination / "selected.json").exists()
    assert (output / "run.json").exists()


def test_two_way_and_loso_protocol_outputs(dataset, tmp_path):
    data, _, samples = dataset
    samples = samples.copy()
    samples["Month"] = np.tile(np.arange(4), 15)
    samples.to_parquet(data / "samples.parquet")

    two_way = tmp_path / "two_way"
    main(
        [
            "ml",
            "--data",
            str(data),
            "--output",
            str(two_way),
            "--targets",
            "BOD",
            "--features",
            "EEMpca",
            "--models",
            "linear",
            "--split",
            "two_way",
            "--pca-components",
            "3",
        ]
    )
    assert (two_way / "BOD" / "holdout_metrics.csv").exists()
    assert not (two_way / "BOD" / "cv_fold_metrics.csv").exists()

    loso = tmp_path / "loso"
    main(
        [
            "ml",
            "--data",
            str(data),
            "--output",
            str(loso),
            "--targets",
            "BOD",
            "--features",
            "EEMpca",
            "--models",
            "linear",
            "--holdout-protocol",
            "loso",
            "--pca-components",
            "3",
        ]
    )
    metrics = pd.read_csv(loso / "BOD" / "holdout_metrics.csv")
    assert set(metrics.protocol) == {"loso"}
    assert metrics.fold.nunique() == 15
    assert not (loso / "BOD" / "cv_fold_metrics.csv").exists()


def test_target_not_reintroduced_as_feature(dataset):
    _, eem, samples = dataset
    builder = FeatureBuilder(Experiment("tab", tabular=(BOD, EC)), BOD)
    assert builder.fit_transform(eem, samples).shape == (len(eem), 1)
    assert builder.tabular_columns == (EC,)
