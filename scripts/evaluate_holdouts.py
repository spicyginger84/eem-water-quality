"""Run fast Phase-2 holdout diagnostics.

The default protocol is leave-one-group-out cross-validation for station (``Point``)
or month (``Month``).  ``--protocol two_way`` instead creates one outer holdout
whose test rows are simultaneously from an unseen primary and secondary group.
Feature preprocessing is fitted inside each fold (or on the two-way training set),
and the supplied model/experiment combinations are evaluated without selecting a
winner on held-out rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eem_water_quality.data import group_kfold_indices, load_data, resolve_column, split_indices
from eem_water_quality.features import FeatureBuilder, experiment_catalog
from eem_water_quality.metrics import regression_metrics
from eem_water_quality.ml import make_model

METRIC_NAMES = ("MSE", "RMSE", "MAE", "MAPE", "R2")


def _fold_specs(args, samples):
    if args.protocol == "loso":
        if args.group_col not in samples:
            raise ValueError(f"group column {args.group_col!r} is absent from samples")
        groups = samples[args.group_col]
        ordered = sorted(groups.dropna().unique().tolist(), key=str)
        indices = np.arange(len(samples))
        return [
            {
                "group": group,
                "train": indices[~groups.eq(group).to_numpy()],
                "test": indices[groups.eq(group).to_numpy()],
            }
            for group in ordered
        ], None

    splits = split_indices(
        samples,
        mode="two_way",
        group_col=args.group_col,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        secondary_group_col=args.secondary_group_col,
    )
    train = np.concatenate([splits["train"], splits["validation"]])
    return [{"group": "two_way", "train": train, "test": splits["test"]}], splits


def _write_split_artifact(output, samples, splits):
    if splits is None:
        return
    frame = samples.copy()
    frame.insert(0, "row_position", np.arange(len(frame)))
    frame["partition"] = ""
    for name, indices in splits.items():
        frame.loc[indices, "partition"] = name
    frame.to_csv(output / "splits.csv", index=False)


def _add_fold_result(rows, predictions, common, status, metrics, truth, prediction, group):
    rows.append({**common, "group": group, "status": status, **metrics})
    predictions.append(
        pd.DataFrame(
            {
                **{key: common[key] for key in ("target", "experiment", "model")},
                "group": group,
                "row_position": common["_test_indices"],
                "y_true": truth,
                "y_pred": prediction,
            }
        )
    )


def _add_aggregates(rows, common, fold_metrics, truths, predictions, protocol, status):
    if protocol != "loso" or len(fold_metrics) < 2:
        return
    metric_frame = pd.DataFrame(fold_metrics)
    mean_metrics = {name: float(metric_frame[name].mean()) for name in METRIC_NAMES}
    mean_metrics["R2_std"] = float(metric_frame["R2"].std(ddof=1) or 0.0)
    rows.append(
        {
            **common,
            "group": "mean",
            "status": status,
            "n_folds": len(fold_metrics),
            "n_test": int(sum(len(values) for values in truths)),
            **mean_metrics,
        }
    )
    pooled = regression_metrics(np.concatenate(truths), np.concatenate(predictions))
    rows.append(
        {
            **common,
            "group": "pooled",
            "status": f"{status}_pooled",
            "n_folds": len(fold_metrics),
            "n_test": int(sum(len(values) for values in truths)),
            **pooled,
        }
    )


def _add_baseline_deltas(frame, protocol):
    """Compare mean rows to mean baseline and pooled rows to pooled baseline."""
    if protocol == "loso":
        mean_status = "baseline_mean"
        pooled_status = "baseline_mean_pooled"
    else:
        mean_status = pooled_status = "baseline"
    mean_baseline = frame[
        (frame["experiment"] == "baseline_global_mean")
        & (frame["status"] == mean_status)
    ].set_index("target")["R2"]
    pooled_baseline = frame[
        (frame["experiment"] == "baseline_global_mean")
        & (frame["status"] == pooled_status)
    ].set_index("target")["R2"]

    def delta(row):
        baselines = pooled_baseline if str(row.get("status", "")).endswith("_pooled") else mean_baseline
        return row["R2"] - baselines[row["target"]] if row["target"] in baselines else np.nan

    frame["delta_R2_vs_global"] = frame.apply(
        lambda row: delta(row) if pd.notna(row.get("R2")) else np.nan,
        axis=1,
    )
    return frame


def _nested_feature_cache(args, eem, samples, fold_specs, experiments):
    """Fit EEM representations for every outer/inner training partition once."""
    cache = {}
    mask_rows = []
    for experiment_name in experiments:
        experiment = experiment_catalog()[experiment_name]
        if experiment.tabular:
            raise ValueError("--cv-folds currently supports EEM-only experiments")
        outer_cache = {}
        for spec in fold_specs:
            outer_train = spec["train"]
            outer_test = spec["test"]
            inner_folds = list(
                group_kfold_indices(
                    samples, outer_train, args.cv_folds, args.cv_group_col
                )
            )
            inner_cache = []
            for inner_train, inner_validation in inner_folds:
                builder = FeatureBuilder(
                    experiment,
                    "__phase2__",
                    args.pca_components,
                    args.pf_rank,
                    False,
                    args.pf_max_iter,
                    args.seed,
                )
                x_train = builder.fit_transform(eem[inner_train], samples.iloc[inner_train])
                x_validation = builder.transform(
                    eem[inner_validation], samples.iloc[inner_validation]
                )
                inner_cache.append(
                    {
                        "train": inner_train,
                        "validation": inner_validation,
                        "x_train": x_train,
                        "x_validation": x_validation,
                    }
                )
            builder = FeatureBuilder(
                experiment,
                "__phase2__",
                args.pca_components,
                args.pf_rank,
                False,
                args.pf_max_iter,
                args.seed,
            )
            x_outer_train = builder.fit_transform(eem[outer_train], samples.iloc[outer_train])
            x_outer_test = builder.transform(eem[outer_test], samples.iloc[outer_test])
            mask_rows.append(
                {
                    "experiment": experiment_name,
                    "group": spec["group"],
                    "raw_features": int(np.prod(eem.shape[1:])),
                    "features_after_zero_mask": int(builder.eem_feature_mask_.sum()),
                    "zero_masked_features": int((~builder.eem_feature_mask_).sum()),
                }
            )
            outer_cache[spec["group"]] = {
                "train": outer_train,
                "test": outer_test,
                "x_train": x_outer_train,
                "x_test": x_outer_test,
                "inner": inner_cache,
            }
        cache[experiment_name] = outer_cache
    return cache, mask_rows


def _nested_cv_score(cache, y, model_name, args):
    """Return mean and pooled inner-CV scores for one outer training set."""
    fold_metrics = []
    truths, predictions = [], []
    for fold in cache["inner"]:
        train = fold["train"]
        validation = fold["validation"]
        train_valid = np.isfinite(y[train])
        validation_valid = np.isfinite(y[validation])
        if train_valid.sum() < 2 or validation_valid.sum() < 2:
            continue
        model = make_model(model_name, args.seed, args.n_jobs)
        model.fit(fold["x_train"][train_valid], y[train[train_valid]])
        prediction = np.asarray(
            model.predict(fold["x_validation"][validation_valid]), dtype=float
        )
        truth = y[validation[validation_valid]]
        fold_metrics.append(regression_metrics(truth, prediction))
        truths.append(truth)
        predictions.append(prediction)
    if len(fold_metrics) < 2:
        raise ValueError("Each outer training set needs at least two valid CV folds")
    metric_frame = pd.DataFrame(fold_metrics)
    pooled = regression_metrics(np.concatenate(truths), np.concatenate(predictions))
    return {
        "cv_R2_mean": float(metric_frame["R2"].mean()),
        "cv_R2_std": float(metric_frame["R2"].std(ddof=1) or 0.0),
        "cv_R2_pooled": pooled["R2"],
        "cv_RMSE_mean": float(metric_frame["RMSE"].mean()),
        "cv_RMSE_pooled": pooled["RMSE"],
        "cv_folds": len(fold_metrics),
    }


def _add_nested_aggregates(rows, common, fold_rows, truths, predictions, protocol):
    """Add mean and pooled outer-holdout rows, retaining both R² definitions."""
    if protocol != "loso" or len(fold_rows) < 2:
        return
    frame = pd.DataFrame(fold_rows)
    mean_metrics = {name: float(frame[name].mean()) for name in METRIC_NAMES}
    mean_metrics["R2_std"] = float(frame["R2"].std(ddof=1) or 0.0)
    for name in ["cv_R2_mean", "cv_R2_std", "cv_R2_pooled", "cv_RMSE_mean", "cv_RMSE_pooled"]:
        mean_metrics[name] = float(frame[name].mean())
    rows.append(
        {
            **common,
            "group": "mean",
            "status": "mean",
            "n_folds": len(fold_rows),
            "n_test": int(sum(len(values) for values in truths)),
            **mean_metrics,
        }
    )
    pooled = regression_metrics(np.concatenate(truths), np.concatenate(predictions))
    rows.append(
        {
            **common,
            "group": "pooled",
            "status": "mean_pooled",
            "n_folds": len(fold_rows),
            "n_test": int(sum(len(values) for values in truths)),
            **pooled,
            **{
                name: float(frame[name].mean())
                for name in [
                    "cv_R2_mean",
                    "cv_R2_std",
                    "cv_R2_pooled",
                    "cv_RMSE_mean",
                    "cv_RMSE_pooled",
                ]
            },
        }
    )


def evaluate_nested_cv(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate outer Phase-2 holdouts with inner grouped CV on training rows."""
    eem, samples, _ = load_data(args.data)
    catalog = experiment_catalog()
    unknown = [name for name in args.experiments if name not in catalog]
    if unknown:
        raise ValueError(f"unknown experiments: {', '.join(unknown)}")
    fold_specs, splits = _fold_specs(args, samples)
    experiments = args.experiments
    feature_cache, mask_rows = _nested_feature_cache(
        args, eem, samples, fold_specs, experiments
    )
    rows = []
    predictions = []
    for target_name in args.targets:
        target = resolve_column(target_name)
        y = pd.to_numeric(samples[target], errors="coerce").to_numpy(dtype=float)

        for baseline_name in ("baseline_global_mean", "baseline_station_mean"):
            fold_metrics, truths, fold_predictions = [], [], []
            for spec in fold_specs:
                train = spec["train"]
                test = spec["test"]
                train_valid = np.isfinite(y[train])
                test_valid = np.isfinite(y[test])
                train, test = train[train_valid], test[test_valid]
                if len(train) < 2 or len(test) < 2:
                    continue
                global_mean = float(np.mean(y[train]))
                prediction = np.full(len(test), global_mean, dtype=float)
                if baseline_name == "baseline_station_mean" and "Point" in samples:
                    means = pd.DataFrame(
                        {"Point": samples.iloc[train]["Point"].to_numpy(), "target": y[train]}
                    ).groupby("Point")["target"].mean()
                    prediction = (
                        pd.Series(samples.iloc[test]["Point"].to_numpy())
                        .map(means)
                        .fillna(global_mean)
                        .to_numpy(dtype=float)
                    )
                metrics = regression_metrics(y[test], prediction)
                common = {
                    "target": target_name,
                    "experiment": baseline_name,
                    "model": "baseline",
                    "n_train": len(train),
                    "n_test": len(test),
                    "_test_indices": test,
                }
                _add_fold_result(
                    rows,
                    predictions,
                    common,
                    "baseline",
                    metrics,
                    y[test],
                    prediction,
                    spec["group"],
                )
                fold_metrics.append(metrics)
                truths.append(y[test])
                fold_predictions.append(prediction)
            _add_aggregates(
                rows,
                {
                    "target": target_name,
                    "experiment": baseline_name,
                    "model": "baseline",
                },
                fold_metrics,
                truths,
                fold_predictions,
                args.protocol,
                "baseline_mean",
            )

        for experiment_name in experiments:
            for model_name in args.models:
                fold_rows, truths, fold_predictions = [], [], []
                for spec in fold_specs:
                    cached = feature_cache[experiment_name][spec["group"]]
                    train, test = cached["train"], cached["test"]
                    train_valid, test_valid = np.isfinite(y[train]), np.isfinite(y[test])
                    train, test = train[train_valid], test[test_valid]
                    if len(train) < 2 or len(test) < 2:
                        continue
                    cv_scores = _nested_cv_score(cached, y, model_name, args)
                    model = make_model(model_name, args.seed, args.n_jobs)
                    model.fit(cached["x_train"][train_valid], y[train])
                    prediction = np.asarray(
                        model.predict(cached["x_test"][test_valid]), dtype=float
                    )
                    metrics = regression_metrics(y[test], prediction)
                    common = {
                        "target": target_name,
                        "experiment": experiment_name,
                        "model": model_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_features": cached["x_train"].shape[1],
                        "_test_indices": test,
                        **cv_scores,
                    }
                    _add_fold_result(
                        rows,
                        predictions,
                        common,
                        "holdout" if args.protocol == "two_way" else "ok",
                        metrics,
                        y[test],
                        prediction,
                        spec["group"],
                    )
                    fold_rows.append({**common, **metrics})
                    truths.append(y[test])
                    fold_predictions.append(prediction)
                _add_nested_aggregates(
                    rows,
                    {"target": target_name, "experiment": experiment_name, "model": model_name},
                    fold_rows,
                    truths,
                    fold_predictions,
                    args.protocol,
                )

    frame = pd.DataFrame(rows)
    if "_test_indices" in frame:
        frame = frame.drop(columns=["_test_indices"])
    frame = _add_baseline_deltas(frame, args.protocol)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "holdout_metrics.csv", index=False)
    prediction_frame.to_csv(output / "holdout_predictions.csv", index=False)
    if mask_rows:
        pd.DataFrame(mask_rows).to_csv(output / "pca_mask_summary.csv", index=False)
    _write_split_artifact(output, samples, splits)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str, ensure_ascii=False) + "\n"
    )
    return frame, prediction_frame


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    eem, samples, _ = load_data(args.data)
    catalog = experiment_catalog()
    unknown = [name for name in args.experiments if name not in catalog]
    if unknown:
        raise ValueError(f"unknown experiments: {', '.join(unknown)}")
    fold_specs, splits = _fold_specs(args, samples)

    # Fit each representation independently inside each held-out fold.  In
    # particular, FeatureBuilder removes all-zero EEM columns before PCA.
    feature_cache = {}
    mask_rows = []
    for experiment_name in args.experiments:
        experiment = catalog[experiment_name]
        cached_folds = {}
        for spec in fold_specs:
            builder = FeatureBuilder(
                experiment,
                "__phase2__",
                args.pca_components,
                args.pf_rank,
                False,
                args.pf_max_iter,
                args.seed,
            )
            x_train = builder.fit_transform(eem[spec["train"]], samples.iloc[spec["train"]])
            x_test = builder.transform(eem[spec["test"]], samples.iloc[spec["test"]])
            cached_folds[spec["group"]] = (x_train, x_test)
            if experiment.eem != "none":
                mask_rows.append(
                    {
                        "experiment": experiment_name,
                        "group": spec["group"],
                        "raw_features": int(np.prod(eem.shape[1:])),
                        "features_after_zero_mask": int(builder.eem_feature_mask_.sum()),
                        "zero_masked_features": int((~builder.eem_feature_mask_).sum()),
                    }
                )
        feature_cache[experiment_name] = cached_folds

    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for target_name in args.targets:
        target = resolve_column(target_name)
        y = pd.to_numeric(samples[target], errors="coerce").to_numpy(dtype=float)

        for baseline_name in ("baseline_global_mean", "baseline_station_mean"):
            fold_metrics, truths, fold_predictions = [], [], []
            for spec in fold_specs:
                train = spec["train"]
                test = spec["test"]
                train_valid = np.isfinite(y[train])
                test_valid = np.isfinite(y[test])
                train = train[train_valid]
                test = test[test_valid]
                if len(train) < 2 or len(test) < 2:
                    continue
                global_mean = float(np.mean(y[train]))
                prediction = np.full(len(test), global_mean, dtype=float)
                if baseline_name == "baseline_station_mean" and "Point" in samples:
                    fit_frame = pd.DataFrame(
                        {"Point": samples.iloc[train]["Point"].to_numpy(), "target": y[train]}
                    )
                    means = fit_frame.groupby("Point")["target"].mean()
                    prediction = (
                        pd.Series(samples.iloc[test]["Point"].to_numpy())
                        .map(means)
                        .fillna(global_mean)
                        .to_numpy(dtype=float)
                    )
                metrics = regression_metrics(y[test], prediction)
                common = {
                    "target": target_name,
                    "experiment": baseline_name,
                    "model": "baseline",
                    "n_train": len(train),
                    "n_test": len(test),
                    "_test_indices": test,
                }
                _add_fold_result(
                    rows,
                    predictions,
                    common,
                    "baseline",
                    metrics,
                    y[test],
                    prediction,
                    spec["group"],
                )
                fold_metrics.append(metrics)
                truths.append(y[test])
                fold_predictions.append(prediction)
            _add_aggregates(
                rows,
                {
                    "target": target_name,
                    "experiment": baseline_name,
                    "model": "baseline",
                },
                fold_metrics,
                truths,
                fold_predictions,
                args.protocol,
                "baseline_mean",
            )

        for experiment_name in args.experiments:
            experiment = catalog[experiment_name]
            if target in experiment.tabular:
                rows.append(
                    {
                        "target": target_name,
                        "experiment": experiment_name,
                        "model": "",
                        "group": "",
                        "status": "skipped: target used as feature",
                    }
                )
                continue
            for model_name in args.models:
                fold_metrics, truths, fold_predictions = [], [], []
                for spec in fold_specs:
                    train_all = spec["train"]
                    test_all = spec["test"]
                    train_valid = np.isfinite(y[train_all])
                    test_valid = np.isfinite(y[test_all])
                    train = train_all[train_valid]
                    test = test_all[test_valid]
                    if len(train) < 2 or len(test) < 2:
                        continue
                    x_train_all, x_test_all = feature_cache[experiment_name][spec["group"]]
                    x_train = x_train_all[train_valid]
                    x_test = x_test_all[test_valid]
                    model = make_model(model_name, args.seed, args.n_jobs)
                    model.fit(x_train, y[train])
                    prediction = np.asarray(model.predict(x_test), dtype=float)
                    metrics = regression_metrics(y[test], prediction)
                    common = {
                        "target": target_name,
                        "experiment": experiment_name,
                        "model": model_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_features": x_train.shape[1],
                        "_test_indices": test,
                    }
                    _add_fold_result(
                        rows,
                        predictions,
                        common,
                        "holdout" if args.protocol == "two_way" else "ok",
                        metrics,
                        y[test],
                        prediction,
                        spec["group"],
                    )
                    fold_metrics.append(metrics)
                    truths.append(y[test])
                    fold_predictions.append(prediction)
                _add_aggregates(
                    rows,
                    {"target": target_name, "experiment": experiment_name, "model": model_name},
                    fold_metrics,
                    truths,
                    fold_predictions,
                    args.protocol,
                    "mean",
                )

    # Internal helper columns should never be written to the public CSV.
    frame = pd.DataFrame(rows)
    if "_test_indices" in frame:
        frame = frame.drop(columns=["_test_indices"])
    frame = _add_baseline_deltas(frame, args.protocol)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "holdout_metrics.csv", index=False)
    prediction_frame.to_csv(output / "holdout_predictions.csv", index=False)
    if mask_rows:
        pd.DataFrame(mask_rows).to_csv(output / "pca_mask_summary.csv", index=False)
    _write_split_artifact(output, samples, splits)
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str, ensure_ascii=False) + "\n"
    )
    return frame, prediction_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=["loso", "two_way"], default="loso")
    parser.add_argument("--targets", nargs="+", default=["BOD", "COD", "TOC", "BOD_COD"])
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["SS_EC", "Temp_pH", "EEMpca", "EEMpca_SS_EC"],
    )
    parser.add_argument(
        "--models", nargs="+", choices=["linear", "xgboost", "tree", "svr"],
        default=["linear"],
    )
    parser.add_argument("--group-col", choices=["Point", "Month"], default="Point")
    parser.add_argument("--secondary-group-col", default="Month")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--pf-rank", type=int, default=4)
    parser.add_argument("--pf-max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    frame, _ = evaluate(args)
    if args.protocol == "loso":
        summary = frame[frame["status"].isin(["mean", "baseline_mean"])]
    else:
        summary = frame[frame["status"].isin(["holdout", "baseline"])]
    if not summary.empty:
        print(summary[["target", "experiment", "model", "group", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
