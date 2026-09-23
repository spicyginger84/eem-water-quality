"""Classical regression benchmark pipeline.

Every requested target/feature/model combination is evaluated independently.
No held-out fold is used for model selection. Random and group splits use
cross-validation by default; two-way, LOSO and LOMO are holdout protocols.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..artifacts import start_run, target_directory_name, write_json
from ..data import load_data, resolve_column
from ..evaluation.baselines import baseline_predictions
from ..evaluation.metrics import regression_metrics
from ..evaluation.protocols import build_evaluation_splits
from ..features import FeatureBuilder, experiment_catalog
from ..models.classical import make_model

LOGGER = logging.getLogger(__name__)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def _prediction_frame(samples, indices, truth, prediction, common):
    frame = samples.iloc[indices].copy()
    frame.insert(0, "row_position", np.asarray(indices, dtype=int))
    for key, value in common.items():
        frame[key] = [value] * len(frame)
    frame["y_true"] = np.asarray(truth, dtype=float)
    frame["y_pred"] = np.asarray(prediction, dtype=float)
    return frame


def _metric_row(common, truth, prediction):
    return {**common, **regression_metrics(truth, prediction)}


def _aggregate_rows(rows, predictions):
    """Add mean and pooled rows for each target/features/model/partition."""
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    fold_frame = pd.DataFrame(rows)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    summary_rows = []
    key_columns = ["target", "features", "model", "protocol", "partition"]
    for key, group in fold_frame.groupby(key_columns, dropna=False):
        target, features, model, protocol, partition = key
        metric_columns = ["R2", "MSE", "RMSE", "MAE", "MAPE"]
        mean_metrics = {
            metric: float(group[metric].mean())
            for metric in metric_columns
            if metric in group and group[metric].notna().any()
        }
        r2_mean = mean_metrics.get("R2", np.nan)
        r2_std = float(group["R2"].std(ddof=1) or 0.0) if len(group) > 1 else 0.0
        summary_rows.append(
            {
                "target": target,
                "features": features,
                "model": model,
                "protocol": protocol,
                "partition": partition,
                "fold": "mean",
                "heldout_group": "mean",
                "n_folds": len(group),
                "n_test": int(group["n_test"].sum()),
                **mean_metrics,
                "R2_mean": r2_mean,
                "R2_std": r2_std,
            }
        )
        if prediction_frame.empty:
            continue
        mask = np.ones(len(prediction_frame), dtype=bool)
        for column, value in zip(key_columns, key):
            mask &= prediction_frame[column].eq(value).to_numpy()
        pooled_prediction = prediction_frame.loc[mask]
        if pooled_prediction.empty:
            continue
        pooled_metrics = regression_metrics(
            pooled_prediction["y_true"], pooled_prediction["y_pred"]
        )
        summary_rows.append(
            {
                "target": target,
                "features": features,
                "model": model,
                "protocol": protocol,
                "partition": partition,
                "fold": "pooled",
                "heldout_group": "pooled",
                "n_folds": len(group),
                "n_test": len(pooled_prediction),
                "R2_mean": r2_mean,
                "R2_std": r2_std,
                "R2_pooled": pooled_metrics["R2"],
                **pooled_metrics,
            }
        )
    return fold_frame, _add_baseline_deltas(pd.DataFrame(summary_rows))


def _add_baseline_deltas(frame):
    if frame.empty or "R2" not in frame:
        return frame
    frame = frame.copy()
    baselines = frame[frame["features"] == "baseline_global_mean"]
    lookup_columns = ["target", "protocol", "partition", "fold"]
    lookup = baselines.set_index(lookup_columns)["R2"]

    def delta(row):
        key = tuple(row[column] for column in lookup_columns)
        baseline = lookup.get(key, np.nan)
        return row["R2"] - baseline if pd.notna(baseline) and pd.notna(row["R2"]) else np.nan

    frame["delta_R2_vs_global"] = frame.apply(delta, axis=1)
    return frame


def _write_model_artifact(root, target_name, feature_name, model_name, fold, builder, model):
    model_dir = (
        Path(root)
        / "models"
        / _safe_name(target_name)
        / _safe_name(feature_name)
        / _safe_name(model_name)
        / f"fold_{fold:02d}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"features": builder, "model": model, "target": target_name},
        model_dir / "model.joblib",
    )
    write_json(
        model_dir / "feature_schema.json",
        {
            "target": target_name,
            "features": feature_name,
            "model": model_name,
            "fold": fold,
            "n_features": getattr(builder, "n_features_", None),
            "eem_features_after_mask": (
                int(builder.eem_feature_mask_.sum())
                if hasattr(builder, "eem_feature_mask_")
                else None
            ),
        },
    )


def _evaluate_target(
    args,
    output_root,
    protocol,
    split_records,
    eem,
    samples,
    y,
    target,
    target_name,
    features,
    model_names,
    log_target,
):
    rows = []
    predictions = []
    catalog = experiment_catalog()
    total_folds = len(split_records)
    for spec in split_records:
        fold = spec["fold"]
        train_indices = np.asarray(spec["train"], dtype=int)
        train_indices = train_indices[np.isfinite(y[train_indices])]
        if len(train_indices) < 2:
            LOGGER.warning(
                "target=%s fold=%s/%s skipped: fewer than two finite training targets",
                target_name,
                fold,
                total_folds,
            )
            continue

        partitions = []
        if protocol == "single_split":
            validation = np.asarray(spec.get("validation", []), dtype=int)
            validation = validation[np.isfinite(y[validation])]
            if len(validation) >= 2:
                partitions.append(("validation", validation))
        test_indices = np.asarray(spec.get("test", []), dtype=int)
        test_indices = test_indices[np.isfinite(y[test_indices])]
        if len(test_indices) >= 2:
            partitions.append(("test", test_indices))
        if not partitions:
            LOGGER.warning(
                "target=%s fold=%s/%s skipped: no evaluation partition with two finite targets",
                target_name,
                fold,
                total_folds,
            )
            continue
        heldout_group = spec.get("heldout_group", "")
        LOGGER.info(
            "target=%s fold=%s/%s train=%d eval=%s heldout=%s",
            target_name,
            fold,
            total_folds,
            len(train_indices),
            ",".join(f"{name}:{len(indices)}" for name, indices in partitions),
            heldout_group or "-",
        )

        for partition, eval_indices in partitions:
            common_base = {
                "target": target_name,
                "protocol": protocol,
                "partition": partition,
                "fold": fold,
                "heldout_group": heldout_group,
                "n_train": len(train_indices),
                "n_test": len(eval_indices),
            }
            baseline_values = baseline_predictions(samples, y, train_indices, eval_indices, "Point")
            for feature_name, prediction in zip(
                ["baseline_global_mean", "baseline_station_mean"], baseline_values
            ):
                common = {**common_base, "features": feature_name, "model": "baseline", "status": "baseline"}
                rows.append(_metric_row(common, y[eval_indices], prediction))
                predictions.append(
                    _prediction_frame(samples, eval_indices, y[eval_indices], prediction, common)
                )

        for feature_name in features:
            feature_started = time.perf_counter()
            feature = catalog[feature_name]
            if target in feature.tabular:
                for partition, eval_indices in partitions:
                    rows.append(
                        {
                            "target": target_name,
                            "features": feature_name,
                            "model": "",
                            "protocol": protocol,
                            "partition": partition,
                            "fold": fold,
                            "heldout_group": heldout_group,
                            "n_train": len(train_indices),
                            "n_test": len(eval_indices),
                            "status": f"skipped: target {target_name} is used as a feature",
                        }
                    )
                LOGGER.warning(
                    "target=%s fold=%s feature=%s skipped: target is used as a feature",
                    target_name,
                    fold,
                    feature_name,
                )
                continue
            builder = FeatureBuilder(
                feature,
                target,
                args.pca_components,
                args.seed,
            )
            x_train = builder.fit_transform(eem[train_indices], samples.iloc[train_indices])
            builder.n_features_ = x_train.shape[1]
            eval_features = {
                partition: builder.transform(eem[eval_indices], samples.iloc[eval_indices])
                for partition, eval_indices in partitions
            }
            LOGGER.info(
                "target=%s fold=%s feature=%s fitted n_features=%d elapsed=%.2fs",
                target_name,
                fold,
                feature_name,
                x_train.shape[1],
                time.perf_counter() - feature_started,
            )
            for model_name in model_names:
                model_started = time.perf_counter()
                LOGGER.info(
                    "target=%s fold=%s feature=%s model=%s fitting",
                    target_name,
                    fold,
                    feature_name,
                    model_name,
                )
                model = make_model(model_name, args.seed, args.n_jobs, log_target)
                model.fit(x_train, y[train_indices])
                _write_model_artifact(
                    output_root, target_name, feature_name, model_name, fold, builder, model
                )
                model_scores = []
                for partition, eval_indices in partitions:
                    prediction = np.asarray(model.predict(eval_features[partition]), dtype=float)
                    common = {
                        "target": target_name,
                        "features": feature_name,
                        "model": model_name,
                        "protocol": protocol,
                        "partition": partition,
                        "fold": fold,
                        "heldout_group": heldout_group,
                        "n_train": len(train_indices),
                        "n_test": len(eval_indices),
                        "status": "ok",
                        "n_features": x_train.shape[1],
                    }
                    metric_row = _metric_row(common, y[eval_indices], prediction)
                    rows.append(metric_row)
                    model_scores.append(f"{partition}:R2={metric_row['R2']:.4f}")
                    predictions.append(
                        _prediction_frame(samples, eval_indices, y[eval_indices], prediction, common)
                    )
                LOGGER.info(
                    "target=%s fold=%s feature=%s model=%s complete %s elapsed=%.2fs",
                    target_name,
                    fold,
                    feature_name,
                    model_name,
                    " ".join(model_scores),
                    time.perf_counter() - model_started,
                )
    fold_frame, summary_frame = _aggregate_rows(rows, predictions)
    return _add_baseline_deltas(fold_frame), summary_frame, predictions


def run_ml(args):
    started = time.perf_counter()
    LOGGER.info("Loading processed data from %s", args.data)
    eem, samples, _ = load_data(args.data)
    protocol, split_records = build_evaluation_splits(samples, args)
    LOGGER.info(
        "Run protocol=%s samples=%d eem_shape=%s folds=%d",
        protocol,
        len(samples),
        tuple(eem.shape),
        len(split_records),
    )
    features = list(args.features or ["EEMpca"])
    model_names = list(args.models)
    LOGGER.info(
        "Run configuration targets=%s features=%s models=%s",
        ",".join(args.targets),
        ",".join(features),
        ",".join(model_names),
    )
    output = start_run(args, samples, split_records)
    LOGGER.info("Writing artifacts to %s", output)
    root_folds = []
    root_summaries = []
    for target_name in args.targets:
        target = resolve_column(target_name)
        if target not in samples:
            raise ValueError(f"Target column {target_name!r} is absent from processed data")
        y = pd.to_numeric(samples[target], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(y).sum() < 2:
            raise ValueError(f"{target_name}: fewer than two finite target values")
        log_target = target_name in args.log_targets or target in args.log_targets
        if log_target and np.any(y[np.isfinite(y)] <= -1):
            raise ValueError("log1p targets must be greater than -1")
        destination = output / target_directory_name(target_name)
        destination.mkdir()
        LOGGER.info("Starting target=%s", target_name)
        fold_frame, summary_frame, predictions = _evaluate_target(
            args,
            output,
            protocol,
            split_records,
            eem,
            samples,
            y,
            target,
            target_name,
            features,
            model_names,
            log_target,
        )
        write_json(
            destination / "target.json",
            {
                "target": target,
                "features": features,
                "models": model_names,
                "protocol": protocol,
                "split": getattr(args, "split", None),
                "excluded_nonfinite_targets": int((~np.isfinite(y)).sum()),
            },
        )
        prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        if protocol == "cv":
            fold_frame.to_csv(destination / "cv_fold_metrics.csv", index=False)
            summary_frame.to_csv(destination / "cv_summary.csv", index=False)
            prediction_frame.to_csv(destination / "cv_predictions.csv", index=False)
        elif protocol in {"loso", "lomo"}:
            fold_frame.to_csv(destination / "holdout_metrics.csv", index=False)
            summary_frame.to_csv(destination / "holdout_summary.csv", index=False)
            prediction_frame.to_csv(destination / "holdout_predictions.csv", index=False)
        elif protocol == "single_split":
            for partition in ("validation", "test"):
                fold_frame[fold_frame["partition"] == partition].to_csv(
                    destination / f"{partition}_metrics.csv", index=False
                )
                summary_frame[summary_frame["partition"] == partition].to_csv(
                    destination / f"{partition}_summary.csv", index=False
                )
                prediction_frame[prediction_frame["partition"] == partition].to_csv(
                    destination / f"{partition}_predictions.csv", index=False
                )
        else:
            fold_frame.to_csv(destination / "holdout_metrics.csv", index=False)
            summary_frame.to_csv(destination / "holdout_summary.csv", index=False)
            prediction_frame.to_csv(destination / "holdout_predictions.csv", index=False)
        root_folds.append(fold_frame)
        root_summaries.append(summary_frame)
        print(
            f"{target_name}: protocol={protocol}, folds={len(split_records)}, "
            f"features={len(features)}, models={len(model_names)}",
            flush=True,
        )
    if root_folds:
        pd.concat(root_folds, ignore_index=True).to_csv(output / "fold_metrics.csv", index=False)
    if root_summaries:
        pd.concat(root_summaries, ignore_index=True).to_csv(output / "summary_metrics.csv", index=False)
    LOGGER.info("Run complete output=%s elapsed=%.2fs", output, time.perf_counter() - started)
    return output
