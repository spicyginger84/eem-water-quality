"""Phase-0 train/validation/test selection for classical experiments."""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .artifacts import save_predictions, start_run, target_directory_name, write_json
from .data import load_data, resolve_column, split_indices
from .features import Experiment, FeatureBuilder, experiment_catalog
from .metrics import regression_metrics


def make_model(name, seed=42, n_jobs=1, log_target=False):
    if name == "linear":
        model = LinearRegression(n_jobs=n_jobs)
    elif name == "tree":
        model = DecisionTreeRegressor(random_state=seed)
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError("Install the 'boosting' extra to run xgboost") from error
        model = XGBRegressor(random_state=seed, n_jobs=n_jobs)
    elif name == "svr":
        model = TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
            ),
            transformer=StandardScaler(),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    if log_target:
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model


def _baseline_predictions(samples, y, fit_indices, eval_indices, station_col="Point"):
    """Return global-mean and station-mean predictions fitted on fit rows only."""
    global_mean = float(np.mean(y[fit_indices]))
    global_prediction = np.full(len(eval_indices), global_mean, dtype=float)
    if station_col not in samples:
        return global_prediction, global_prediction.copy()
    grouped = pd.DataFrame(
        {
            "group": samples.iloc[fit_indices][station_col].to_numpy(),
            "target": y[fit_indices],
        }
    )
    group_means = grouped.groupby("group")["target"].mean()
    eval_groups = pd.Series(samples.iloc[eval_indices][station_col].to_numpy())
    group_prediction = eval_groups.map(group_means).fillna(global_mean).to_numpy(dtype=float)
    return global_prediction, group_prediction


def _add_baseline_deltas(frame):
    """Add deltas against the global baseline for validation and test rows."""
    if frame.empty:
        frame["delta_R2_vs_global"] = np.nan
        return frame
    validation_baseline = frame.loc[
        (frame["experiment"] == "baseline_global_mean")
        & (frame["status"] == "baseline_validation"),
        ["target", "R2"],
    ].set_index("target")["R2"]
    test_baseline = frame.loc[
        (frame["experiment"] == "baseline_global_mean")
        & (frame["status"] == "baseline"),
        ["target", "R2"],
    ].set_index("target")["R2"]

    def delta(row):
        if pd.isna(row.get("R2")):
            return np.nan
        baseline = validation_baseline if row.get("status") in {"ok", "baseline_validation"} else test_baseline
        return row["R2"] - baseline[row["target"]] if row["target"] in baseline else np.nan

    frame["delta_R2_vs_global"] = frame.apply(delta, axis=1)
    return frame


def _run_phase0_target(
    args,
    eem,
    samples,
    y,
    target,
    target_name,
    train_indices,
    validation_indices,
    test_indices,
    experiments,
    model_names,
    standalone,
    log_target,
):
    """Fit candidates on train, select on validation, then score untouched test."""
    validation_rows = []
    validation_candidates = []

    validation_baselines = _baseline_predictions(
        samples, y, train_indices, validation_indices, "Point"
    )
    for name, prediction in zip(
        ["baseline_global_mean", "baseline_station_mean"], validation_baselines
    ):
        validation_rows.append(
            {
                "target": target_name,
                "experiment": name,
                "model": "baseline",
                "fold": "validation",
                "status": "baseline_validation",
                "n_train": len(train_indices),
                "n_validation": len(validation_indices),
                **regression_metrics(y[validation_indices], prediction),
            }
        )

    for experiment in experiments:
        if target in experiment.tabular:
            validation_rows.append(
                {
                    "target": target_name,
                    "experiment": experiment.name,
                    "model": "",
                    "fold": "",
                    "status": f"skipped: target {target_name} is used as a feature",
                }
            )
            continue
        if experiment.eem == "none" and not experiment.parafac and not experiment.tabular:
            validation_rows.append(
                {
                    "target": target_name,
                    "experiment": experiment.name,
                    "model": "",
                    "fold": "",
                    "status": "skipped: no non-target features",
                }
            )
            continue

        builder = FeatureBuilder(
            experiment,
            target,
            args.pca_components,
            args.pf_rank,
            standalone,
            args.pf_max_iter,
            args.seed,
        )
        x_train = builder.fit_transform(eem[train_indices], samples.iloc[train_indices])
        x_validation = builder.transform(eem[validation_indices], samples.iloc[validation_indices])
        for model_name in model_names:
            model = make_model(model_name, args.seed, args.n_jobs, log_target)
            model.fit(x_train, y[train_indices])
            prediction = np.asarray(model.predict(x_validation), dtype=float)
            validation_row = {
                "target": target_name,
                "experiment": experiment.name,
                "model": model_name,
                "fold": "validation",
                "status": "ok",
                "n_train": len(train_indices),
                "n_validation": len(validation_indices),
                "n_features": x_train.shape[1],
                **regression_metrics(y[validation_indices], prediction),
            }
            validation_rows.append(validation_row)
            validation_candidates.append(
                {
                    "experiment": experiment,
                    "experiment_name": experiment.name,
                    "model_name": model_name,
                    "validation_row": validation_row,
                    "validation_prediction": pd.DataFrame(
                        {
                            "target": target_name,
                            "experiment": experiment.name,
                            "model": model_name,
                            "fold": "validation",
                            "row_position": validation_indices,
                            "y_true": y[validation_indices],
                            "y_pred": prediction,
                        }
                    ),
                }
            )
            del model
        del builder, x_train, x_validation

    if not validation_candidates:
        raise ValueError(f"No runnable experiments remain for {target_name}")

    selected = min(validation_candidates, key=lambda item: item["validation_row"]["RMSE"])
    selected_exp_name = selected["experiment_name"]
    selected_model_name = selected["model_name"]
    selected_validation_row = selected["validation_row"]
    selected_validation_prediction = selected["validation_prediction"]

    # Refit every candidate on train+validation so test_metrics.csv contains a
    # fair, directly comparable test score for every experiment/model. Only the
    # validation winner is retained as the deployable model artifact.
    refit_indices = np.concatenate([train_indices, validation_indices])
    test_rows = []
    selected_builder = None
    selected_model = None
    selected_test_prediction = None
    selected_test_metrics = None
    experiment_names = list(dict.fromkeys(item["experiment_name"] for item in validation_candidates))
    for experiment_name in experiment_names:
        experiment = next(
            item["experiment"] for item in validation_candidates if item["experiment_name"] == experiment_name
        )
        builder = FeatureBuilder(
            experiment,
            target,
            args.pca_components,
            args.pf_rank,
            standalone,
            args.pf_max_iter,
            args.seed,
        )
        x_refit = builder.fit_transform(eem[refit_indices], samples.iloc[refit_indices])
        x_test = builder.transform(eem[test_indices], samples.iloc[test_indices])
        candidate_models = [
            item["model_name"]
            for item in validation_candidates
            if item["experiment_name"] == experiment_name
        ]
        selected_in_experiment = False
        for model_name in candidate_models:
            model = make_model(model_name, args.seed, args.n_jobs, log_target)
            model.fit(x_refit, y[refit_indices])
            prediction = np.asarray(model.predict(x_test), dtype=float)
            metrics = regression_metrics(y[test_indices], prediction)
            is_selected = experiment_name == selected_exp_name and model_name == selected_model_name
            test_rows.append(
                {
                    "target": target_name,
                    "experiment": experiment_name,
                    "model": model_name,
                    "status": "ok",
                    "selected": is_selected,
                    "n_train": len(refit_indices),
                    "n_test": len(test_indices),
                    **metrics,
                }
            )
            if is_selected:
                selected_builder = builder
                selected_model = model
                selected_test_prediction = prediction
                selected_test_metrics = dict(metrics)
                selected_in_experiment = True
            else:
                del model
        if not selected_in_experiment:
            del builder, x_refit, x_test

    baseline_test_predictions = _baseline_predictions(
        samples, y, refit_indices, test_indices, "Point"
    )
    for name, prediction in zip(
        ["baseline_global_mean", "baseline_station_mean"], baseline_test_predictions
    ):
        test_rows.append(
            {
                "target": target_name,
                "experiment": name,
                "model": "baseline",
                "status": "baseline",
                "selected": False,
                "n_train": len(refit_indices),
                "n_test": len(test_indices),
                **regression_metrics(y[test_indices], prediction),
            }
        )

    test_frame = pd.DataFrame(test_rows)
    baseline_test_r2 = test_frame.loc[
        test_frame["experiment"] == "baseline_global_mean", "R2"
    ].iloc[0]
    test_frame["delta_R2_vs_global"] = test_frame["R2"] - baseline_test_r2
    if selected_builder is None or selected_model is None or selected_test_metrics is None:
        raise RuntimeError("Selected model was not refitted")
    selected_test_metrics["delta_R2_vs_global"] = selected_test_metrics["R2"] - baseline_test_r2

    validation_frame = _add_baseline_deltas(pd.DataFrame(validation_rows))
    return {
        "validation_frame": validation_frame,
        "selected_validation_row": selected_validation_row,
        "selected_validation_prediction": selected_validation_prediction,
        "test_frame": test_frame,
        "selected_builder": selected_builder,
        "selected_model": selected_model,
        "selected_test_prediction": selected_test_prediction,
        "selected_test_metrics": selected_test_metrics,
        "selected_exp": selected["experiment"],
    }


def run_ml(args):
    eem, samples, _ = load_data(args.data)
    group_col = getattr(args, "group_col", "Point")
    splits = split_indices(
        samples,
        args.split,
        group_col,
        args.seed,
        args.test_size,
        args.val_size,
        getattr(args, "secondary_group_col", "Month"),
    )
    catalog = experiment_catalog()
    standalone = args.command == "parafac"
    experiments = (
        [Experiment("nonnegative_PARAFAC_SVR", parafac=True)]
        if standalone
        else [catalog[name] for name in (args.experiments or list(catalog))]
    )
    model_names = ["svr"] if standalone else args.models
    for name in model_names:
        make_model(name, args.seed, args.n_jobs)
    output = start_run(args, samples, splits)
    summaries = []
    for target_name in args.targets:
        target = resolve_column(target_name)
        y = pd.to_numeric(samples[target], errors="coerce").to_numpy(dtype=float)
        train_indices = splits["train"][np.isfinite(y[splits["train"]])]
        validation_indices = splits["validation"][np.isfinite(y[splits["validation"]])]
        test_indices = splits["test"][np.isfinite(y[splits["test"]])]
        if len(train_indices) < 2:
            raise ValueError(f"{target_name}: training partition needs at least two finite targets")
        if len(validation_indices) < 2:
            raise ValueError(f"{target_name}: validation partition needs at least two finite targets")
        if len(test_indices) < 2:
            raise ValueError(f"{target_name}: test partition needs at least two finite targets")
        log_target = target_name in args.log_targets or target in args.log_targets
        if log_target and np.any(y[np.isfinite(y)] <= -1):
            raise ValueError("log1p targets must be greater than -1")
        destination = output / target_directory_name(target_name)
        destination.mkdir()
        write_json(
            destination / "target.json",
            {
                "target": target,
                "outer_holdout": args.split,
                "selection": "independent_validation",
                "indices": splits,
                "excluded_nonfinite_targets": int((~np.isfinite(y)).sum()),
            },
        )
        result = _run_phase0_target(
            args,
            eem,
            samples,
            y,
            target,
            target_name,
            train_indices,
            validation_indices,
            test_indices,
            experiments,
            model_names,
            standalone,
            log_target,
        )
        result["validation_frame"].to_csv(destination / "validation_metrics.csv", index=False)
        result["test_frame"].to_csv(destination / "test_metrics.csv", index=False)
        save_predictions(
            destination / "validation_predictions.csv",
            samples,
            validation_indices,
            y[validation_indices],
            result["selected_validation_prediction"]["y_pred"].to_numpy(),
        )
        write_json(
            destination / "selected.json",
            {
                "target": target,
                "validation": result["selected_validation_row"],
                "test": result["selected_test_metrics"],
                "model_selection": {
                    "method": "independent_validation",
                    "test_used_for_selection": False,
                },
                "model_parameters": repr(result["selected_model"].get_params(deep=True)),
            },
        )
        save_predictions(
            destination / "test_predictions.csv",
            samples,
            test_indices,
            y[test_indices],
            result["selected_test_prediction"],
        )
        joblib.dump(
            {
                "features": result["selected_builder"],
                "model": result["selected_model"],
                "target": target,
            },
            destination / "model.joblib",
        )
        builder = result["selected_builder"]
        if hasattr(builder, "parafac_"):
            pf = builder.parafac_
            np.savez(
                destination / "parafac_loadings.npz",
                emission=pf.em_loadings_,
                excitation=pf.ex_loadings_,
            )
            write_json(
                destination / "parafac_diagnostics.json",
                {
                    "train_relative_error": pf.train_relative_error_,
                    "test_relative_error": pf.relative_error(
                        eem[test_indices], pf.transform(eem[test_indices])
                    ),
                },
            )
        summaries.extend(result["test_frame"].to_dict("records"))
        print(
            f"{target_name}: selected {result['selected_exp'].name}/"
            f"{result['selected_model'].__class__.__name__} by independent validation; "
            f"test R2={result['selected_test_metrics']['R2']:.6g}",
            flush=True,
        )
    pd.DataFrame(summaries).to_csv(output / "test_metrics.csv", index=False)
    return output
