"""Validation-based selection across the notebook's classical experiments."""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .artifacts import save_predictions, start_run, write_json
from .data import load_data, resolve_column, split_indices, target_partitions
from .features import Experiment, FeatureBuilder, ParafacFeatures, experiment_catalog
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


def run_ml(args):
    eem, samples, _ = load_data(args.data)
    splits = split_indices(
        samples, args.split, args.group_col, args.seed, args.test_size, args.val_size
    )
    catalog = experiment_catalog()
    standalone = args.command == "parafac"
    experiments = (
        [Experiment("nonnegative_PARAFAC_SVR", parafac=True)]
        if standalone
        else [catalog[name] for name in (args.experiments or list(catalog))]
    )
    model_names = ["svr"] if standalone else args.models
    # Check optional dependencies before starting an expensive feature fit.
    for name in model_names:
        make_model(name, args.seed, args.n_jobs)
    output = start_run(args, samples, splits)
    summaries = []
    for target_number, target_name in enumerate(args.targets):
        target = resolve_column(target_name)
        y, parts = target_partitions(samples, target, splits)
        # Classical ML follows the original notebook's single holdout protocol:
        # combine the provisional train/validation rows for fitting and use the
        # held-out test rows for every model evaluation. Neural models retain a
        # separate validation split for early stopping in neural.py.
        train = np.concatenate([parts["train"], parts["validation"]])
        val = test = parts["test"]
        log_target = target_name in args.log_targets or target in args.log_targets
        if log_target and np.any(y[np.isfinite(y)] <= -1):
            raise ValueError("log1p targets must be greater than -1")
        destination = output / f"target_{target_number:02d}"
        destination.mkdir()
        write_json(
            destination / "target.json",
            {
                "target": target,
                "indices": parts,
                "excluded_nonfinite_targets": int((~np.isfinite(y)).sum()),
            },
        )
        pf_cache = None
        if any(exp.parafac for exp in experiments):
            pf = ParafacFeatures(args.pf_rank, standalone, args.pf_max_iter, seed=args.seed)
            pf_cache = pf, pf.fit_transform(eem[train])
        best = None
        rows = []
        test_rows = []
        for exp in experiments:
            builder = FeatureBuilder(
                exp,
                target,
                args.pca_components,
                args.pf_rank,
                standalone,
                args.pf_max_iter,
                args.seed,
            )
            if exp.eem == "none" and not exp.parafac and not builder.tabular_columns:
                rows.append({"experiment": exp.name, "status": "skipped: no non-target features"})
                continue
            x_train = builder.fit_transform(eem[train], samples.iloc[train], pf_cache)
            x_val = builder.transform(eem[val], samples.iloc[val])
            for name in model_names:
                model = make_model(name, args.seed, args.n_jobs, log_target)
                model.fit(x_train, y[train])
                prediction = model.predict(x_val)
                metrics = regression_metrics(y[val], prediction)
                test_prediction = model.predict(builder.transform(eem[test], samples.iloc[test]))
                test_result = regression_metrics(y[test], test_prediction)
                row = {
                    "experiment": exp.name,
                    "model": name,
                    "status": "ok",
                    "n_train": len(train),
                    "n_features": x_train.shape[1],
                    **metrics,
                }
                rows.append(row)
                test_rows.append(
                    {
                        "experiment": exp.name,
                        "model": name,
                        "status": "ok",
                        "n_train": len(train),
                        "n_test": len(test),
                        **test_result,
                    }
                )
                print(
                    f"{target_name} / {exp.name} / {name}: test RMSE={metrics['RMSE']:.6g}",
                    flush=True,
                )
                if best is None or metrics["RMSE"] < best[0]["RMSE"]:
                    best = row, builder, model, prediction
            pd.DataFrame(rows).to_csv(destination / "validation_metrics.csv", index=False)
            pd.DataFrame(test_rows).to_csv(destination / "test_metrics.csv", index=False)
        if best is None:
            raise ValueError("No runnable experiments remain for this target")
        row, builder, model, val_prediction = best
        prediction = model.predict(builder.transform(eem[test], samples.iloc[test]))
        test_metrics = regression_metrics(y[test], prediction)
        test_metrics_frame = pd.DataFrame(test_rows)
        test_metrics_frame["selected"] = (
            (test_metrics_frame["experiment"] == row["experiment"])
            & (test_metrics_frame["model"] == row["model"])
        )
        test_metrics_frame.to_csv(destination / "test_metrics.csv", index=False)
        write_json(
            destination / "selected.json",
            {
                "target": target,
                "validation": row,
                "test": test_metrics,
                "model_parameters": repr(model.get_params(deep=True)),
            },
        )
        save_predictions(destination / "test_predictions.csv", samples, test, y[test], prediction)
        save_predictions(
            destination / "validation_predictions.csv", samples, val, y[val], val_prediction
        )
        joblib.dump(
            {"features": builder, "model": model, "target": target}, destination / "model.joblib"
        )
        if hasattr(builder, "parafac_"):
            pf = builder.parafac_
            np.savez(
                destination / "parafac_loadings.npz",
                excitation=pf.ex_loadings_,
                emission=pf.em_loadings_,
            )
            write_json(
                destination / "parafac_diagnostics.json",
                {
                    "train_relative_error": pf.train_relative_error_,
                    "test_relative_error": pf.relative_error(eem[test], pf.transform(eem[test])),
                },
            )
        for test_row in test_rows:
            summaries.append({"target": target, **test_row, "selected": (
                test_row["experiment"] == row["experiment"]
                and test_row["model"] == row["model"]
            )})
    pd.DataFrame(summaries).to_csv(output / "test_metrics.csv", index=False)
    return output
