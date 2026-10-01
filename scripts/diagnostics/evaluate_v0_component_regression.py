"""Fold-local decomposition and regression CV for improved Gaussian components."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
XGB_PARAMS = {
    "n_estimators": 80, "max_depth": 2, "learning_rate": 0.05,
    "min_child_weight": 2, "subsample": 0.9, "colsample_bytree": 1.,
    "reg_lambda": 2., "objective": "reg:squarederror", "eval_metric": "rmse",
    "random_state": 42, "n_jobs": 1, "tree_method": "hist",
}


def component_max_features(surfaces, amplitudes, mask):
    """Max contribution on the observed domain, matching the decomposition plots."""
    surfaces, amplitudes, mask = np.asarray(surfaces), np.asarray(amplitudes), np.asarray(mask, bool)
    if surfaces.ndim != 3 or amplitudes.ndim != 2 or surfaces.shape[1:] != mask.shape:
        raise ValueError("Invalid surface/amplitude/mask dimensions")
    if amplitudes.shape[1] != len(surfaces) or not mask.any():
        raise ValueError("Component counts do not match, or empty mask")
    return amplitudes * surfaces[:, mask].max(axis=1)[None, :]


def evaluate_regressor(x, y, estimator, *, n_splits=5, seed=42, feature_fitter=None):
    """With feature_fitter, X contains sample IDs; fit features on train only.

    Return train R², fold R², pooled OOF R², predictions and positional folds."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or x.shape[1] < 1 or len(x) != len(y) or y.ndim != 1:
        raise ValueError("Expected one or more peak features: X=(samples, features), y=(samples,)")
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    positions = np.flatnonzero(valid)
    x, y = x[valid], y[valid]
    row = {"n": len(y), "R2_train": np.nan, "R2_cv_mean": np.nan,
           "R2_cv_std": np.nan, "R2_oof": np.nan, "cv_splits": n_splits,
           "constant_test_folds": 0, "status": "ok"}
    # Each test fold needs >=2 samples for meaningful R².
    if len(y) < max(8, 2*n_splits) or np.ptp(y) == 0 or np.all(np.ptp(x, axis=0) == 0):
        row["status"] = "insufficient_or_constant"
        return row, pd.DataFrame()
    full_x = x if feature_fitter is None else feature_fitter(x, x)[0]
    full = clone(estimator).fit(full_x, y)
    row["R2_train"] = float(r2_score(y, full.predict(full_x)))
    oof, fold_scores, records = np.full(len(y), np.nan), [], []
    for fold, (train, test) in enumerate(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x)):
        train_x, test_x = (x[train], x[test]) if feature_fitter is None else feature_fitter(x[train], x[test])
        fitted = clone(estimator).fit(train_x, y[train])
        oof[test] = fitted.predict(test_x)
        # Match the old sklearn r2 scorer's force_finite behavior; flag constants.
        score = float(r2_score(y[test], oof[test]))
        fold_scores.append(score)
        row["constant_test_folds"] += int(np.ptp(y[test]) == 0)
        records.extend({"position": int(positions[i]), "fold": fold,
                        "observed": y[i], "predicted": oof[i],
                        "fold_R2": score, "n_train": len(train), "n_test": len(test)}
                       for i in test)
    row.update(R2_cv_mean=float(np.mean(fold_scores)), R2_cv_std=float(np.std(fold_scores, ddof=1)),
               R2_oof=float(r2_score(y, oof)))
    return row, pd.DataFrame(records)


def cached_component_features(eems, fit_model):
    """Cache by exact training membership; transform never updates the fit.

    Cache features only, so memory does not grow with fitted spectral models.
    IDs passed by the evaluator refer to rows of this entity's EEM tensor.
    """
    cache = {}

    def features(train_ids, test_ids):
        train = np.asarray(train_ids, int).ravel()
        test = np.asarray(test_ids, int).ravel()
        key = tuple(train)
        if key not in cache:
            fitted = fit_model(eems[train])
            cache[key] = component_max_features(
                fitted.full_surfaces, fitted.transform(eems), fitted.mask)
        values = cache[key]
        return values[train], values[test]

    return features


def best_components(table):
    """Select per variant/station/target/model; all-component models are separate."""
    entity = "Group" if "Group" in table else "station"
    valid = table[table.status.eq("ok") & table.component.ne("all_components")
                  & np.isfinite(table.R2_cv_mean)]
    return (valid.sort_values(["variant", entity, "target", "model", "R2_cv_mean", "component"],
                             ascending=[True, True, True, True, False, True])
            .groupby(["variant", entity, "target", "model"], sort=False).head(1).reset_index(drop=True))


def comparison_table(best):
    entity = "Group" if "Group" in best else "station"
    columns = ["variant", entity, "target", "component", "R2_train", "R2_cv_mean", "R2_cv_std", "R2_oof"]
    left = best[best.model.eq("linear")][columns]
    right = best[best.model.eq("xgb")][columns]
    return left.merge(right, on=["variant", entity, "target"], suffixes=("_linear", "_xgb"), how="outer")


def markdown_table(frame):
    """Small Markdown export without an optional tabulate dependency."""
    def value(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:.3f}" if np.isfinite(v) else "—"
        return str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(map(value, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


@threadpool_limits.wrap(limits=2)
def run_component_regression(output=None, *, n_splits=5, seed=42, xgb_params=None):
    import sklearn
    import xgboost
    from xgboost import XGBRegressor

    parent = Path(output or ROOT / "runs/diagnostics/v0_improved_10stations")
    out = parent / "component_regression"
    out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(ROOT / "data/processed/samples.csv")
    eems = np.load(ROOT / "data/processed/eem.npy", mmap_mode="r")
    with np.load(ROOT / "data/processed/wavelengths.npz") as axes:
        ex, em = axes["excitation"], axes["emission"]
    from eem_water_quality.features.improved_v0 import fit_improved_v0
    config = json.loads((parent / "config.json").read_text())
    group_mode = config.get("benchmark_column") == "Group"
    entity = "Group" if group_mode else "station"
    sample_column = "Group" if group_mode else "Point"
    stations = pd.read_csv(parent / ("selected_groups.csv" if group_mode else "selected_stations.csv"))[sample_column].tolist()
    names = [r[0] for r in config["regions"]]
    params = dict(XGB_PARAMS if xgb_params is None else xgb_params)
    estimators = {"linear": LinearRegression(), "xgb": XGBRegressor(**params)}
    metadata = {"cv": "KFold with train-only decomposition and fixed held-out shapes", "n_splits": n_splits,
                "shuffle": True, "seed": seed, "feature": "raw max observed contribution; no log transform",
                "include_background": False, "include_all_components": True,
                "decomposition": {"max_steps": config.get("cv_steps", 300),
                                  "n_starts": config.get("cv_starts", 2),
                                  "variants": config["variants"], "regions": config["regions"]},
                "exported_features": "full-entity descriptive fit only; never used for CV",
                "n_features_per_model": {"single": 1, "all_components": len(names)},
                "xgb_params": params, "sklearn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
                "selection": "best single component by same CV mean; exploratory, not nested",
                "scope": "end-to-end shuffled CV; not chronological CV"}
    metadata["benchmark_column"] = sample_column
    if group_mode:
        metadata["cv"] = metadata["cv"].replace("station", "group")
    (out / "config.json").write_text(json.dumps(metadata, indent=2))
    def artifact(name):
        return out / (name.replace("station_", "group_") if group_mode else name)

    rows, predictions, feature_frames = [], [], []
    for variant in config["variants"]:
        for station in stations:
            indices = np.flatnonzero(samples[sample_column].eq(station))
            raw = np.asarray(eems[indices], float)
            fit_params = {**config["variants"][variant], "regions": config["regions"],
                          "max_steps": config.get("cv_steps", 300),
                          "n_starts": config.get("cv_starts", 2), "seed": seed}
            feature_fitter = cached_component_features(
                raw, lambda train: fit_improved_v0(train, ex, em, **fit_params))
            ids = np.arange(len(indices))[:, None]
            features = feature_fitter(ids, ids)[0]
            feature_frame = pd.DataFrame(features[:, :-1], columns=[f"max_{n}" for n in names])
            feature_frame.insert(0, "sample_index", indices)
            feature_frame.insert(0, entity, station)
            feature_frame.insert(0, "variant", variant)
            feature_frames.append(feature_frame)
            groups = [(name, [k]) for k, name in enumerate(names)]
            groups.append(("all_components", list(range(len(names)))))
            for target in ["BOD", "COD", "TOC"]:
                column = next(c for c in samples if c.splitlines()[0].strip() == target)
                y = pd.to_numeric(samples.loc[indices, column], errors="coerce").to_numpy(float)
                for component, selected in groups:
                    for model, estimator in estimators.items():
                        metrics, oof = evaluate_regressor(
                            ids, y, estimator, n_splits=n_splits, seed=seed,
                            feature_fitter=lambda train, test: tuple(
                                values[:, selected] for values in feature_fitter(train, test)))
                        identity = {"variant": variant, entity: station, "target": target,
                                    "model": model, "component": component,
                                    "feature": "+".join(f"max_{names[k]}" for k in selected),
                                    "n_features": len(selected)}
                        rows.append({**identity, **metrics})
                        if not oof.empty:
                            oof["sample_index"] = indices[oof.pop("position").to_numpy(int)]
                            for key, value in identity.items():
                                oof[key] = value
                            predictions.append(oof)
            print(f"[component regression] {variant}: {station}", flush=True)
    table = pd.DataFrame(rows)
    best = best_components(table)
    comparison = comparison_table(best)
    table.to_csv(artifact("station_regression_r2.csv"), index=False)
    best.to_csv(artifact("station_best_component_r2.csv"), index=False)
    comparison.to_csv(artifact("station_best_component_train_cv_comparison.csv"), index=False)
    pd.concat(feature_frames, ignore_index=True).to_csv(artifact("station_component_max_features.csv"), index=False)
    (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()).to_csv(out / "oof_predictions.csv", index=False)
    all_components = table[table.component.eq("all_components")].copy()
    assert table.loc[table.component.ne("all_components"), "n_features"].eq(1).all()
    assert all_components.n_features.eq(len(names)).all()
    all_components.to_csv(artifact("station_all_components_r2.csv"), index=False)
    for model, filename in [("linear", "linear_cv"), ("xgb", "xgb")]:
        renamed = {name: name.replace("R2_", f"R2_{model}_", 1)
                   for name in ["R2_train", "R2_cv_mean", "R2_cv_std", "R2_oof"]}
        table[table.model.eq(model) & table.component.ne("all_components")].rename(columns=renamed).to_csv(
            artifact(f"station_component_{filename}_r2.csv"), index=False)
        best[best.model.eq(model)].rename(columns=renamed).to_csv(
            artifact(f"station_best_component_{filename}_r2.csv"), index=False)
    summary = best.groupby(["variant", "model", "target"])[["R2_train", "R2_cv_mean", "R2_oof"]].agg(["mean", "median"])
    summary.to_csv(out / "best_component_summary.csv")
    report = "# Linear Regression và XGBoost theo trạm — V0 cải tiến\n\n"
    report += "Feature là cực đại intensity đóng góp của từng component, không gồm nền. "
    report += f"KFold {n_splits} folds, shuffle=True, seed={seed}; XGBoost dùng cấu hình notebook cũ. "
    report += "Train R² là fit trên toàn bộ dữ liệu trạm. CV mean ± SD là giữa các fold (SD mẫu). "
    report += "OOF R² tính từ các dự đoán held-out ghép lại, nên không đồng nhất với mean fold R².\n\n"
    report += "**Mỗi fold fit decomposition trên train; held-out chỉ suy ra biên độ với component và mask train cố định.** "
    report += "Component tốt nhất được chọn trên chính CV báo cáo, nên có selection optimism. "
    report += "Không so trực tiếp với temporal CV ở phần trước như cùng một protocol. "
    report += "Benchmark gồm từng Fmax riêng và tất cả Fmax cùng lúc, không gồm background.\n\n"
    for variant in config["variants"]:
        report += f"## {variant}\n\n"
        subset = comparison[comparison.variant.eq(variant)].drop(columns="variant")
        report += markdown_table(subset) + "\n\n"
    report += "## Kết quả từng peak component\n\n"
    report += "Bảng đầy đủ trong `station_regression_r2.csv`: mỗi dòng là một peak × target × estimator × trạm × biến thể.\n"
    report += "\n## Tất cả component cùng lúc\n\n" + markdown_table(all_components) + "\n"
    if group_mode:
        report = report.replace("trạm", "group").replace("station_", "group_")
    (out / "report.md").write_text(report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for variant in config["variants"]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
        for i, model in enumerate(estimators):
            subset = best[best.variant.eq(variant) & best.model.eq(model)]
            for j, metric in enumerate(["R2_train", "R2_cv_mean"]):
                matrix = subset.pivot(index=entity, columns="target", values=metric).reindex(index=stations, columns=["BOD", "COD", "TOC"])
                im = axes[i, j].imshow(matrix, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
                axes[i, j].set_xticks(range(3), matrix.columns)
                axes[i, j].set_yticks(range(len(stations)), stations)
                axes[i, j].set_title(f"{model}: {metric}")
                for r in range(len(stations)):
                    for c in range(3):
                        axes[i, j].text(c, r, f"{matrix.iloc[r,c]:.2f}", ha="center", va="center", fontsize=8)
                fig.colorbar(im, ax=axes[i, j], extend="min")
        fig.suptitle(f"{variant}: best single component by CV\nTrain-only decomposition CV; color clipped below -1, labels show actual R²")
        fig.savefig(out / f"{variant}_train_cv_r2.png", dpi=160)
        plt.close(fig)
    return {"output": out, "scores": table, "best": best, "comparison": comparison,
            "summary": summary, "all_components": all_components}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/diagnostics/v0_improved_10stations")
    args = parser.parse_args()
    run_component_regression(args.output)
