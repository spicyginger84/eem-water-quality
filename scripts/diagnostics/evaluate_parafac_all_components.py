"""Train-only PARAFAC decomposition and regression CV within each group."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_v0_component_regression import (
    ROOT, XGB_PARAMS, cached_component_features, evaluate_regressor,
)
from eem_water_quality.features.peak_decomposition import fit_peak_model
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor


@threadpool_limits.wrap(limits=2)
def run_all_components(output=None, *, rank=4, n_splits=5, seed=42,
                       n_starts=3, max_steps=500, include_single=False):
    """Rank is prespecified, never loaded from full-group rank selection.

    Single C labels are ordered by train-only Em/Ex peaks in each fold; they
    need not represent identical chemistry across folds. Best-single tables
    remain exploratory selection on the reported CV, not nested estimates.
    """
    output = Path(output or ROOT / "runs/diagnostics/parafac/by_group")
    samples = pd.read_csv(ROOT / "data/processed/samples.csv")
    eems = np.load(ROOT / "data/processed/eem.npy", mmap_mode="r")
    with np.load(ROOT / "data/processed/wavelengths.npz") as axes:
        ex, em = axes["excitation"], axes["emission"]
    selected = pd.read_csv(output / "selected_top_groups.csv").Group.tolist()
    if rank < 1:
        raise ValueError("rank must be positive and prespecified")

    def fit_train(train):
        fitted = fit_peak_model(train, ex, em, family="parafac", rank=rank,
                                weighting="raw", background=False, n_starts=n_starts,
                                max_steps=max_steps, seed=seed)
        peaks = [np.unravel_index(surface.argmax(), surface.shape)
                 for surface in fitted.full_surfaces]
        order = sorted(range(rank), key=lambda k: peaks[k])
        fitted.full_surfaces = fitted.full_surfaces[order]
        fitted.surfaces = fitted.surfaces[order]
        return fitted

    rows, predictions = [], []
    for group in selected:
        indices = np.flatnonzero(samples.Group.eq(group))
        raw = np.maximum(np.asarray(eems[indices], float), 0)
        feature_fitter = cached_component_features(raw, fit_train)
        ids = np.arange(len(indices))[:, None]
        components = [("all_components", list(range(rank)))]
        if include_single:
            components += [(f"C{k+1}", [k]) for k in range(rank)]
        for target in ("BOD", "COD", "TOC"):
            column = next(c for c in samples if c.splitlines()[0].strip() == target)
            y = pd.to_numeric(samples.loc[indices, column], errors="coerce").to_numpy(float)
            for component, columns in components:
                for model, estimator in (("linear", LinearRegression()),
                                         ("xgb", XGBRegressor(**XGB_PARAMS))):
                    metrics, oof = evaluate_regressor(
                        ids, y, estimator, n_splits=n_splits, seed=seed,
                        feature_fitter=lambda train, test: tuple(
                            values[:, columns] for values in feature_fitter(train, test)))
                    identity = {"Group": group, "target": target, "model": model,
                                "component": component, "n_features": len(columns),
                                "selected_rank": rank,
                                "feature": "+".join(f"max_C{k+1}" for k in columns)}
                    rows.append({**identity, **metrics})
                    if not oof.empty:
                        oof["sample_index"] = indices[oof.pop("position").to_numpy(int)]
                        for key, value in identity.items():
                            oof[key] = value
                        predictions.append(oof)
        print(f"[PARAFAC train-only CV] {group}", flush=True)
    table = pd.DataFrame(rows)
    all_components = table[table.component.eq("all_components")]
    all_components.to_csv(output / "group_all_components_r2.csv", index=False)
    oof = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    oof.to_csv(output / "group_component_oof_predictions.csv", index=False)
    (oof[oof.component.eq("all_components")] if not oof.empty else oof).to_csv(
        output / "group_all_components_oof_predictions.csv", index=False)
    if include_single:
        for model, suffix in (("linear", "linear_cv"), ("xgb", "xgb")):
            single = table[table.model.eq(model) & table.component.ne("all_components")]
            rename = {key: key.replace("R2_", f"R2_{model}_", 1)
                      for key in ("R2_train", "R2_cv_mean", "R2_cv_std", "R2_oof")}
            single.rename(columns=rename).to_csv(
                output / f"group_component_{suffix}_r2.csv", index=False)
            best = (single[single.status.eq("ok")].sort_values("R2_cv_mean", ascending=False)
                    .groupby(["Group", "target"]).head(1))
            best.rename(columns=rename).to_csv(
                output / f"group_best_component_{suffix}_r2.csv", index=False)
    (output / "regression_cv_config.json").write_text(json.dumps({
        "scope": "end-to-end shuffled CV with train-only decomposition",
        "rank": rank, "rank_selection": "prespecified; no full-group selected ranks",
        "n_splits": n_splits, "seed": seed, "n_starts": n_starts,
        "max_steps": max_steps, "solver": "fit_peak_model masked nonnegative CP, raw loss",
        "mask_and_scaling": "training fold only", "heldout": "fixed spectra NNLS scores",
        "single_selection": "exploratory same-CV selection; not nested",
        "component_order": "fold-local Em then Ex peak", "xgb_params": XGB_PARAMS,
    }, indent=2))
    return all_components


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--include-single", action="store_true")
    args = parser.parse_args()
    print(run_all_components(args.output, rank=args.rank,
                             include_single=args.include_single).to_string(index=False))
