"""Relate held-out-station EEM distance to LOSO performance.

This diagnostic consumes an existing LOSO run.  It does not retrain a model.
For every held-out station it fits a representation of the EEM on that fold's
training rows, measures the held-out station's distance to every training
station, and joins those distances to the per-fold R² already written by the
pipeline.

The default PCA representation follows the feature pipeline: training-only
standardization, then PCA.  ``--distance-space raw`` instead uses the masked,
standardized flattened EEM.  In both cases the mask and transformations are
fit separately inside each LOSO fold.

Example::

    PYTHONPATH=src python scripts/diagnostics/analyze_loso_eem_distance.py \
        --data data/processed \
        --run runs/loso_all_eempca \
        --output runs/diagnostics/loso_eem_distance \
        --targets BOD COD TOC BOD_COD \
        --features EEMpca --models linear xgboost
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from eem_water_quality.data import load_processed_dataset
from eem_water_quality.data.schema import resolve_column


def _representation(
    eem: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    space: str,
    components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a fold-local EEM representation and transform train/test rows."""
    flat = eem.reshape(len(eem), -1).astype(np.float64)
    mask = np.any(np.abs(flat[train_indices]) > 0, axis=0)
    if not np.any(mask):
        raise ValueError("The LOSO training EEM contains no nonzero feature")
    train_flat = flat[train_indices][:, mask]
    test_flat = flat[test_indices][:, mask]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat)
    test_scaled = scaler.transform(test_flat)
    if space == "raw":
        return train_scaled, test_scaled
    rank = min(components, *train_scaled.shape)
    if rank < 1:
        raise ValueError("PCA needs at least one training sample and feature")
    pca = PCA(n_components=rank, random_state=42)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def _distance_row(
    train_repr: np.ndarray,
    test_repr: np.ndarray,
    train_stations: pd.Series,
    heldout_station: str,
) -> dict[str, object]:
    """Summarize held-out-station distance to train station centroids/samples."""
    train_stations = train_stations.astype(str).to_numpy()
    station_names = np.array(sorted(np.unique(train_stations)), dtype=object)
    centroids = np.vstack(
        [train_repr[train_stations == station].mean(axis=0) for station in station_names]
    )
    heldout_centroid = test_repr.mean(axis=0, keepdims=True)
    station_distances = pairwise_distances(heldout_centroid, centroids, metric="euclidean")[0]
    nearest_station_index = int(np.argmin(station_distances))
    sample_distances = pairwise_distances(test_repr, train_repr, metric="euclidean")
    nearest_sample = float(sample_distances.min(axis=1).mean())
    return {
        "heldout_group": heldout_station,
        "n_test_eem": len(test_repr),
        "n_train_eem": len(train_repr),
        "n_train_stations": len(station_names),
        "nearest_train_station": station_names[nearest_station_index],
        "nearest_station_distance": float(station_distances[nearest_station_index]),
        "mean_station_distance": float(station_distances.mean()),
        "median_station_distance": float(np.median(station_distances)),
        "farthest_station_distance": float(station_distances.max()),
        "nearest_sample_distance": nearest_sample,
    }


def compute_fold_distances(
    eem: np.ndarray,
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    target: str,
    station_col: str = "Point",
    distance_space: str = "pca",
    components: int = 30,
    distance_cache: dict[tuple[bytes, bytes, str, int], dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Compute one EEM-distance row per LOSO fold for ``target``."""
    if station_col not in samples:
        raise ValueError(f"Station column {station_col!r} is absent from processed data")
    target_column = resolve_column(target)
    if target_column not in samples:
        raise ValueError(f"Target column {target_column!r} is absent from processed data")
    target_values = pd.to_numeric(samples[target_column], errors="coerce").to_numpy(dtype=float)
    station_values = samples[station_col].astype(str)
    rows: list[dict[str, object]] = []
    fold_metrics = metrics.loc[
        (metrics["target"].astype(str) == str(target))
        & (metrics["protocol"] == "loso")
        & (metrics["partition"] == "test")
    ].copy()
    fold_metrics = fold_metrics.drop_duplicates(subset=["fold", "heldout_group"])
    for record in fold_metrics.itertuples(index=False):
        heldout_station = str(record.heldout_group)
        test_mask = (station_values == heldout_station).to_numpy() & np.isfinite(target_values)
        train_mask = (station_values != heldout_station).to_numpy() & np.isfinite(target_values)
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        if len(train_indices) < 2 or len(test_indices) < 2:
            continue
        cache_key = (train_indices.tobytes(), test_indices.tobytes(), distance_space, components)
        if distance_cache is not None and cache_key in distance_cache:
            distance = dict(distance_cache[cache_key])
        else:
            train_repr, test_repr = _representation(
                eem, train_indices, test_indices, distance_space, components
            )
            distance = _distance_row(
                train_repr,
                test_repr,
                samples.iloc[train_indices][station_col],
                heldout_station,
            )
            if distance_cache is not None:
                distance_cache[cache_key] = dict(distance)
        rows.append(
            {
                "target": str(record.target),
                "fold": record.fold,
                "distance_space": distance_space,
                **distance,
            }
        )
    return pd.DataFrame(rows)


def _correlations(joined: pd.DataFrame) -> pd.DataFrame:
    distance_columns = [
        "nearest_station_distance",
        "mean_station_distance",
        "median_station_distance",
        "nearest_sample_distance",
    ]
    rows: list[dict[str, object]] = []
    for (target, features, model, space), group in joined.groupby(
        ["target", "features", "model", "distance_space"], dropna=False
    ):
        for distance_column in distance_columns:
            valid = group[[distance_column, "R2"]].dropna()
            row: dict[str, object] = {
                "target": target,
                "features": features,
                "model": model,
                "distance_space": space,
                "distance": distance_column,
                "n_folds": len(valid),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
            }
            if len(valid) >= 3 and valid[distance_column].nunique() > 1 and valid["R2"].nunique() > 1:
                pearson = pearsonr(valid[distance_column], valid["R2"])
                spearman = spearmanr(valid[distance_column], valid["R2"])
                row.update(
                    {
                        "pearson_r": float(pearson.statistic),
                        "pearson_p": float(pearson.pvalue),
                        "spearman_rho": float(spearman.statistic),
                        "spearman_p": float(spearman.pvalue),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--run", required=True, help="existing LOSO run directory")
    parser.add_argument("--output", default="runs/diagnostics/loso_eem_distance")
    parser.add_argument(
        "--targets", nargs="+", default=["BOD", "COD", "TOC", "BOD_COD"]
    )
    parser.add_argument("--features", nargs="+", default=["EEMpca"])
    parser.add_argument("--models", nargs="+", default=["linear", "xgboost"])
    parser.add_argument("--station-col", default="Point")
    parser.add_argument("--distance-space", choices=["pca", "raw"], default="pca")
    parser.add_argument("--components", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.components < 1:
        raise ValueError("--components must be positive")
    eem, samples, _ = load_processed_dataset(args.data)
    run_root = Path(args.run)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    distance_frames: list[pd.DataFrame] = []
    joined_frames: list[pd.DataFrame] = []
    distance_cache: dict[tuple[bytes, bytes, str, int], dict[str, object]] = {}

    for target_name in args.targets:
        target = resolve_column(target_name)
        target_dir = run_root / target_name
        metrics_path = target_dir / "holdout_metrics.csv"
        if target not in samples:
            raise ValueError(f"Target {target_name!r} resolves to missing column {target!r}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing completed LOSO metrics: {metrics_path}")
        metrics = pd.read_csv(metrics_path)
        distance = compute_fold_distances(
            eem,
            samples,
            metrics,
            target_name,
            station_col=args.station_col,
            distance_space=args.distance_space,
            components=args.components,
            distance_cache=distance_cache,
        )
        distance.insert(0, "target_name", target_name)
        distance_frames.append(distance)

        model_metrics = metrics.loc[
            (metrics["partition"] == "test")
            & (metrics["protocol"] == "loso")
            & metrics["features"].isin(args.features)
            & metrics["model"].isin(args.models)
            & (metrics["status"] == "ok"),
            ["target", "fold", "heldout_group", "features", "model", "R2", "RMSE", "MAE"],
        ].copy()
        model_metrics["heldout_group"] = model_metrics["heldout_group"].astype(str)
        distance["heldout_group"] = distance["heldout_group"].astype(str)
        joined = model_metrics.merge(
            distance.drop(columns=["target_name"], errors="ignore"),
            on=["target", "fold", "heldout_group"],
            how="inner",
            validate="many_to_one",
        )
        joined.insert(0, "target_name", target_name)
        joined_frames.append(joined)

    distance_frame = pd.concat(distance_frames, ignore_index=True)
    joined_frame = pd.concat(joined_frames, ignore_index=True)
    distance_frame.to_csv(output / "loso_station_eem_distances.csv", index=False)
    joined_frame.to_csv(output / "loso_distance_vs_performance.csv", index=False)
    _correlations(joined_frame).to_csv(output / "loso_distance_correlations.csv", index=False)
    print(f"Wrote diagnostics to {output}", flush=True)


if __name__ == "__main__":
    main()
