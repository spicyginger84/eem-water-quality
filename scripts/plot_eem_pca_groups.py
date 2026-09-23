"""Create EEM PCA scatter plots coloured by station and month.

This is an exploratory diagnostic rather than a training transform.  The PCA is
fit on all rows in the supplied processed dataset.  Columns that are exactly
zero for every sample are removed before scaling and PCA; this prevents the
zero-filled Rayleigh/scatter region from dominating the representation.

Run from the repository root with ``PYTHONPATH=src``, for example::

    PYTHONPATH=src python scripts/plot_eem_pca_groups.py \
        --data data/processed --output runs/eem_pca_groups

The output directory contains the PCA scores, explained variance, and one plot
for station and one for month.  The plots are intended to reveal whether EEM
clusters are primarily associated with station, month, or both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eem_water_quality.data import load_data


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot EEM PCA scores coloured by station and month"
    )
    parser.add_argument("--data", default="data/processed", help="processed data directory")
    parser.add_argument("--output", default="runs/eem_pca_groups", help="output directory")
    parser.add_argument(
        "--pca-components",
        type=positive_int,
        default=10,
        help="number of PCA components to fit (the first two are plotted)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=positive_int, default=180)
    parser.add_argument(
        "--point-size",
        type=positive_float,
        default=12.0,
        help="marker area in points squared for the scatter plots",
    )
    parser.add_argument(
        "--no-global-zero-mask",
        action="store_true",
        help="keep EEM cells that are zero in every sample (not recommended)",
    )
    return parser


def fit_pca(eem: np.ndarray, components: int, seed: int, mask_global_zero: bool):
    if eem.ndim != 3:
        raise ValueError(f"Expected a 3D EEM array, got shape {eem.shape}")
    flat = np.asarray(eem, dtype=np.float32).reshape(len(eem), -1)
    if not np.isfinite(flat).all():
        raise ValueError("EEM contains NaN or Inf")
    if mask_global_zero:
        feature_mask = np.any(np.abs(flat) > 0, axis=0)
    else:
        feature_mask = np.ones(flat.shape[1], dtype=bool)
    if not np.any(feature_mask):
        raise ValueError("No nonzero EEM cells remain after masking")
    masked = flat[:, feature_mask]
    scaled = StandardScaler().fit_transform(masked)
    n_components = min(components, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(scaled)
    return scores, pca, feature_mask


def _group_values(samples: pd.DataFrame, column: str) -> pd.Series:
    if column not in samples:
        raise KeyError(f"Processed metadata is missing required column {column!r}")
    values = samples[column].astype("string").fillna("<missing>")
    return values


def plot_group_scores(
    scores: np.ndarray,
    groups: pd.Series,
    explained: np.ndarray,
    path: Path,
    title_prefix: str,
    dpi: int,
    point_size: float,
) -> None:
    categories = sorted(groups.unique().tolist(), key=str)
    # tab20 is readable for a small number of groups; hsv gives distinct hues
    # when there are many stations.
    cmap_name = "tab20" if len(categories) <= 20 else "hsv"
    cmap = plt.get_cmap(cmap_name, len(categories))
    figure_width = 9.5 if len(categories) <= 20 else 12.5
    figure, axis = plt.subplots(figsize=(figure_width, 7.0))
    for index, category in enumerate(categories):
        selected = groups.to_numpy() == category
        axis.scatter(
            scores[selected, 0],
            scores[selected, 1],
            s=point_size,
            alpha=0.78,
            color=cmap(index),
            edgecolors="none",
            label=str(category),
        )
    axis.set_xlabel(f"PC1 ({explained[0] * 100:.2f}% variance)")
    axis.set_ylabel(f"PC2 ({explained[1] * 100:.2f}% variance)")
    axis.set_title(f"{title_prefix}: EEM PCA scores")
    axis.grid(True, alpha=0.2)
    axis.axhline(0, color="0.75", linewidth=0.7)
    axis.axvline(0, color="0.75", linewidth=0.7)
    axis.legend(
        title=title_prefix,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        ncol=1 if len(categories) <= 24 else 2,
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    eem, samples, _wavelengths = load_data(Path(args.data))
    scores, pca, feature_mask = fit_pca(
        eem,
        args.pca_components,
        args.seed,
        mask_global_zero=not args.no_global_zero_mask,
    )
    if scores.shape[1] < 2:
        raise ValueError("At least two PCA components are required for a scatter plot")

    station = _group_values(samples, "Point")
    month = _group_values(samples, "Month")
    score_frame = pd.DataFrame(
        {f"EEM_PC{index + 1}": scores[:, index] for index in range(scores.shape[1])}
    )
    score_frame.insert(0, "row_position", np.arange(len(score_frame)))
    score_frame["station"] = station.to_numpy()
    score_frame["month"] = month.to_numpy()
    score_frame.to_csv(output / "eem_pca_scores.csv", index=False)

    explained = pd.DataFrame(
        {
            "component": [f"EEM_PC{index + 1}" for index in range(scores.shape[1])],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_percent": 100 * pca.explained_variance_ratio_,
        }
    )
    explained.to_csv(output / "eem_pca_explained_variance.csv", index=False)
    pd.DataFrame(
        {
            "raw_features": [int(np.prod(eem.shape[1:]))],
            "kept_features": [int(feature_mask.sum())],
            "zero_features_masked": [int((~feature_mask).sum())],
            "samples": [len(eem)],
            "emission_wavelengths": [eem.shape[1]],
            "excitation_wavelengths": [eem.shape[2]],
            "pca_components": [scores.shape[1]],
        }
    ).to_csv(output / "eem_pca_mask_summary.csv", index=False)

    plot_group_scores(
        scores,
        station,
        pca.explained_variance_ratio_,
        output / "eem_pca_by_station.png",
        "Station",
        args.dpi,
        args.point_size,
    )
    plot_group_scores(
        scores,
        month,
        pca.explained_variance_ratio_,
        output / "eem_pca_by_month.png",
        "Month",
        args.dpi,
        args.point_size,
    )

    print(f"Samples: {len(samples)}")
    print(
        f"EEM shape: {tuple(eem.shape)}; kept features: {int(feature_mask.sum())}/"
        f"{int(feature_mask.size)}"
    )
    print(f"Stations: {station.nunique()}; months: {month.nunique()}")
    print(f"Explained variance (first two PCs): {pca.explained_variance_ratio_[:2].sum():.3f}")
    print(f"Saved PCA scores and plots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
