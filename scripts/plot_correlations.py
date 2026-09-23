"""Plot correlations between water-quality variables and EEM PCA scores.

The PCA is fitted only to create an exploratory compact representation of the
EEM matrix for the correlation plot. It is not a model-training transform.
Run from the repository root with ``PYTHONPATH=src``.
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

from eem_water_quality.data import BOD, COD, EC, PH, SS, TEMP, TOC, load_data

VARIABLES = (
    ("BOD", BOD),
    ("COD", COD),
    ("TOC", TOC),
    ("BOD_COD", "BOD_COD"),
    ("SS", SS),
    ("EC", EC),
    ("Temp", TEMP),
    ("pH", PH),
)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create Pearson/Spearman correlation tables and EEM PCA heatmaps"
    )
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--output", default="runs/correlation_analysis")
    parser.add_argument("--pca-components", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def build_frame(data: Path, pca_components: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    eem, samples, _ = load_data(data)
    values: dict[str, np.ndarray] = {}
    for name, column in VARIABLES:
        if column in samples:
            values[name] = pd.to_numeric(samples[column], errors="coerce").to_numpy(float)

    flat = eem.reshape(len(eem), -1).astype(np.float64)
    scaled = StandardScaler().fit_transform(flat)
    n_components = min(pca_components, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(scaled)
    for index in range(n_components):
        values[f"EEM_PC{index + 1}"] = scores[:, index]

    frame = pd.DataFrame(values)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    explained = pd.DataFrame(
        {
            "component": [f"EEM_PC{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_percent": 100 * pca.explained_variance_ratio_,
        }
    )
    return frame, explained


def plot_heatmap(correlation: pd.DataFrame, path: Path, title: str) -> None:
    labels = list(correlation.columns)
    size = max(8.0, 0.72 * len(labels))
    figure, axis = plt.subplots(figsize=(size, size * 0.85))
    image = axis.imshow(correlation.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title(title)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlation.iloc[row, column]
            if np.isfinite(value):
                text_color = "white" if abs(value) >= 0.55 else "black"
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=text_color)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Correlation")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def top_pairs(correlation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row, first in enumerate(correlation.index):
        for column in range(row + 1, len(correlation.columns)):
            second = correlation.columns[column]
            value = correlation.iloc[row, column]
            if np.isfinite(value):
                rows.append(
                    {
                        "variable_1": first,
                        "variable_2": second,
                        "correlation": float(value),
                        "absolute_correlation": float(abs(value)),
                    }
                )
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False)


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame, explained = build_frame(Path(args.data), args.pca_components, args.seed)
    frame.to_csv(output / "correlation_variables.csv", index=False)
    explained.to_csv(output / "eem_pca_explained_variance.csv", index=False)

    for method in ("pearson", "spearman"):
        correlation = frame.corr(method=method)
        correlation.to_csv(output / f"{method}_correlation.csv")
        plot_heatmap(correlation, output / f"{method}_correlation_heatmap.png", f"{method.title()} correlation")
        top_pairs(correlation).to_csv(output / f"{method}_top_pairs.csv", index=False)

    print(f"Variables: {', '.join(frame.columns)}")
    print(f"Samples: {len(frame)}; PCA components: {len(explained)}")
    print(f"Saved correlation outputs to {output}")
    print("Top Pearson pairs:")
    print(top_pairs(frame.corr(method="pearson")).head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
