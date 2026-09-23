"""Summarize target distributions by station and quantify LOSO extrapolation.

The diagnostic uses only the target table; EEM features are not needed.  For
each station it reports the number of finite target values, location and spread
statistics.  It also treats that station as an LOSO test fold and compares its
target range with the range observed in the remaining stations.

Example::

    PYTHONPATH=src python scripts/diagnostics/analyze_target_by_station.py \
        --data data/processed \
        --output runs/diagnostics/target_by_station \
        --targets BOD COD TOC BOD_COD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eem_water_quality.data import load_processed_dataset
from eem_water_quality.data.schema import resolve_column


def _summary(values: pd.Series) -> dict[str, float | int]:
    """Return stable summary statistics for a finite numeric series."""
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if values.empty:
        return {
            "n_valid": 0,
            "mean": np.nan,
            "variance": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }
    return {
        "n_valid": int(values.size),
        "mean": float(values.mean()),
        "variance": float(values.var(ddof=1)) if values.size > 1 else np.nan,
        "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
        "min": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def target_by_station(samples: pd.DataFrame, target: str, station_col: str = "Point") -> pd.DataFrame:
    """Build per-station target distribution statistics."""
    if station_col not in samples:
        raise ValueError(f"Station column {station_col!r} is absent from the sample table")
    if target not in samples:
        raise ValueError(f"Target column {target!r} is absent from the sample table")
    if samples[station_col].isna().any():
        raise ValueError(f"Station column {station_col!r} contains missing values")

    numeric = pd.to_numeric(samples[target], errors="coerce")
    rows: list[dict[str, object]] = []
    for station, group in samples.groupby(station_col, sort=True, dropna=False):
        stats = _summary(numeric.loc[group.index])
        rows.append(
            {
                "station": station,
                "n_total": len(group),
                "n_missing_target": int(numeric.loc[group.index].isna().sum()),
                **stats,
            }
        )
    result = pd.DataFrame(rows)
    result.insert(0, "target_column", target)
    return result


def loso_extrapolation(
    samples: pd.DataFrame, target: str, station_col: str = "Point"
) -> pd.DataFrame:
    """Compare every station's target range with the other stations' range.

    ``outside_train_range_fraction`` is the fraction of held-out station
    observations below the minimum or above the maximum target observed in the
    LOSO training stations.  This is a direct, leakage-free indicator that the
    fold requires target extrapolation.
    """
    numeric = pd.to_numeric(samples[target], errors="coerce")
    valid = pd.DataFrame({"station": samples[station_col], "value": numeric}).dropna()
    rows: list[dict[str, object]] = []
    for station in sorted(valid["station"].unique()):
        test_values = valid.loc[valid["station"] == station, "value"]
        train_values = valid.loc[valid["station"] != station, "value"]
        station_stats = _summary(test_values)
        train_stats = _summary(train_values)
        below = test_values < train_stats["min"]
        above = test_values > train_stats["max"]
        outside = below | above
        test_range = station_stats["max"] - station_stats["min"]
        overlap = max(
            0.0,
            min(station_stats["max"], train_stats["max"])
            - max(station_stats["min"], train_stats["min"]),
        )
        range_overlap_fraction = (
            1.0 if test_range == 0 and not bool(outside.any()) else overlap / test_range
        ) if test_range > 0 else (1.0 if not bool(outside.any()) else 0.0)
        train_std = train_stats["std"]
        mean_z = (
            (station_stats["mean"] - train_stats["mean"]) / train_std
            if pd.notna(train_std) and train_std > 0
            else np.nan
        )
        rows.append(
            {
                "target_column": target,
                "station": station,
                "n_test": len(test_values),
                "station_mean": station_stats["mean"],
                "station_min": station_stats["min"],
                "station_max": station_stats["max"],
                "train_n": train_stats["n_valid"],
                "train_mean": train_stats["mean"],
                "train_std": train_stats["std"],
                "train_min": train_stats["min"],
                "train_max": train_stats["max"],
                "below_train_min_count": int(below.sum()),
                "above_train_max_count": int(above.sum()),
                "outside_train_range_count": int(outside.sum()),
                "outside_train_range_fraction": float(outside.mean()),
                "range_overlap_fraction": float(range_overlap_fraction),
                "mean_z_vs_train": float(mean_z) if pd.notna(mean_z) else np.nan,
                "requires_target_extrapolation": bool(outside.any()),
                "fully_outside_train_range": bool(
                    station_stats["max"] < train_stats["min"]
                    or station_stats["min"] > train_stats["max"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed", help="processed dataset directory")
    parser.add_argument(
        "--output",
        default="runs/diagnostics/target_by_station",
        help="directory for diagnostic CSV files",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["BOD", "COD", "TOC", "BOD_COD"],
        help="user-facing target names or processed column names",
    )
    parser.add_argument("--station-col", default="Point")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _, samples, _ = load_processed_dataset(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    station_frames: list[pd.DataFrame] = []
    extrapolation_frames: list[pd.DataFrame] = []
    global_rows: list[dict[str, object]] = []
    for target_name in args.targets:
        target = resolve_column(target_name)
        if target not in samples:
            raise ValueError(f"Target {target_name!r} resolves to missing column {target!r}")
        station_frame = target_by_station(samples, target, args.station_col)
        station_frame.insert(0, "target", target_name)
        station_frames.append(station_frame)
        extrapolation_frame = loso_extrapolation(samples, target, args.station_col)
        extrapolation_frame.insert(0, "target", target_name)
        extrapolation_frames.append(extrapolation_frame)
        global_stats = _summary(pd.to_numeric(samples[target], errors="coerce"))
        global_rows.append({"target": target_name, "target_column": target, **global_stats})

        n_extrapolation = int(extrapolation_frame["requires_target_extrapolation"].sum())
        print(
            f"{target_name}: {len(station_frame)} stations, "
            f"{int(station_frame['n_valid'].sum())} finite samples, "
            f"{n_extrapolation} stations with values outside the LOSO train range",
            flush=True,
        )

    pd.concat(station_frames, ignore_index=True).to_csv(output / "target_by_station.csv", index=False)
    pd.concat(extrapolation_frames, ignore_index=True).to_csv(
        output / "loso_extrapolation.csv", index=False
    )
    global_frame = pd.DataFrame(global_rows)
    global_frame.to_csv(output / "target_global_summary.csv", index=False)
    metadata = {
        "data": str(Path(args.data)),
        "station_column": args.station_col,
        "targets": args.targets,
        "outputs": [
            "target_by_station.csv",
            "loso_extrapolation.csv",
            "target_global_summary.csv",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
