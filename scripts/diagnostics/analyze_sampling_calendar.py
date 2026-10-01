"""Audit the retained sample calendar and export tables and a static figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/sampling_calendar"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(args.data / "samples.parquet").copy()
    s["date"] = pd.to_datetime(s["Sampling date"], format="%Y.%m.%d", errors="raise")
    s["period"] = s.date.dt.to_period("M")
    periods = pd.period_range(s.period.min(), s.period.max(), freq="M")
    counts = pd.crosstab(s.Point, s.period).reindex(columns=periods, fill_value=0)
    rows, gap_rows = [], []
    for station, g in s.groupby("Point"):
        dates = g.date.drop_duplicates().sort_values()
        gaps = dates.diff().dt.days.dropna()
        for previous, current in zip(dates.iloc[:-1], dates.iloc[1:]):
            gap_rows.append(
                {
                    "Point": station,
                    "previous_date": previous.date(),
                    "date": current.date(),
                    "gap_days": (current - previous).days,
                }
            )
        row = counts.loc[station]
        rows.append(
            {
                "Point": station,
                "samples": len(g),
                "unique_dates": len(dates),
                "months_observed": int(row.gt(0).sum()),
                "missing_months": ", ".join(str(p) for p in periods[row.eq(0)]),
                "first_date": dates.min().date(),
                "last_date": dates.max().date(),
                "min_gap_days": gaps.min(),
                "median_gap_days": gaps.median(),
                "max_gap_days": gaps.max(),
                "mean_gap_days": gaps.mean(),
                "gap_cv": gaps.std(ddof=0) / gaps.mean() if len(gaps) else np.nan,
                "min_samples_per_observed_month": int(row[row.gt(0)].min()),
                "max_samples_per_month": int(row.max()),
            }
        )
    stations = pd.DataFrame(rows).sort_values(["samples", "Point"], ascending=[False, True])
    gaps = pd.DataFrame(gap_rows)
    monthly = s.groupby("period").agg(
        samples=("Point", "size"), stations=("Point", "nunique"), sampling_dates=("date", "nunique")
    )
    monthly["samples_per_observed_station"] = monthly.samples / monthly.stations
    times = pd.to_datetime(s["Sampling time"], format="%H:%M", errors="coerce")
    hours = times.dt.hour.value_counts().sort_index().rename_axis("hour").rename("samples")
    for name, frame in [
        ("station_summary", stations),
        ("station_month_counts", counts),
        ("sampling_gaps", gaps),
        ("monthly_summary", monthly),
        ("sampling_hours", hours),
    ]:
        frame.to_csv(
            args.output / f"{name}.csv", index=name not in ["station_summary", "sampling_gaps"]
        )
    summary = {
        "samples": len(s),
        "stations": len(stations),
        "years": sorted(s.date.dt.year.unique().tolist()),
        "first_date": str(s.date.min().date()),
        "last_date": str(s.date.max().date()),
        "unique_dates": s.date.nunique(),
        "station_months": int(counts.gt(0).sum().sum()),
        "possible_station_months": int(counts.size),
        "full_coverage_stations": int(counts.gt(0).all(axis=1).sum()),
        "duplicate_station_dates": int(s.duplicated(["Point", "date"]).sum()),
        "date_month_mismatches": int(s.date.dt.month.ne(s.Month).sum()),
        "date_year_mismatches": int(s.date.dt.year.ne(s.Year).sum()),
        "invalid_times": int(times.isna().sum()),
        "earliest_time": str(times.min().time()),
        "latest_time": str(times.max().time()),
        "gap_quantiles": gaps.gap_days.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 1]).to_dict(),
        "gap_bins": pd.cut(
            gaps.gap_days, [0, 10, 20, 40, float("inf")], labels=["1-10", "11-20", "21-40", ">40"]
        )
        .value_counts()
        .sort_index()
        .to_dict(),
        "station_sample_quantiles": stations.samples.quantile([0, 0.25, 0.5, 0.75, 1]).to_dict(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(20, 14), gridspec_kw={"width_ratios": [2.7, 1, 1]})
    ordered = counts.loc[stations.Point]
    im = axes[0].imshow(ordered, aspect="auto", vmin=0, vmax=counts.to_numpy().max(), cmap="YlGnBu")
    axes[0].set_yticks(range(len(ordered)), ordered.index, fontsize=8)
    axes[0].set_xticks(range(len(periods)), [str(p) for p in periods], rotation=60)
    axes[0].set_title("Retained samples per station and month")
    fig.colorbar(im, ax=axes[0], fraction=0.025, label="Samples")
    axes[1].hist(gaps.gap_days, bins=np.arange(0, gaps.gap_days.max() + 6, 5), edgecolor="white")
    axes[1].set(
        xlabel="Days between observations at the same station",
        ylabel="Intervals",
        title="Sampling intervals (5-day bins)",
    )
    axes[2].bar(range(len(monthly)), monthly.samples)
    axes[2].set_xticks(range(len(monthly)), [str(p) for p in monthly.index], rotation=90)
    axes[2].set(title="Retained samples by month", ylabel="Samples")
    fig.tight_layout()
    fig.savefig(args.output / "sampling_calendar.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(stations.to_string(index=False))
    print(monthly.to_string())
    print(hours.to_string())


if __name__ == "__main__":
    main()
