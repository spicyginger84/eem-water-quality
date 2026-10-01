"""Apply the reviewed station-to-Group mapping without reordering EEM samples."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    directory = args.data
    mapping = pd.read_csv(directory / "station_groups.csv")
    if mapping["Point"].duplicated().any() or mapping[["Point", "Group"]].isna().any().any():
        raise ValueError("Station mapping must contain unique stations and nonmissing groups")
    lookup = mapping.set_index("Point")["Group"]
    samples = pd.read_parquet(directory / "samples.parquet")
    csv_samples = pd.read_csv(directory / "samples.csv")
    pd.testing.assert_frame_equal(
        samples.drop(columns="Group", errors="ignore").reset_index(drop=True),
        csv_samples.drop(columns="Group", errors="ignore").reset_index(drop=True),
        check_dtype=False,
    )
    originals = [frame.copy(deep=True) for frame in (samples, csv_samples)]
    for frame in (samples, csv_samples):
        groups = frame["Point"].map(lookup)
        if groups.isna().any():
            missing = frame.loc[groups.isna(), "Point"].unique().tolist()
            raise ValueError(f"Stations missing from mapping: {missing}")
        frame["Group"] = groups
    samples.to_parquet(directory / "samples.parquet", index=False)
    csv_samples.to_csv(directory / "samples.csv", index=False)
    reloaded = [
        pd.read_parquet(directory / "samples.parquet"),
        pd.read_csv(directory / "samples.csv"),
    ]
    for original, result in zip(originals, reloaded):
        pd.testing.assert_frame_equal(
            original.drop(columns="Group", errors="ignore").reset_index(drop=True),
            result.drop(columns="Group").reset_index(drop=True),
        )
    pd.testing.assert_series_equal(reloaded[0]["Group"], reloaded[1]["Group"])
    print(samples.groupby("Group").agg(stations=("Point", "nunique"), samples=("Point", "size")))


if __name__ == "__main__":
    main()
