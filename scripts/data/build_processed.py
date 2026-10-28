"""Build data/processed from the raw data folder and the curated decisions in data/manifest.

  python scripts/data/build_processed.py RAW_DIR [--out data/processed] [--zip data/processed.zip]

Steps:
  1. raw_to_samples.process: match EEM files to laboratory rows, depth <= 2 m, required parameters present.
  2. data/manifest/exclusions.csv: drop listed EEM files and stations.
  3. Station-CV folds (station_folds): 5 folds grouped by station, stratified by station type (industrial
     stream or river/lake, from the Location sheet) x brackish (station median EC > 2000 uS/cm), balanced in
     sample count. station_cv_fold uses seed 0; station_cv_fold_gkf5_seed{11,23,42,2026,3407} are alternatives.
  4. data/manifest/label_flags.csv: label_flag for laboratory rows kept but flagged (Point + Sampling date).
  5. Fixed EEM mask: cells that are exactly zero in every kept EEM (one 4-connected corner) are excluded.
  6. Write samples.csv/.parquet, eem.npy, wavelengths.npz, eem_fixed_mask.npy/.json, eem_mapping.csv,
     locations.csv/.parquet, station_cv_folds.csv, processing_summary.json, and the zip archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_to_samples import MAX_DEPTH_M, REQUIRED, WORKBOOK, process

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifest"
N_FOLDS = 5
FOLD_SEEDS = {"station_cv_fold": 0} | {f"station_cv_fold_gkf5_seed{s}": s for s in (11, 23, 42, 2026, 3407)}
BRACKISH_EC = 2000.0  # uS/cm, station median; same rule as eem_water_quality.data


def fixed_mask(x):
    zero = np.all(x == 0, axis=0)
    comp, count = label(zero)
    if count != 1 or comp[0, -1] == 0:
        raise ValueError("shared zeros must form one component at Em-min / Ex-max")
    return ~zero


def station_strata(samples, locations):
    """Stratum per station: (industrial stream?, brackish?)."""
    key = lambda s: str(s).replace(" ", "").casefold()
    site = "Point.1" if "Point.1" in locations else "Point"
    kind = {key(p): c for p, c in zip(locations[site], locations.Classification)}
    st = samples.groupby("Point").agg(n=("sample_id", "size"), ec=("EC\n(0)", "median"))
    unknown = [p for p in st.index if key(p) not in kind]
    if unknown:
        raise ValueError(f"stations without a Classification in the Location sheet: {unknown}")
    st["stratum"] = [f"{'industrial' if kind[key(p)] == 'Industrial stream' else 'river_lake'}"
                     f"|{'brackish' if ec > BRACKISH_EC else 'fresh'}" for p, ec in zip(st.index, st.ec)]
    return st


def station_folds(st, seed, k=N_FOLDS):
    """Fold 1..k per station. Strata are dealt in turn (largest stratum first); within a stratum the largest
    station goes to the fold holding the fewest stations of that stratum, then the fewest samples. The seed
    only breaks ties between stations of equal size, so every fold gets an even share of each station type."""
    order = st.iloc[np.random.default_rng(seed).permutation(len(st))]
    samples, fold = np.zeros(k), {}
    for _, grp in sorted(order.groupby("stratum", sort=False), key=lambda g: -len(g[1])):
        count = np.zeros(k)
        for point, n in grp.n.sort_values(ascending=False, kind="stable").items():
            f = min(range(k), key=lambda i: (count[i], samples[i], i))
            fold[point], count[f], samples[f] = f + 1, count[f] + 1, samples[f] + n
    return fold


def build(raw: Path, out: Path, archive: Path):
    samples, x, em, ex, mapping, locations = process(raw)
    n_matched = len(samples)

    exc = pd.read_csv(MANIFEST / "exclusions.csv")
    files = exc[exc.kind == "eem_file"].set_index("key").reason
    stations = exc[exc.kind == "station"].set_index("key").reason
    missing = (set(files.index) - set(samples.eem_relative_path)) | (set(stations.index) - set(samples.Point))
    if missing:
        raise ValueError(f"exclusions not found in the matched samples: {sorted(missing)}")
    drop = samples.eem_relative_path.isin(files.index) | samples.Point.isin(stations.index)
    samples, x = samples[~drop].reset_index(drop=True), x[~drop.to_numpy()]

    strata = station_strata(samples, locations)
    for col, seed in FOLD_SEEDS.items():
        samples[col] = samples.Point.map(station_folds(strata, seed)).astype(int)
    flags = pd.read_csv(MANIFEST / "label_flags.csv")
    key = samples.Point + "|" + samples["Sampling date"].astype(str)
    flag = dict(zip(flags.Point + "|" + flags["Sampling date"], flags.label_flag))
    if set(flag) - set(key):
        raise ValueError(f"label flags match no sample: {sorted(set(flag) - set(key))}")
    samples["label_flag"] = key.map(flag)

    mapping["included_in_processed"] = mapping.relative_path.isin(samples.eem_relative_path)
    mapping["exclusion_reason"] = mapping.relative_path.map(files)
    by_station = (mapping.mapping_status == "matched") & ~mapping.included_in_processed & \
        mapping.exclusion_reason.isna()
    mapping.loc[by_station, "exclusion_reason"] = "excluded station (data/manifest/exclusions.csv)"

    mask = fixed_mask(x)
    folds = (samples.groupby(["Point", "station_cv_fold"], as_index=False).size()
             .rename(columns={"size": "n_samples"}).sort_values(["station_cv_fold", "Point"]))
    site = "Point.1" if "Point.1" in locations else "Point"
    locations = locations[~locations[site].isin(stations.index)]
    summary = {
        "raw_dir": raw.name, "laboratory_workbook": WORKBOOK, "required_parameters": list(REQUIRED),
        "max_depth_m": MAX_DEPTH_M, "mapping_status_counts": mapping.mapping_status.value_counts().to_dict(),
        "matched_samples": n_matched, "accepted_samples": len(samples), "eem_shape": list(x.shape),
        "station_count": int(samples.Point.nunique()), "station_cv_folds": int(samples.station_cv_fold.nunique()),
        "excluded_eem_files": files.to_dict(), "excluded_stations": stations.to_dict(),
        "label_flags": int(samples.label_flag.notna().sum()), "masked_cells": int((~mask).sum()),
    }

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        samples.to_csv(stage / "samples.csv", index=False)
        samples.to_parquet(stage / "samples.parquet", index=False)
        np.save(stage / "eem.npy", x)
        np.savez(stage / "wavelengths.npz", excitation=ex, emission=em)
        np.save(stage / "eem_fixed_mask.npy", mask)
        (stage / "eem_fixed_mask.json").write_text(json.dumps({
            "source_sha256": hashlib.sha256((stage / "eem.npy").read_bytes()).hexdigest(),
            "shape": list(mask.shape), "axes": ["emission", "excitation"], "true": "keep",
            "false": "exactly zero in every EEM (shared calibration corner)",
            "excluded_cells": int((~mask).sum())}, indent=2) + "\n")
        mapping.to_csv(stage / "eem_mapping.csv", index=False)
        locations.to_csv(stage / "locations.csv", index=False)
        locations.to_parquet(stage / "locations.parquet", index=False)
        folds.to_csv(stage / "station_cv_folds.csv", index=False)
        (stage / "processing_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(stage.iterdir()):
            shutil.copy2(f, out / f.name)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file():
                z.write(f, f"processed/{f.name}")
    print(json.dumps({k: summary[k] for k in ("mapping_status_counts", "matched_samples", "accepted_samples",
                                              "station_count", "label_flags", "masked_cells")}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw", type=Path, help='unzipped raw folder ("2026ER_data_for Viet")')
    p.add_argument("--out", type=Path, default=ROOT / "data/processed")
    p.add_argument("--zip", type=Path, default=None, help="archive path (default: <out>.zip)")
    a = p.parse_args()
    build(a.raw, a.out, a.zip or a.out.with_suffix(".zip"))


if __name__ == "__main__":
    main()
