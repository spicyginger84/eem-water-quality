"""Align monthly EEM matrices with the laboratory workbook.

The input directory contains one EEM workbook per sample and a laboratory
workbook with the target parameters.  The script deliberately rejects an EEM
file when its station/round cannot be identified uniquely.  In particular,
files such as ``ha20_3.xlsx`` are not assigned to an arbitrary round.

Outputs are suitable for :func:`eem_water_quality.data.load_data`:

    eem.npy, samples.parquet, wavelengths.npz

Auditable mapping tables are written alongside them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MONTH_RE = re.compile(r"^(\d+)\.", re.IGNORECASE)
STEM_MONTH_RE = re.compile(r"^(.*)_(\d+)$")
ROUND_RE = re.compile(r"^(.*?)-(\d+)$")
DEFAULT_REQUIRED = ("BOD\n(0.0)", "COD\n(0.0)", "TOC\n(0.0)", "SS\n(0.0)", "EC\n(0)")
MAX_DEPTH_M = 2.0


def norm_label(value: object) -> str:
    return re.sub(r"\s+", "", str(value).strip()).casefold()


def month_from_dir(path: Path) -> int | None:
    match = MONTH_RE.match(path.name)
    return int(match.group(1)) if match else None


def read_eem(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an EEM workbook as ``(matrix, emission, excitation)``.

    The first column contains the row wavelengths (emission, 280--550 nm)
    and the first row contains the column wavelengths (excitation,
    220--500 nm).  The matrix is kept in its original row/column order.
    """
    frame = pd.read_excel(path, sheet_name=0, header=None)
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError("workbook has no matrix")
    excitation = pd.to_numeric(frame.iloc[0, 1:], errors="coerce").to_numpy(float)
    emission = pd.to_numeric(frame.iloc[1:, 0], errors="coerce").to_numpy(float)
    matrix = frame.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isnan(emission).any() or np.isnan(excitation).any():
        raise ValueError("wavelength header contains non-numeric values")
    if matrix.shape != (len(emission), len(excitation)):
        raise ValueError("matrix and wavelength dimensions disagree")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains NaN/Inf")
    return matrix, emission, excitation


def _parse_name(path: Path, month: int, labels: dict[str, str]) -> dict[str, object]:
    """Return station and optional round parsed from a filename.

    Exact workbook labels have priority.  This matters for labels such as
    ``Ho1-1``; otherwise ``ha1-1`` is interpreted as station Ha1, round 1.
    """
    stem = path.stem.casefold()
    stem_match = STEM_MONTH_RE.match(stem)
    if stem_match and int(stem_match.group(2)) == month:
        stem = stem_match.group(1)
    elif stem_match and int(stem_match.group(2)) != month:
        return {"status": "unmatched", "reason": "filename month suffix disagrees with directory"}
    if stem in labels:
        return {"status": "candidate", "label_key": stem, "round": None, "method": "label_only"}
    round_match = ROUND_RE.match(stem)
    if not round_match:
        return {"status": "unmatched", "reason": "station/round pattern not recognised"}
    station, round_text = round_match.groups()
    if station not in labels:
        return {"status": "unmatched", "reason": "station label not present in workbook"}
    return {
        "status": "candidate",
        "label_key": station,
        "round": int(round_text),
        "method": "label_and_round",
    }


def map_eem_files(raw_dir: Path, lab: pd.DataFrame) -> pd.DataFrame:
    labels = {norm_label(x): str(x) for x in lab["Label"].dropna().unique()}
    records: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("*/*.xlsx")):
        month = month_from_dir(path.parent)
        record: dict[str, object] = {
            "relative_path": path.relative_to(raw_dir).as_posix(),
            "filename": path.name,
            "month": month,
            # Keep provenance portable: the raw root is supplied on the
            # command line, so only the path below that root is persisted.
            "eem_path": path.relative_to(raw_dir).as_posix(),
            "mapping_status": "unmatched",
            "mapping_reason": "",
            "match_method": "",
            "excel_row": pd.NA,
            "label": pd.NA,
            "round": pd.NA,
        }
        if month is None:
            record["mapping_reason"] = "month directory does not start with N."
            records.append(record)
            continue
        parsed = _parse_name(path, month, labels)
        if parsed["status"] != "candidate":
            record["mapping_reason"] = parsed["reason"]
            records.append(record)
            continue
        station_key = str(parsed["label_key"])
        candidate = lab[lab["Label"].map(norm_label).eq(station_key) & lab["Month"].eq(month)]
        if parsed["round"] is not None:
            candidate = candidate[candidate["Round"].eq(parsed["round"])]
        if len(candidate) == 0:
            record["mapping_reason"] = "no laboratory row for station/month/round"
            records.append(record)
            continue
        if len(candidate) > 1:
            record["mapping_reason"] = (
                "round is absent and multiple laboratory rows match: "
                + json.dumps(candidate["Round"].astype(int).tolist())
            )
            records.append(record)
            continue
        row = candidate.iloc[0]
        record.update(
            {
                "mapping_status": "matched",
                "mapping_reason": "unique laboratory key",
                "match_method": parsed["method"],
                "excel_row": int(row.name) + 3,  # Excel row: title row + header row
                "label": row["Label"],
                "round": int(row["Round"]),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def process(
    raw_dir: Path,
    output_dir: Path,
    required: tuple[str, ...],
    reject_bod_greater_cod: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    workbook = raw_dir / "20260120_ER_data_HS.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(workbook)
    lab = pd.read_excel(workbook, sheet_name="Data", header=1)
    if not {"Label", "Month", "Round"}.issubset(lab.columns):
        raise ValueError("Data sheet must contain Label, Month and Round columns")
    for col in required:
        if col not in lab.columns:
            raise ValueError(f"required parameter is absent from workbook: {col!r}")
    location = pd.read_excel(workbook, sheet_name="Location", header=2)
    # Convert all known numeric parameters.  ND and blanks become NaN, while
    # preserving the original Excel values in the provenance table.
    numeric_lab = lab.copy()
    for col in lab.columns:
        if col not in {"Label", "Point", "Sampling date", "Sampling time"}:
            converted = pd.to_numeric(lab[col], errors="coerce")
            if converted.notna().any():
                numeric_lab[col] = converted
    mapping = map_eem_files(raw_dir, numeric_lab)
    output_dir.mkdir(parents=True, exist_ok=True)
    location.to_csv(output_dir / "locations.csv", index=False)
    location.to_parquet(output_dir / "locations.parquet", index=False)
    mapping.to_csv(output_dir / "eem_mapping.csv", index=False)

    matrices: list[np.ndarray] = []
    rows: list[pd.Series] = []
    accepted: list[int] = []
    emission: np.ndarray | None = None
    excitation: np.ndarray | None = None
    for map_index, item in mapping.iterrows():
        if item["mapping_status"] != "matched":
            continue
        path = raw_dir / str(item["eem_path"])
        try:
            matrix, em, ex = read_eem(path)
        except (OSError, ValueError, ImportError) as exc:
            mapping.loc[map_index, "mapping_status"] = "invalid_eem"
            mapping.loc[map_index, "mapping_reason"] = str(exc)
            continue
        if emission is None:
            emission, excitation = em, ex
        elif not (np.array_equal(emission, em) and np.array_equal(excitation, ex)):
            mapping.loc[map_index, "mapping_status"] = "invalid_eem"
            mapping.loc[map_index, "mapping_reason"] = "wavelength grid differs from first valid EEM"
            continue
        lab_rows = numeric_lab[
            numeric_lab.index == int(item["excel_row"]) - 3
        ]
        if len(lab_rows) != 1:
            mapping.loc[map_index, "mapping_status"] = "unmatched"
            mapping.loc[map_index, "mapping_reason"] = "laboratory row disappeared during validation"
            continue
        row = lab_rows.iloc[0]
        depth = pd.to_numeric(row.get("Depth"), errors="coerce")
        if pd.notna(depth) and float(depth) > MAX_DEPTH_M:
            mapping.loc[map_index, "mapping_status"] = "depth_filtered"
            mapping.loc[map_index, "mapping_reason"] = (
                f"sample depth {depth!s} m exceeds the {MAX_DEPTH_M:g} m limit"
            )
            continue
        missing = []
        for col in required:
            value = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(value) or not np.isfinite(float(value)):
                missing.append(col)
        if missing:
            mapping.loc[map_index, "mapping_status"] = "incomplete_parameters"
            mapping.loc[map_index, "mapping_reason"] = "missing required parameter(s): " + ", ".join(missing)
            continue
        if reject_bod_greater_cod and "BOD\n(0.0)" in required and "COD\n(0.0)" in required:
            bod = float(row["BOD\n(0.0)"])
            cod = float(row["COD\n(0.0)"])
            if bod > cod:
                mapping.loc[map_index, "mapping_status"] = "quality_filtered"
                mapping.loc[map_index, "mapping_reason"] = "BOD > COD"
                continue
        matrices.append(matrix.astype(np.float32))
        rows.append(row)
        accepted.append(map_index)

    mapping.to_csv(output_dir / "eem_mapping.csv", index=False)
    if matrices:
        samples = pd.DataFrame(rows).reset_index(drop=True)
        selected = mapping.loc[accepted].reset_index(drop=True)
        samples.insert(0, "sample_id", np.arange(len(samples), dtype=np.int64))
        samples.insert(1, "eem_filename", selected["filename"])
        samples.insert(2, "eem_relative_path", selected["relative_path"])
        samples.insert(3, "eem_match_method", selected["match_method"])
        samples.insert(4, "eem_excel_row", selected["excel_row"].astype(int))
        np.save(output_dir / "eem.npy", np.stack(matrices))
        samples.to_parquet(output_dir / "samples.parquet", index=False)
        np.savez(output_dir / "wavelengths.npz", excitation=excitation, emission=emission)
    else:
        raise RuntimeError("no EEM/laboratory samples satisfy the mapping and parameter filters")
    samples.to_csv(output_dir / "samples.csv", index=False)
    summary = {
        "raw_dir": str(raw_dir),
        "laboratory_workbook": str(workbook),
        "required_parameters": list(required),
        "max_depth_m": MAX_DEPTH_M,
        "reject_bod_greater_cod": reject_bod_greater_cod,
        "mapping_status_counts": mapping["mapping_status"].value_counts(dropna=False).to_dict(),
        "accepted_samples": len(samples),
        "eem_shape": list(np.stack(matrices).shape),
    }
    (output_dir / "processing_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return mapping, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/2026ER_data_for Viet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--required", nargs="+", default=list(DEFAULT_REQUIRED))
    parser.add_argument(
        "--reject-bod-greater-cod",
        action="store_true",
        help="exclude rows where BOD is greater than COD",
    )
    args = parser.parse_args()
    mapping, samples = process(
        args.raw,
        args.output,
        tuple(args.required),
        reject_bod_greater_cod=args.reject_bod_greater_cod,
    )
    print(f"accepted samples: {len(samples)}")
    print(mapping["mapping_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
