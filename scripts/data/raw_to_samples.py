"""Match the monthly EEM workbooks in a raw data folder to the laboratory workbook.

RAW_DIR layout ("2026ER_data_for Viet"):
  20260120_ER_data_HS.xlsx   sheet "Data" (header on row 2): Label, Point, Month, Round, ... targets
                             sheet "Location" (header on row 3): station metadata
  <N>.<MON>/<label>[-<round>][_<N>].xlsx
                             one EEM per sample: row 1 = excitation, column A = emission

A file is matched only when its label (and round, if present) identifies exactly one laboratory row of
that month; ambiguous files are reported, never guessed. Matched samples must be at most 2 m deep and
have every required parameter. Samples are numbered 0..n-1 in sorted file order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

WORKBOOK = "20260120_ER_data_HS.xlsx"
REQUIRED = ("BOD\n(0.0)", "COD\n(0.0)", "TOC\n(0.0)", "SS\n(0.0)", "EC\n(0)")
TEXT_COLUMNS = {"Label", "Point", "Sampling date", "Sampling time"}
MAX_DEPTH_M = 2.0


def norm(value) -> str:
    return re.sub(r"\s+", "", str(value).strip()).casefold()


def read_eem(path: Path):
    """(matrix, emission, excitation) of one EEM workbook."""
    f = pd.read_excel(path, sheet_name=0, header=None)
    ex = pd.to_numeric(f.iloc[0, 1:], errors="coerce").to_numpy(float)
    em = pd.to_numeric(f.iloc[1:, 0], errors="coerce").to_numpy(float)
    x = f.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isnan(em).any() or np.isnan(ex).any() or not np.isfinite(x).all():
        raise ValueError("non-numeric wavelength or intensity")
    return x.astype(np.float32), em, ex


def parse_name(path: Path, month: int, labels: set):
    """(label key, round or None, method) or (None, reason, None). Exact labels win, so 'Ho1-1' stays one label."""
    stem = path.stem.casefold()
    if m := re.match(r"^(.*)_(\d+)$", stem):
        if int(m.group(2)) != month:
            return None, "filename month suffix disagrees with directory", None
        stem = m.group(1)
    if stem in labels:
        return stem, None, "label_only"
    m = re.match(r"^(.*?)-(\d+)$", stem)
    if not m:
        return None, "station/round pattern not recognised", None
    if m.group(1) not in labels:
        return None, "station label not present in workbook", None
    return m.group(1), int(m.group(2)), "label_and_round"


def match(raw: Path, lab: pd.DataFrame) -> pd.DataFrame:
    labels = set(lab.Label.dropna().map(norm))
    key = lab.Label.map(norm)
    rows = []
    for path in sorted(raw.glob("*/*.xlsx")):
        rel = path.relative_to(raw).as_posix()
        rec = {"relative_path": rel, "filename": path.name, "month": None, "eem_path": rel,
               "mapping_status": "unmatched", "mapping_reason": "", "match_method": "", "excel_row": pd.NA,
               "label": pd.NA, "round": pd.NA}
        rows.append(rec)
        if not (m := re.match(r"^(\d+)\.", path.parent.name)):
            rec["mapping_reason"] = "month directory does not start with N."
            continue
        rec["month"] = month = int(m.group(1))
        label, rnd, method = parse_name(path, month, labels)
        if label is None:
            rec["mapping_reason"] = rnd
            continue
        cand = lab[(key == label) & (lab.Month == month)]
        if rnd is not None:
            cand = cand[cand.Round == rnd]
        if len(cand) != 1:
            rec["mapping_reason"] = ("no laboratory row for station/month/round" if len(cand) == 0 else
                                     "round is absent and multiple laboratory rows match: "
                                     + json.dumps(cand.Round.astype(int).tolist()))
            continue
        row = cand.iloc[0]
        rec |= {"mapping_status": "matched", "mapping_reason": "unique laboratory key", "match_method": method,
                "excel_row": int(row.name) + 3, "label": row.Label, "round": int(row.Round)}
    return pd.DataFrame(rows)


def process(raw: Path, required=REQUIRED):
    """Returns (samples, eem n x Em x Ex, emission, excitation, mapping, locations)."""
    lab = pd.read_excel(raw / WORKBOOK, sheet_name="Data", header=1)
    for c in lab.columns:  # ND and blanks become NaN
        if c not in TEXT_COLUMNS and (v := pd.to_numeric(lab[c], errors="coerce")).notna().any():
            lab[c] = v
    missing = [c for c in ("Label", "Month", "Round", *required) if c not in lab]
    if missing:
        raise ValueError(f"laboratory workbook lacks {missing}")
    locations = pd.read_excel(raw / WORKBOOK, sheet_name="Location", header=2)
    mapping = match(raw, lab)

    x, keep, grid = [], [], None
    for i, rec in mapping[mapping.mapping_status == "matched"].iterrows():
        try:
            m, em, ex = read_eem(raw / rec.eem_path)
        except (OSError, ValueError) as e:
            mapping.loc[i, ["mapping_status", "mapping_reason"]] = ["invalid_eem", str(e)]
            continue
        if grid is None:
            grid = (em, ex)
        elif not (np.array_equal(grid[0], em) and np.array_equal(grid[1], ex)):
            mapping.loc[i, ["mapping_status", "mapping_reason"]] = ["invalid_eem", "wavelength grid differs"]
            continue
        row = lab.loc[rec.excel_row - 3]
        if pd.notna(row.get("Depth")) and float(row.Depth) > MAX_DEPTH_M:
            mapping.loc[i, ["mapping_status", "mapping_reason"]] = [
                "depth_filtered", f"sample depth {row.Depth!s} m exceeds the {MAX_DEPTH_M:g} m limit"]
            continue
        absent = [c for c in required if not np.isfinite(pd.to_numeric(row[c], errors="coerce"))]
        if absent:
            mapping.loc[i, ["mapping_status", "mapping_reason"]] = [
                "incomplete_parameters", "missing required parameter(s): " + ", ".join(absent)]
            continue
        x.append(m)
        keep.append(i)
    if not keep:
        raise RuntimeError("no EEM/laboratory pair passed the filters")

    sel = mapping.loc[keep]
    samples = lab.loc[sel.excel_row.astype(int) - 3].reset_index(drop=True)
    samples.insert(0, "sample_id", np.arange(len(samples), dtype=np.int64))
    for j, (name, col) in enumerate([("eem_filename", "filename"), ("eem_relative_path", "relative_path"),
                                     ("eem_match_method", "match_method"), ("eem_excel_row", "excel_row")]):
        samples.insert(1 + j, name, sel[col].to_numpy())
    samples["eem_excel_row"] = samples.eem_excel_row.astype(int)
    return samples, np.stack(x), grid[0], grid[1], mapping, locations
