"""Run provenance and portable prediction tables."""

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_DIRECTORY_NAMES = {
    "BOD": "BOD",
    "BOD\n(0.0)": "BOD",
    "COD": "COD",
    "COD\n(0.0)": "COD",
    "TOC": "TOC",
    "TOC\n(0.0)": "TOC",
    "BOD_COD": "BOD_COD",
}


def target_directory_name(target: str) -> str:
    """Return a stable, readable directory name for a target argument."""
    if target in TARGET_DIRECTORY_NAMES:
        return TARGET_DIRECTORY_NAMES[target]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(target)).strip("_")
    return safe or "target"


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Cannot serialize {type(obj)}")

    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def start_run(args, samples, splits):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    repo_dir = Path(__file__).resolve().parents[2]
    versions = {}
    for name in [
        "eem-water-quality",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "pyarrow",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    hashes = {}
    for name in ["eem.npy", "samples.parquet", "wavelengths.npz"]:
        path = Path(args.data) / name
        if path.exists():
            with path.open("rb") as file:
                hashes[name] = hashlib.file_digest(file, "sha256").hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_dir, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    write_json(
        output / "run.json",
        {
            "config": vars(args),
            "versions": versions,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "data_sha256": hashes,
            "git_commit": commit,
            "git_dirty": dirty,
        },
    )
    if isinstance(splits, dict):
        split_records = [{"fold": 1, **splits}]
    else:
        split_records = list(splits)
    split_rows = []
    for spec in split_records:
        fold = spec.get("fold", 1)
        for partition in ("train", "validation", "test", "excluded"):
            indices = np.asarray(spec.get(partition, []), dtype=int)
            if indices.size == 0:
                continue
            selected = samples.iloc[indices].copy()
            selected.insert(0, "row_position", indices)
            selected.insert(1, "fold", fold)
            selected.insert(2, "partition", partition)
            selected.insert(
                3,
                "heldout_group",
                [spec.get("heldout_group", "")] * len(selected),
            )
            split_rows.append(selected)
    split_frame = (
        pd.concat(split_rows, ignore_index=True)
        if split_rows
        else pd.DataFrame(columns=["row_position", "fold", "partition", "heldout_group"])
    )
    split_frame.to_csv(output / "splits.csv", index=False)
    return output


def save_predictions(path, samples, indices, y, prediction):
    frame = samples.iloc[indices].copy()
    frame.insert(0, "row_position", indices)
    frame["y_true"] = y
    frame["y_pred"] = prediction
    frame.to_csv(path, index=False)
