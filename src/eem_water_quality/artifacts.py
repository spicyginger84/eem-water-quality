"""Run provenance and portable prediction tables."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np


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
    versions = {}
    for name in [
        "eem-water-quality",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "tensorly",
        "torch",
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
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
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
    frame = samples.copy()
    frame.insert(0, "row_position", np.arange(len(frame)))
    frame["partition"] = ""
    for name, indices in splits.items():
        frame.loc[indices, "partition"] = name
    frame.to_csv(output / "splits.csv", index=False)
    return output


def save_predictions(path, samples, indices, y, prediction):
    frame = samples.iloc[indices].copy()
    frame.insert(0, "row_position", indices)
    frame["y_true"] = y
    frame["y_pred"] = prediction
    frame.to_csv(path, index=False)
