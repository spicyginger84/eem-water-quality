"""Reload a saved fold artifact and predict aligned EEM/metadata rows."""

from pathlib import Path

import joblib
import numpy as np


def predict_saved(
    directory,
    eem,
    samples,
    device="cpu",
    batch_size=32,
    feature_name=None,
    model_name=None,
    fold=1,
):
    directory = Path(directory)
    if len(eem) != len(samples) or np.asarray(eem).ndim != 3 or not np.isfinite(eem).all():
        raise ValueError("Expected aligned finite 3D EEM data and metadata")
    eem = np.asarray(eem, dtype=np.float32)
    model_path = directory / "model.joblib"
    if not model_path.exists():
        # Classical CV artifacts are stored at the run root, while callers
        # commonly pass the target directory (for example ``run/BOD``).
        target_name = directory.name
        root = directory.parent if directory.name in {"BOD", "COD", "TOC", "BOD_COD"} else directory
        pattern = f"models/{target_name}/*/*/fold_{int(fold):02d}/model.joblib"
        candidates = sorted(root.glob(pattern))
        if feature_name is not None:
            candidates = [path for path in candidates if path.parts[-4] == feature_name]
        if model_name is not None:
            candidates = [path for path in candidates if path.parts[-3] == model_name]
        if len(candidates) == 1:
            model_path = candidates[0]
        elif candidates:
            raise ValueError(
                "Multiple classical models are available; specify feature_name and model_name"
            )
    if model_path.exists():
        artifact = joblib.load(model_path)
        return artifact["model"].predict(artifact["features"].transform(eem, samples))
    raise FileNotFoundError(
        "No classical model artifact found. Specify feature_name/model_name/fold "
        "for a nested CV artifact."
    )
