"""Reload a selected run artifact and predict aligned EEM/metadata rows."""

from pathlib import Path

import joblib
import numpy as np


def predict_saved(directory, eem, samples, device="cpu", batch_size=32):
    directory = Path(directory)
    if len(eem) != len(samples) or np.asarray(eem).ndim != 3 or not np.isfinite(eem).all():
        raise ValueError("Expected aligned finite 3D EEM data and metadata")
    eem = np.asarray(eem, dtype=np.float32)
    if (directory / "model.joblib").exists():
        artifact = joblib.load(directory / "model.joblib")
        return artifact["model"].predict(artifact["features"].transform(eem, samples))
    import torch

    from .features import numeric_tabular
    from .neural import build_model, construct_images, make_loader, predict_model

    prep = joblib.load(directory / "preprocessing.joblib")
    checkpoint = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model = build_model(checkpoint["model"], checkpoint["in_channels"], checkpoint["n_tabular"]).to(
        device
    )
    model.load_state_dict(checkpoint["state_dict"])
    images = prep["image_scaler"].transform(construct_images(eem, prep["fft"]))
    columns = prep["tabular_columns"]
    tabular = (
        prep["tabular_scaler"].transform(numeric_tabular(samples, columns))
        if columns
        else np.empty((len(eem), 0))
    )
    loader = make_loader(images, tabular, np.zeros(len(eem)), np.arange(len(eem)), batch_size)
    _, prediction = predict_model(model, loader, device)
    return prep["target_scaler"].inverse_transform(prediction[:, None]).ravel()
