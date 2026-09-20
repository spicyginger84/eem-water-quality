import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y).ravel(), np.asarray(prediction).ravel()
    if len(y) < 2 or y.shape != prediction.shape:
        raise ValueError("Metrics require aligned arrays with at least two observations")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Cannot evaluate nonfinite targets or predictions")
    mse = mean_squared_error(y, prediction)
    return {
        "R2": float(r2_score(y, prediction)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, prediction)),
        "MAPE": float(mean_absolute_percentage_error(y, prediction)),
        "n": len(y),
        "n_zero_targets": int((y == 0).sum()),
    }
