"""Small pure helpers for the flattened processed EEM representation."""

import numpy as np


def flatten_eem(eem):
    """Flatten ``(sample, emission, excitation)`` EEMs by sample."""
    array = np.asarray(eem)
    if array.ndim != 3:
        raise ValueError("EEM must have shape (samples, emission, excitation)")
    return array.reshape(len(array), -1)


def nonzero_training_mask(eem):
    """Return the train-only mask for columns that contain signal."""
    flat = flatten_eem(eem)
    return np.any(np.abs(flat) > 0, axis=0)

