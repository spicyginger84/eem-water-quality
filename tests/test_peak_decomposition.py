"""Recovery and leakage checks for the experimental notebook models."""

import numpy as np
import pytest
from scipy.optimize import nnls

pytest.importorskip("torch")

from eem_water_quality.features.peak_decomposition import (
    fit_peak_model,
    infer_scores,
    match_surfaces,
    valid_grid,
)


def test_reduced_nnls_matches_direct_with_signed_observations():
    rng = np.random.default_rng(2)
    surfaces = rng.random((3, 150))
    observations = rng.normal(size=(8, 150))
    expected = np.array([nnls(surfaces.T, row)[0] for row in observations])
    np.testing.assert_allclose(infer_scores(observations, surfaces), expected, atol=1e-10)


@pytest.mark.parametrize("family", ["parafac", "gaussian_diag", "gaussian_full"])
def test_recovery_and_fixed_shape_test_inference(family):
    ex, em = np.linspace(220, 400, 15), np.linspace(290, 500, 21)
    xx, yy = np.meshgrid(ex, em)
    shapes = np.array([np.exp(-((xx-270)/25)**2/2 - ((yy-340)/30)**2/2),
                       np.exp(-((xx-350)/30)**2/2 - ((yy-440)/35)**2/2)])
    amplitudes = np.random.default_rng(3).uniform(0.2, 3, (25, 2))
    data = np.einsum("nk,kij->nij", amplitudes, shapes)
    mask = yy > xx + 5
    data[:, ~mask] = 0
    model = fit_peak_model(data[:20], ex, em, family=family, rank=2, mask=mask,
                           n_starts=1, max_steps=250)
    before = model.surfaces.copy()
    inferred = model.transform(data[20:])
    reconstructed = inferred @ model.surfaces
    relative_error = np.linalg.norm(reconstructed-data[20:, mask]) / np.linalg.norm(data[20:, mask])
    assert relative_error < 0.08
    np.testing.assert_array_equal(model.surfaces, before)
    np.testing.assert_allclose(model.surfaces.sum(axis=1), 1)
    _, _, similarities = match_surfaces(shapes[:, mask], model.surfaces)
    assert min(similarities) > 0.9
    # NNLS with frozen basis preserves positive amplitude scaling.
    np.testing.assert_allclose(model.transform(data[20:] * 3), inferred * 3, rtol=1e-5, atol=1e-6)


def test_empirical_mask_uses_only_supplied_training_data():
    train = np.ones((3, 4, 5))
    train[:, 0, 0] = 0
    mask = valid_grid(train)
    assert not mask[0, 0]
    supplied = np.ones((4, 5), dtype=bool)
    assert valid_grid(train, supplied)[0, 0]
