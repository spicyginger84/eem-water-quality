"""Recovery and out-of-sample invariants of the improved additive model."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

pytest.importorskip("torch")

from eem_water_quality.features.improved_v0 import fit_improved_v0, peak_initialization


def synthetic():
    ex, em = np.arange(240., 371., 5), np.arange(290., 481., 3)
    xx, yy = np.meshgrid(ex, em)
    centers = np.array([[275, 315], [280, 343], [340, 435]])
    widths = np.array([[9, 9], [11, 12], [20, 25]])
    shapes = np.array([np.exp(-0.5*((xx-c[0])/w[0])**2 - 0.5*((yy-c[1])/w[1])**2)
                       for c, w in zip(centers, widths)])
    rng = np.random.default_rng(6)
    values = np.einsum("nk,kij->nij", rng.uniform(0.1, 3, (30, 3)), shapes) + 0.015
    mask = yy > xx+20
    values[:, ~mask] = 0
    regions = (("B", 270, 280, 305, 325), ("T", 275, 290, 330, 355),
               ("C", 325, 355, 415, 455))
    return ex, em, values, mask, centers, regions


def test_masked_overlapping_peak_recovery_and_frozen_inference():
    ex, em, values, mask, centers, regions = synthetic()
    fit = fit_improved_v0(values[:24], ex, em, regions=regions, n_starts=2,
                         max_steps=300, mask=mask)
    error = np.linalg.norm((fit.reconstruct(values[24:])-values[24:])[:, mask])
    assert error / np.linalg.norm(values[24:, mask]) < 0.08
    costs = np.linalg.norm(centers[:, None]-fit.centers[None], axis=2)
    rows, cols = linear_sum_assignment(costs)
    assert costs[rows, cols].max() < 6
    before = fit.surfaces.copy()
    a = fit.transform(values[24:])
    np.testing.assert_allclose(fit.transform(values[24:]*3), a*3, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(fit.surfaces, before)
    np.testing.assert_allclose(fit.surfaces.sum(1), 1)
    assert np.all(fit.centers[:, 0] >= ex.min())
    assert np.all(fit.centers[:, 0] <= ex.max())
    # No pressure to reproduce zero-filled missing cells as zero.
    assert fit.reconstruct(values[24:])[:, ~mask].mean() > 0


def test_unsupported_region_is_not_reported_as_detected_peak():
    ex, em, values, mask, _, _ = synthetic()
    _, supported = peak_initialization(values, ex, em, mask,
                                        regions=(("missing", 350, 370, 290, 300),))
    assert supported == [False]


def test_mask_only_depends_on_training_and_signed_noise_is_allowed():
    ex, em, values, _mask, _, regions = synthetic()
    values[:20, 0, 0] = -0.002
    fit = fit_improved_v0(values[:20], ex, em, regions=regions, n_starts=1, max_steps=5)
    assert fit.mask[0, 0]
    test = values[20:].copy()
    iy, ix = np.argwhere(~fit.mask)[0]
    test[:, iy, ix] = 100
    before = fit.mask.copy()
    fit.transform(test)
    np.testing.assert_array_equal(fit.mask, before)
