"""Regridded bins keep the samples' true mean time."""
import numpy as np

from pentaceratops.preprocessing.candidate import regular_bin


def test_regular_bin_returns_true_mean_time():
    dt = 30 / 1440
    offset = 7 / 1440                                  # samples sit 7 min after every grid node
    time = 100 + np.arange(50) * dt + offset
    grid, flux, error, true_time = regular_bin(time, np.ones(50), np.full(50, 1e-3), dt)
    good = np.isfinite(flux)
    np.testing.assert_allclose(true_time[good], time, atol=1e-10)
    np.testing.assert_allclose((true_time - grid)[good], offset, atol=1e-10)   # the grid alone would be off by 7 min


def test_regular_bin_true_time_is_mean_of_bin_samples():
    dt = 30 / 1440
    time = 100 + np.arange(30) * (10 / 1440) + 2 / 1440   # 10-min samples averaged into 30-min bins
    grid, flux, error, true_time = regular_bin(time, np.ones(30), np.full(30, 1e-3), dt)
    ids = np.floor((time - grid[0]) / dt + .5).astype(int)
    for i in np.unique(ids):
        np.testing.assert_allclose(true_time[i], time[ids == i].mean(), atol=1e-10)
