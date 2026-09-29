"""The sparse backend must reproduce the installed PyTransit public model."""
import numpy as np
import pytest

from pentaceratops.experimental.sparse_kernels import (
    orbit_setup, project_strict, project_fast,
)
from pytransit import QuadraticModel


@pytest.mark.parametrize("project", [project_strict, project_fast])
@pytest.mark.parametrize("nsamples", [1, 7, 20])
@pytest.mark.parametrize("radius", [.025, .3, .999, 1.1])
def test_new_orbit_bounds_wrapping_and_exposure_match_public_model(project, nsamples, radius):
    period, t0 = 2., .93
    p = np.array([[radius, t0, period, 12., np.deg2rad(89.5), .6, 1.1, .3, .2, 1.]])
    coefficients, bounds = orbit_setup(p)
    assert coefficients.shape == (1, 2, 5)
    assert bounds.shape == (1, 2)
    # Include both sides of the actual asymmetric support, displaced epochs,
    # shuffled data ordering, and duplicate times (even/odd grids can overlap).
    points = np.r_[np.linspace(-.3, .3, 81),
                   bounds[0]+1e-8, bounds[0]-1e-8, 0., 0.]
    time = t0+points+period*(np.arange(len(points)) % 3-1)
    np.random.default_rng(24).shuffle(time)
    phase = time % period
    order = np.argsort(phase, kind="stable")
    args = (p, coefficients, bounds, time, phase[order], order,
            np.arange(len(time)), np.ones(len(time)), len(time), 30/1440., nsamples)
    signal, _ = project(*args)
    model = QuadraticModel(interpolate=False)
    model.set_data(time, exptimes=30/1440., nsamples=nsamples)
    expected = model.evaluate(k=radius, ldc=[.3, .2], t0=t0, p=period, a=12.,
                              i=p[0, 4], e=p[0, 5], w=p[0, 6])
    np.testing.assert_allclose(1+signal[0], expected, atol=2e-11, rtol=0)


def test_supported_evaluate_has_same_scalar_result_as_old_api():
    # Do not suppress deprecations globally: check this one legacy call only.
    from astropy.utils.exceptions import AstropyDeprecationWarning
    model = QuadraticModel(interpolate=False)
    model.set_data(np.linspace(-.2, .2, 31), exptimes=.002, nsamples=7)
    args = dict(k=.05, ldc=[.3, .2], t0=0., p=3., a=12., i=1.56, e=.3, w=1.2)
    expected = model.evaluate(**args)
    with pytest.warns(AstropyDeprecationWarning, match="evaluate_ps"):
        legacy = model.evaluate_ps(**args)
    np.testing.assert_array_equal(expected, legacy)
