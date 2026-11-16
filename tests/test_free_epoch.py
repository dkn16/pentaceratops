"""Free transit-time offset: model shift, sampler coordinate, and public options."""
import numpy as np
import pytest

from pentaceratops.evidence.joint_fourier import (
    DEFAULT_EPOCH_SIGMA, resolve_epoch_sigma, sample_joint_scenario,
)
from pentaceratops.experimental.joint_fourier import JointFourierAdapter
from pentaceratops.likelihoods.joint_fourier import GaussianBlock
from test_joint_fourier import fixture


def test_option_resolution():
    assert resolve_epoch_sigma() is None
    assert resolve_epoch_sigma(True) == DEFAULT_EPOCH_SIGMA == pytest.approx(3/1440)
    assert resolve_epoch_sigma(True, .001) == .001
    assert resolve_epoch_sigma(False) is None
    for arguments in ((False, .001), (True, 0.), (True, -1.), (True, np.inf), (True, True), (1, None)):
        with pytest.raises(ValueError):
            resolve_epoch_sigma(*arguments)


@pytest.mark.parametrize("kind", ["planet", "binary", "x2p"])
def test_offset_equals_a_shifted_epoch_in_scalar_and_batched_models(kind):
    delta = .0031
    adapter, p = fixture(kind)
    rows = [{k: v[i] for k, v in p.items()} for i in range(6)]
    fixed = np.array([adapter.native_model(kind, row) for row in rows])
    adapter.scalar_epoch_offset = delta
    moved = np.array([adapter.native_model(kind, row) for row in rows])
    adapter.scalar_epoch_offset = 0.
    assert np.max(abs(moved-fixed)) > 1e-5
    # Batched rows carry their own offsets and must match the scalar reference.
    offsets = np.array([delta, 0., -delta, delta, 0., 2*delta])
    dense = []
    for row, value in zip(rows, offsets):
        adapter.scalar_epoch_offset = value
        dense.append(adapter.native_model(kind, row))
    adapter.scalar_epoch_offset = 0.
    sparse = adapter._project(kind, dict(p, epoch_offset=offsets))
    np.testing.assert_allclose(sparse, np.array(dense), atol=3e-9, rtol=1e-12)
    np.testing.assert_allclose(adapter.native_model(kind, rows[1]), fixed[1])
    # A later transit is the same model seen at earlier relative times.
    n = 400; t = np.linspace(-.5, 7.5, n)
    late = JointFourierAdapter([GaussianBlock.from_spectrum(t, np.ones(n), np.full(n//2+1, n*.002**2), t[1]-t[0])],
                               4., delta, nsamples=7)
    early = JointFourierAdapter([GaussianBlock.from_spectrum(t, np.ones(n), np.full(n//2+1, n*.002**2), t[1]-t[0])],
                                4., 0., nsamples=7)
    early.scalar_epoch_offset = delta
    np.testing.assert_allclose(early.native_model(kind, rows[2]), late.native_model(kind, rows[2]), atol=1e-12)


@pytest.mark.parametrize("scenario", ["TP", "EB"])
def test_sampler_returns_offsets_and_restores_the_adapter(scenario):
    t = np.linspace(-.3, 7.7, 120)
    block = GaussianBlock.from_covariance(t, np.ones(len(t)), np.eye(len(t))*.1**2, .003)
    adapter = JointFourierAdapter([block], 4., 0.)
    star = dict(mass=1., rad=1., Teff=5700., plx=10., Tmag=10., Jmag=9., Hmag=9., Kmag=9.)
    fixed = sample_joint_scenario(adapter, scenario, star, trilegal="unused", N=8, steps=2,
                                  seed=307, posterior_samples=19)
    assert "epoch_offset" not in fixed
    sigma = .002
    result = sample_joint_scenario(adapter, scenario, star, trilegal="unused", N=8, steps=2,
                                   seed=307, posterior_samples=19, epoch_sigma=sigma)
    assert np.isfinite(result["lnBF"])
    assert result["epoch_sigma"] == sigma
    assert result["epoch_offset"].shape == (len(result["physical_parameters"]),)
    assert result["physical_parameters"].shape[1] == fixed["physical_parameters"].shape[1]
    np.testing.assert_array_equal(result["posterior_epoch_offset"],
                                  result["epoch_offset"][result["posterior_source_index"]])
    # Uninformative data: the offsets keep the scale of their prior.
    assert np.all(abs(result["epoch_offset"]) < 8*sigma)
    assert result["bestfit"]["scalar_replay"] == pytest.approx(result["bestfit"]["log_target"], abs=2e-5)
    assert "epoch_offset" in result["bestfit"]["parameters"]
    assert adapter.epoch_sigma is None and adapter.scalar_epoch_offset == 0.


def test_public_options_default_off_for_fourier_and_rejected_for_real():
    from pentaceratops.prepared import _options
    options = _options("fourier", {})
    assert options["free_epoch"] is False and options["epoch_sigma"] is None
    assert _options("fourier", dict(free_epoch=True))["free_epoch"] is True
    assert _options("fourier", dict(free_epoch=True, epoch_sigma=.001))["epoch_sigma"] == .001
    for settings in (dict(epoch_sigma=.001), dict(free_epoch=True, epoch_sigma=-1.), dict(free_epoch="yes")):
        with pytest.raises(ValueError):
            _options("fourier", settings)
    assert "free_epoch" not in _options("real", {})
    for settings in (dict(free_epoch=True), dict(epoch_sigma=.001)):
        with pytest.raises(ValueError):
            _options("real", settings)
