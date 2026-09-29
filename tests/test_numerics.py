import numpy as np
import pytest


@pytest.mark.parametrize("length", [7, 8, 31, 32])
def test_fourier_white_noise_parseval(length):
    from pentaceratops.likelihoods.fourier import _rfft_complex_bins, _fourier_chi2
    residual = np.random.default_rng(14).normal(size=length)
    transform = np.fft.rfft(residual)
    transform[0] = 0
    if length % 2 == 0:
        transform[-1] = 0
    projected = np.fft.irfft(transform, n=length)
    coeff = _rfft_complex_bins(residual)
    assert len(coeff) == (length-1)//2
    sigma = .7
    assert _fourier_chi2(coeff, 0, np.full(len(coeff), length*sigma**2)) == pytest.approx(
        .5*np.sum(projected**2/sigma**2))
    np.testing.assert_allclose(_rfft_complex_bins(residual+8), coeff, atol=1e-13)


def test_evenodd_sort_trimming_errors_and_no_input_mutation():
    from pentaceratops._target import _concatenate_even_odd_observations
    even = np.array([.9, .8, .7])
    odd = np.array([.6, .5])
    time, flux, sigma = _concatenate_even_odd_observations(
        [-1, 0, 1], even, [.01, .02, .03], [-1, 0], odd, .04)
    np.testing.assert_array_equal(time, [-1, -1, 0, 0])
    np.testing.assert_array_equal(flux, [.9, .6, .8, .5])
    np.testing.assert_array_equal(sigma, [.01, .04, .02, .04])
    np.testing.assert_array_equal(even, [.9, .8, .7])
    np.testing.assert_array_equal(odd, [.6, .5])


def test_evenodd_unequal_windows_reversed_assignment(monkeypatch):
    from pentaceratops.likelihoods import real
    def model(t_even, t_odd, *args, **kwargs):
        return np.full(len(t_even), .8), np.full(len(t_odd), .95)
    monkeypatch.setattr(real, "simulate_EB_transit_evenodd", model)
    monkeypatch.setattr(real, "mean_anomaly_difference", lambda *args: .5)
    cost = real.lnL_EB_evenodd(
        np.array([-.1, 0, .1]), np.full(3, .95), .01,
        np.array([-.1, .1]), np.full(2, .8), .01,
        .2, .1, 5., 89., 1e12, 1., .3, .2, 0., 90.)
    assert cost == 0


def test_dilution_restores_flux_and_errors_once():
    from pentaceratops.preprocessing.flux import restore_aperture_flux, require_aperture_frame
    data = np.array([1., .98])
    flux, error = restore_aperture_flux(data, [.01, .01], .4)
    np.testing.assert_allclose(flux, [1., .992])
    np.testing.assert_allclose(error, [.004, .004])
    np.testing.assert_array_equal(data, [1., .98])
    with pytest.raises(ValueError, match="Regenerate"):
        require_aperture_frame({})


def test_covariance_likelihood_is_same_data_null_ratio():
    from pentaceratops.experimental.covariance import GaussianMetric
    flux = np.array([1.02, .96, 1.01])
    sigma = np.array([.02, .03, .02])
    factor = np.array([[.01], [.02], [.01]])
    model = np.array([1., .97, 1.])
    metric = GaussianMetric(flux, sigma, factor)
    covariance = np.diag(sigma**2) + factor@factor.T
    residual = flux-model
    null = flux-1
    expected = -.5*(residual@np.linalg.solve(covariance, residual)
                     - null@np.linalg.solve(covariance, null))
    assert metric.gain(model) == pytest.approx(expected)
    assert metric.gain(np.ones(3)) == 0
    # Affine host-frame changes cancel when data, model and covariance all change.
    rescaled = GaussianMetric(1+(flux-1)/.2, sigma/.2, factor/.2)
    assert rescaled.gain(1+(model-1)/.2) == pytest.approx(expected)


def test_fgp_posterior_against_dense_conditioning():
    from pentaceratops.preprocessing.posterior import conditional_posterior
    values = np.sin(np.arange(16))*.01
    mask = np.ones(16, dtype=bool)
    mask[5:9] = False
    psd = np.full(8, .03)
    variance = .002
    posterior = conditional_posterior(values, mask, psd, variance, dt=1., cutoff=.15)
    design = posterior.design
    prior = np.diag(posterior.prior_sigma**2)
    expected_cov = np.linalg.inv(np.linalg.inv(prior) + design[mask].T@design[mask]/variance)
    expected_mean = expected_cov@design[mask].T@values[mask]/variance
    np.testing.assert_allclose(posterior.coefficient_mean, expected_mean)
    np.testing.assert_allclose(posterior.coefficient_factor@posterior.coefficient_factor.T, expected_cov)
    changed = values.copy()
    changed[~mask] = 1e6
    second = conditional_posterior(changed, mask, psd, variance, 1., .15)
    np.testing.assert_array_equal(posterior.coefficient_mean, second.coefficient_mean)
