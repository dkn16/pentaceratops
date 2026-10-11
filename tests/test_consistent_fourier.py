"""Regression tests for common-data folded Fourier evidence weights."""
from types import SimpleNamespace
import importlib

import numpy as np
import pandas as pd
import pytest

from pentaceratops.likelihoods.folded_fourier import prepare_folded_metric, complex_bins
from pentaceratops.experimental.uniform_fourier import UniformFourierAdapter
from test_all_scenarios import kwargs_for, population, cube


def metric_data(n=101, period=5., parity=True):
    rng = np.random.default_rng(9271)
    even = 1+rng.normal(0, .002, n)
    odd = 1+rng.normal(0, .003, n)
    # Deliberately different spectral shapes; constant transit-count weights
    # cannot reproduce this precision-weighted combination.
    k = np.arange(n//2)+1
    pe = .003*(1+20/k)
    po = .008*(1+np.sin(k*.8)**2)
    full = (3*even+2*odd)/5
    return prepare_folded_metric(full, (pe+po)/2, .37*period, period/n, period, 5,
        **(dict(parity_flux=np.r_[even, odd], n_even=3, n_odd=2,
                psd_even=pe, psd_odd=po) if parity else {}))


@pytest.mark.parametrize('n', [31, 32, 101, 102])
@pytest.mark.parametrize('parity', [False, True])
def test_dense_fft_sparse_and_combined_equivalence(n, parity):
    metric = metric_data(n=n, parity=parity)
    rng = np.random.default_rng(847)
    models = np.ones((8, n))
    models[:, :6] -= rng.uniform(0, .02, (8, 6))
    repeated = np.tile(models, (1, len(metric.blocks)))
    expected = np.array([metric.fft_gain(model) for model in repeated])
    np.testing.assert_allclose(metric.gains(repeated), expected, atol=2e-10, rtol=2e-12)
    np.testing.assert_allclose(metric.common_gains(models), expected, atol=2e-10, rtol=2e-12)
    m = complex_bins(models-1.)
    compressed = np.sum((2*np.real(np.conj(metric.combined_coefficients)*m)-abs(m)**2)
                        /metric.combined_variance, axis=1)
    np.testing.assert_allclose(compressed, expected, atol=2e-10, rtol=2e-12)
    # Independent dense covariance precision, including the same omitted modes.
    direct = np.zeros(len(models))
    for b in metric.blocks:
        fourier = np.exp(-2j*np.pi*np.outer(np.arange(1, len(b.variance)+1), np.arange(n))/n)
        q = 2*np.real(fourier.conj().T @ (fourier/b.variance[:, None]))
        signal = models-1
        direct += signal @ q @ (b.flux-1) - .5*np.einsum('bi,ij,bj->b', signal, q, signal)
    np.testing.assert_allclose(direct, expected, atol=2e-10, rtol=2e-12)
    if parity:
        # An alternating mean contains information discarded by simple pooling.
        alternating = repeated.copy()
        alternating[:, n:n+6] -= .01
        np.testing.assert_allclose(metric.gains(alternating),
            [metric.fft_gain(row) for row in alternating], atol=2e-10, rtol=2e-12)
        assert np.max(abs(metric.gains(alternating)-expected)) > .1


def test_invalid_data_fail_without_discarding_observations():
    p, n = 5., 31
    kwargs = dict(flux=np.ones(n), psd=np.ones(n//2), phase=0., dt=p/n, period=p, ntransits=4)
    for change in (dict(psd=np.zeros(n//2)), dict(flux=np.full(n, np.nan)),
                   dict(parity_flux=np.ones(2*n)), dict(dt=.1), dict(psd_even=np.ones(n//2))):
        with pytest.raises(ValueError):
            prepare_folded_metric(**dict(kwargs, **change))


SCENARIOS = [('fourier', f'lnZ_{family}_fourier') for family in ('TTP', 'PTP', 'STP', 'DTP', 'BTP')]
SCENARIOS += [('fourier_eclipses', f'lnZ_{family}_{kind}_fourier')
              for family in ('TEB', 'PEB', 'SEB', 'DEB', 'BEB') for kind in ('secondary', 'evenodd')]


@pytest.mark.parametrize('module_name,name', SCENARIOS)
@pytest.mark.parametrize('timing_policy', ['observed', 'legacy'])
def test_all_recipes_share_metric_and_scalar_batch_physics(monkeypatch, tmp_path, module_name, name, timing_policy):
    module = importlib.import_module('pentaceratops.evidence.'+module_name)
    function = getattr(module, name)
    kwargs = kwargs_for(function, 'default', tmp_path)
    metric = metric_data()
    kwargs.update(exptime=5/101, nsamples=3)
    if 'max_shift' in kwargs:
        kwargs['max_shift'] = .3 if timing_policy == 'legacy' else None
    adapter = UniformFourierAdapter(metric, 5., 1.85, nsamples=3,
        timing_policy=timing_policy, max_shift=.3 if timing_policy == 'legacy' else None)
    adapter.aperture_fraction = .13
    monkeypatch.setattr(module, 'trilegal_results', population, raising=False)
    calls = []

    def sampler(**options):
        u = cube(options['ndim'])
        theta = options['prior_transform_batched'](u)
        np.testing.assert_allclose(theta, [options['prior_transform'](v) for v in u], atol=2e-11)
        scalar = np.array([options['log_likelihood_function'](v) for v in theta])
        batched = options['log_likelihood_function_batched'](theta)
        np.testing.assert_array_equal(np.isfinite(scalar), np.isfinite(batched))
        good = np.isfinite(scalar)
        np.testing.assert_allclose(scalar[good], batched[good], atol=2e-5, rtol=2e-10)
        calls.append((scalar, batched))
        weights = np.exp(np.where(good, scalar-np.max(scalar[good]), -np.inf)) if good.any() else np.ones(len(scalar))
        weights /= weights.sum()
        return (float(np.max(scalar[good])) if good.any() else -np.inf), weights, theta, scalar

    monkeypatch.setattr('pentaceratops.experimental.uniform_fourier.persistent_sampling', sampler)
    original = module._run_persistent_evidence
    adapter.call(function, **kwargs)
    assert module._run_persistent_evidence is original
    assert len(calls) == 1


def test_public_driver_uses_one_null_and_records_pools(tmp_path):
    from pentaceratops import calc_probs_fourier, RunResult
    metric = metric_data(n=65)
    e, o = metric.blocks
    target = SimpleNamespace(mission='TESS', stars=pd.DataFrame([
        dict(ID=1, mass=.8, rad=.75, Teff=4800., plx=10., Tmag=10., Jmag=9.,
             Hmag=8.5, Kmag=8., tdepth=.003, fluxratio=.9)]))
    skipped = ('PTP','PEB','PEBx2P','STP','SEB','SEBx2P','DTP','DEB','DEBx2P','BTP','BEB','BEBx2P')
    # Reinsert the unused real Nyquist variance for the even-length convention.
    destination = tmp_path/'consistent.npz'
    result = calc_probs_fourier(target, 5., (3*e.flux+2*o.flux)/5,
        e.variance*3, 1.85, 5/65, 5, 5,
        flux_even_odd_full=np.r_[e.flux,o.flux], ntransits_even=3, ntransits_odd=2,
        psd_even=e.variance*3, psd_odd=o.variance*2, N=12, steps=2, nsamples=3,
        drop_scenario=skipped, verbose=0, seed=194, output_path=destination)
    assert result.output.attrs['weighting'] == 'consistent'
    assert result.output.attrs['backend'] == 'optimized'
    assert result.output['null'].nunique() == 1
    np.testing.assert_allclose(result.output['null'], metric.null_loglike)
    assert sum(r['kind'] == 'sampler' for r in result.sampling) == 3
    saved = RunResult.load(destination)
    assert saved.metadata['weighting'] == 'consistent'
    assert saved.output.attrs['noise_model']['representation'] == 'full_period_parities'
    assert np.isfinite(saved.output.attrs['FPP'])
