"""Independent covariance, masking, normalization and full-orbit model checks."""
import importlib
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from pentaceratops.likelihoods.joint_fourier import (
    GaussianBlock, JointFourierMetric, observed_covariance, total_fft_power,
)
from pentaceratops.experimental.joint_fourier import JointFourierAdapter


@pytest.mark.parametrize("n", [17, 18, 31, 32])
def test_covariance_matches_explicit_fourier_basis(n):
    power = 1.+np.arange(n//2+1)**.5
    ids = np.array([0, 2, 3, n//2, n-1])
    # Independent real Fourier design, including both special real modes.
    k = np.arange(1, (n+1)//2)
    phase = 2*np.pi*np.outer(ids, k)/n
    basis = np.column_stack([np.ones(len(ids))/n, np.cos(phase)*2/n, -np.sin(phase)*2/n])
    variance = np.r_[power[0], power[k]/2, power[k]/2]
    if n%2 == 0:
        basis = np.column_stack([basis, (-1.)**ids/n])
        variance = np.r_[variance, power[-1]]
    expected = (basis*variance) @ basis.T
    np.testing.assert_allclose(observed_covariance(power, n, ids), expected, atol=1e-16)


@pytest.mark.parametrize("n", [17, 18])
def test_noise_simulations_validate_analytic_covariance(n):
    rng = np.random.default_rng(720+n)
    power = n*(.01+1/(1+np.arange(n//2+1)**2))
    ids = np.array([0, 2, 5, n-2])
    white = rng.normal(size=(40000, n))
    draws = np.fft.irfft(np.fft.rfft(white, axis=1)*np.sqrt(power/n), n=n, axis=1)[:, ids]
    expected = observed_covariance(power, n, ids)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), expected,
                               atol=.015*np.max(np.diag(expected)), rtol=.04)


def test_white_noise_added_once_and_explicit_dc():
    n=20; variance=.04
    power=total_fft_power(np.zeros(n//2), variance, n, .5, dc_power=n*variance*.25)
    np.testing.assert_allclose(observed_covariance(power,n,np.array([0,3,8,15])),
                               np.eye(4)*variance*.25,atol=1e-17)


def test_observed_gaussian_is_marginal_not_conditioned_on_gap_values():
    n=24; t=np.arange(n)*.1; observed=np.arange(n)%3 != 0
    power=n/(1+np.arange(n//2+1))
    flux=1+.1*np.sin(t); altered=flux.copy(); altered[~observed]=1e12
    a=GaussianBlock.from_spectrum(t,flux,power,.1,observed=observed)
    b=GaussianBlock.from_spectrum(t,altered,power,.1,observed=observed)
    metric=JointFourierMetric([a]); model=1+.03*np.cos(t[observed])
    covariance=observed_covariance(power,n,np.flatnonzero(observed))
    expected=multivariate_normal.logpdf(flux[observed],mean=model,cov=covariance)
    assert metric.loglike(model)==pytest.approx(expected,abs=1e-11)
    assert JointFourierMetric([b]).loglike(model)==metric.loglike(model)
    np.testing.assert_array_equal(a.time,t[observed])
    # Selecting a full-grid precision would CONDITION on the missing samples.
    whole=observed_covariance(power,n,np.arange(n))
    assert np.max(abs(a.precision-np.linalg.inv(whole)[np.ix_(observed,observed)]))>.01


@pytest.mark.parametrize("n", [31,32])
def test_complete_grid_matches_existing_fft_cost_for_same_modes(n):
    from pentaceratops.likelihoods.fourier import _rfft_complex_bins, _fourier_chi2
    t=np.arange(n)/n
    residual=.003*np.sin(2*np.pi*t)+.004*np.cos(4*np.pi*t)
    signal=.001*np.sin(2*np.pi*t)-.002*np.cos(6*np.pi*t)
    power=np.linspace(.001,.01,n//2+1)
    metric=JointFourierMetric([GaussianBlock.from_spectrum(t,1+residual,power,1/n)])
    var=power[1:] if n%2 else power[1:-1]
    data_ft=_rfft_complex_bins(1+residual)
    expected=(_fourier_chi2(data_ft,np.zeros_like(data_ft),var)
              -_fourier_chi2(data_ft,_rfft_complex_bins(1+signal),var))
    assert metric.gain(1+signal)==pytest.approx(expected,abs=1e-12)


def test_multiple_blocks_and_sparse_batch_equal_direct_gaussian():
    rng=np.random.default_rng(6);blocks=[];expected=0.;models=[]
    for n in (15,18):
        x=rng.normal(size=(n,n));cov=x@x.T+np.eye(n)*.3
        flux=1+rng.normal(size=n);model=np.ones(n);model[::4]-=.2
        block=GaussianBlock.from_covariance(np.arange(n),flux,cov,.1)
        blocks.append(block);models.extend(model)
        expected+=multivariate_normal.logpdf(flux,mean=model,cov=cov)
    metric=JointFourierMetric(blocks);model=np.array(models)
    assert metric.loglike(model)==pytest.approx(expected,abs=1e-10)
    scores=metric.gains(np.stack([model,np.ones(len(model))]))
    np.testing.assert_allclose(scores,[metric.gain(model),0],atol=1e-12)
    assert len(metric.gains(np.empty((0,len(model)))))==0


@pytest.mark.parametrize("bad", [np.array([0,0]),np.array([-1,2]),np.array([0,8]),np.array([.1,2.])])
def test_bad_observation_indices_fail(bad):
    with pytest.raises(ValueError):observed_covariance(np.ones(5),8,bad)


def test_invalid_mask_and_irregular_grid_fail():
    with pytest.raises(ValueError):GaussianBlock.from_spectrum([0,1,3],[1,1,1],[1,1],1)
    with pytest.raises(ValueError):GaussianBlock.from_spectrum([0,1,2],[1,np.nan,1],[1,1],1,observed=np.ones(3,bool))
    with pytest.raises(ValueError):GaussianBlock.from_spectrum([0,1,2],[1,1,1],[1,1],1,observed=np.ones(3,int))


def fixture(kind="binary", parity="profile", mission="TESS"):
    n=400;t=np.linspace(-.5,7.5,n);mask=(np.arange(n)%7!=0)&~((t>1.7)&(t<2.3))
    flux=1+.0002*np.sin(t)
    block=GaussianBlock.from_spectrum(t,flux,np.full(n//2+1,n*.002**2),t[1]-t[0],observed=mask)
    adapter=JointFourierAdapter([block],4.,0.,parity=parity,nsamples=7,chunk_size=3,mission=mission)
    count=6
    p=dict(P_orb=np.full(count,8. if kind=="x2p" else 4.),inc=np.full(count,89.8),
        ecc=np.array([0.,.03,.1,.3,.02,.2]),argp=np.array([90.,110.,250.,80.,180.,270.]),
        a=np.full(count,8e11),R_s=np.ones(count),u1=np.full(count,.3),u2=np.full(count,.2),
        R_p=np.array([2.,5.,10.,15.,1.,3.]),R_EB=np.array([.2,.5,1.,.99999999,1.2,.1]),
        EB_fluxratio=np.full(count,.2),companion_fluxratio=np.full(count,.15),
        companion_is_host=np.array([False,True,False,True,False,True]))
    return adapter,p


@pytest.mark.parametrize("kind",["planet","binary","x2p"])
@pytest.mark.parametrize("parity",["profile","marginalize"])
def test_sparse_optimized_model_and_scores_match_dense_pytransit(kind,parity):
    adapter,p=fixture(kind,parity)
    for aperture in (1.,.07):
        adapter.aperture_fraction=aperture
        dense=np.array([adapter.native_model(kind,{k:v[i] for k,v in p.items()}) for i in range(6)])
        sparse=adapter._project(kind,p)
        np.testing.assert_allclose(sparse,dense,atol=3e-9,rtol=1e-12)
        values=[]
        for i in range(6):
            q={k:v[i] for k,v in p.items()}
            q.pop("R_p" if kind!="planet" else "R_EB")
            if kind=="planet":q.pop("EB_fluxratio")
            q.update(exptime=adapter.exptime,nsamples=7)
            values.append(-adapter._scalar(kind,q))
        np.testing.assert_allclose(adapter.photometric_columns(kind,p),values,atol=2e-5,rtol=2e-12)


def test_missing_secondary_has_no_phantom_penalty():
    t=np.linspace(-.15,.15,51);block=GaussianBlock.from_covariance(t,np.ones(51),np.eye(51)*.002**2,.001)
    adapter=JointFourierAdapter([block],4.,0.,nsamples=7)
    _,columns=fixture();p={k:v[0] for k,v in columns.items()};p.pop("R_p")
    p.update(exptime=.001,nsamples=7)
    adapter.record=True;adapter._scalar("binary",p)
    assert adapter.snapshot["phantom_loglike_ratio"]==0
    # Deep predicted secondary at P/2 is absent from every observed time.
    assert len(adapter.snapshot["model"])==len(t)
    assert adapter.snapshot["photometric_loglike_ratio"]==pytest.approx(adapter.metric.gain(adapter.native_model("binary",p)))


def test_alternating_eclipses_preserved_without_splitting():
    adapter,columns=fixture("x2p");p={k:v[0] for k,v in columns.items()}
    a=adapter.native_model("x2p",p);b=adapter.native_model("x2p",p,reverse=True)
    assert np.max(abs(a-b))>1e-3
    assert adapter.metric.gain(a)!=adapter.metric.gain(b)
    changed=dict(p,P_orb=16.)
    with pytest.raises(ValueError,match="exactly 2P"):adapter.native_model("x2p",changed)


@pytest.mark.parametrize("mission", ["TESS", "Kepler"])
def test_real_package_scenario_callbacks_and_batched_priors(monkeypatch, mission):
    from pentaceratops.experimental.covariance import patched_engine,scenario_functions
    from pentaceratops.experimental.batched_priors import make_prior
    from pentaceratops.evidence import real,eclipses
    adapter,_=fixture(mission=mission)
    star=dict(mass=1.,rad=1.,Teff=5700.,plx=10.,Tmag=10.,Jmag=9.,Hmag=9.,Kmag=9.)
    class Capture(BaseException):pass
    from test_all_scenarios import population
    monkeypatch.setattr(real, "trilegal_results", population)
    monkeypatch.setattr(eclipses, "trilegal_results", population)
    for name in scenario_functions():
        fn=scenario_functions()[name];module=importlib.import_module(fn.__module__);saved={}
        def intercept(loglike,prior_transform,ndim,**kwargs):
            saved.update(like=loglike,prior=prior_transform,ndim=ndim);raise Capture()
        with monkeypatch.context() as patch:
            patch.setattr(module,"_run_persistent_evidence",intercept)
            with patched_engine(adapter):
                with pytest.raises(Capture):fn(**adapter.scenario_kwargs(fn,star,trilegal="unused",N=8,steps=2))
        rng=np.random.default_rng(7);u=rng.uniform(.05,.9,(16,saved["ndim"]));u[:,1]=.999999;u[:,2]=.02
        physical=np.array([saved["prior"](x) for x in u]);vector=make_prior(saved["prior"])(u)
        np.testing.assert_allclose(physical,vector,atol=2e-12)
        with patched_engine(adapter):
            scalar=np.array([saved["like"](x) for x in physical]);fast=adapter.likelihood_batch(saved["like"],physical)
        assert np.isfinite(scalar).any()
        np.testing.assert_array_equal(np.isfinite(scalar),np.isfinite(fast))
        np.testing.assert_allclose(fast,scalar,atol=2e-5,rtol=2e-12)


@pytest.mark.parametrize("scenario", ["TP", "EB", "EBx2P"])
def test_real_sampler_retains_pool_and_replays_bestfit(scenario):
    from pentaceratops.evidence.joint_fourier import sample_joint_scenario
    from pentaceratops.evidence import real, eclipses
    # Weak data make this a quick integration check, not a convergence claim.
    t = np.linspace(-.3, 7.7, 120)
    block = GaussianBlock.from_covariance(t, np.ones(len(t)), np.eye(len(t))*.1**2, .003)
    adapter = JointFourierAdapter([block], 4., 0.)
    star = dict(mass=1., rad=1., Teff=5700., plx=10., Tmag=10., Jmag=9., Hmag=9., Kmag=9.)
    original = (real.POSTERIOR_NSAMPLES, eclipses.POSTERIOR_NSAMPLES,
                real._run_persistent_evidence, eclipses._run_persistent_evidence, real.lnL_TP)
    state = np.random.get_state()
    result = sample_joint_scenario(adapter, scenario, star, trilegal="unused", N=8, steps=2,
                                   seed=307, posterior_samples=19)
    assert np.isfinite(result["lnBF"])
    assert result["lnZ"] == pytest.approx(result["lnBF"]+adapter.metric.null_loglike)
    assert result["posterior_parameters"].shape[0] == 19
    ids = result["posterior_source_index"]
    np.testing.assert_array_equal(result["posterior_parameters"], result["physical_parameters"][ids])
    assert result["bestfit"]["scalar_replay"] == pytest.approx(result["bestfit"]["log_target"], abs=2e-5)
    assert result["sampling_stats"]["likelihood_batches"] > 0
    assert result["sampling_stats"]["prior_batches"] > 0
    restored = (real.POSTERIOR_NSAMPLES, eclipses.POSTERIOR_NSAMPLES,
                real._run_persistent_evidence, eclipses._run_persistent_evidence, real.lnL_TP)
    assert original == restored
    current = np.random.get_state()
    assert state[0] == current[0] and state[2:] == current[2:]
    np.testing.assert_array_equal(state[1], current[1])


def test_sector_preparation_preserves_observations_and_serialization(tmp_path):
    from pentaceratops.preprocessing.joint_fourier import prepare_sector, save_block, load_block
    rng = np.random.default_rng(72)
    t = np.arange(256)*.05
    flux = 3.*(1+.005*np.sin(t)+rng.normal(0,.002,len(t)))
    keep = np.arange(len(t))%6 != 0
    block, diagnostic, spectrum = prepare_sector(t[keep], flux[keep], np.full(keep.sum(),.006),
        period=30., epoch=5., duration=.03, cadence=.05, iterations=1)
    np.testing.assert_allclose(block.time, t[keep], atol=1e-12)
    np.testing.assert_array_equal(block.flux, flux[keep]/diagnostic["center"])
    assert diagnostic["n_observed"] == keep.sum()
    assert diagnostic["interpolated_likelihood_samples"] == 0
    assert diagnostic["white_noise_count"] == 1
    assert spectrum["total_fft_power"][0] == spectrum["total_fft_power"][1]
    directory = tmp_path/"block"
    save_block(block, directory)
    restored = load_block(directory)
    assert isinstance(restored.precision, np.memmap) and not restored.precision.flags.writeable
    metric = JointFourierMetric([block]); loaded = JointFourierMetric([restored])
    model = 1-.01*np.exp(-.5*((block.time-5)/.08)**2)
    assert loaded.loglike(model) == metric.loglike(model)


@pytest.mark.parametrize('n', [31, 32, 57])
def test_chunked_spectrum_save_matches_dense_gaussian_with_gaps(tmp_path, n):
    from pentaceratops.preprocessing.joint_fourier import save_spectrum_block
    rng = np.random.default_rng(n)
    t = np.arange(n)*.1
    observed = np.ones(n, dtype=bool)
    observed[4:12] = False
    observed[::9] = False
    flux = 1+.01*rng.normal(size=n)
    flux[~observed] = np.nan
    power = n*(.001+1/(1+np.arange(n//2+1))**3)
    expected = GaussianBlock.from_spectrum(t, flux, power, .1, observed=observed)
    actual = save_spectrum_block(t, flux, power, .1, tmp_path/'block',
                                 observed=observed, chunk_rows=5)
    assert actual.precision.flags.c_contiguous
    for key in ('precision', 'projected_residual', 'sigma', 'time', 'flux'):
        np.testing.assert_allclose(getattr(actual, key), getattr(expected, key),
                                   atol=3e-10, rtol=2e-12)
    assert actual.null_loglike == pytest.approx(expected.null_loglike, abs=2e-11)
    model = 1+.005*rng.normal(size=observed.sum())
    assert JointFourierMetric([actual]).loglike(model) == pytest.approx(
        JointFourierMetric([expected]).loglike(model), abs=2e-11)
    np.testing.assert_allclose(JointFourierMetric([actual]).gains(np.array([model])),
        [JointFourierMetric([expected]).gain(model)], atol=2e-11)


def test_original_fourier_timing_constraint_survives_metric_change():
    from types import SimpleNamespace
    from pentaceratops.experimental.fourier_baseline import FourierBaselineObservation
    from pentaceratops.evidence.scenario import ScenarioLikelihood
    from pentaceratops.likelihoods.fourier import lnL_EB_second_fourier, lnL_EB_evenodd_fourier
    from pentaceratops.likelihoods import real as lk
    adapter=SimpleNamespace(exptime=.1,nsamples=7,lk=lk,record=False,period=5.,epoch=.2,
                            metric=SimpleNamespace(time=np.array([0.,1.,6.])))
    for kind,cost in [('binary',lnL_EB_second_fourier),('x2p',lnL_EB_evenodd_fourier)]:
        original=ScenarioLikelihood(model=SimpleNamespace(kind=kind),time=np.array([-1.,1.]),
            data=np.zeros(1),noise=np.ones(1),normalization=4.,residual_cost=cost,
            exptime=.1,nsamples=7,domain='fourier',
            secondary=(np.array([-1.,1.]),np.zeros(1),np.ones(1)),cost_options={'max_shift':.03})
        bridge=FourierBaselineObservation(original,adapter);bridge.reference=True
        assert bridge.timing_allowed(dict(ecc=0.,argp=90.,P_orb=10.))
        assert bridge.cost(ecc=.5,argp=45.,P_orb=10.,max_shift=.03)==np.inf
        assert bridge.original is original
        assert bridge.observation.model is original.model
        assert bridge.observation.cost_options is original.cost_options
        assert bridge.observation.normalization==0.
