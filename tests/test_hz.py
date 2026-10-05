"""Public folded workflows: covariance, host frames, sampling and saved results."""
from types import SimpleNamespace
import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag

from pentaceratops import (FoldedFourierData, prepare_folded_fourier,
                           calc_probs_folded_real, calc_probs_folded_fourier, RunResult)
from pentaceratops.hz import real_adapter
from pentaceratops.experimental.folded_joint import full_orbit_fold
from pentaceratops.experimental.covariance import scenario_functions, patched_engine
from pentaceratops.likelihoods.joint_fourier import GaussianBlock, observed_covariance
from test_all_scenarios import population
from test_exposure import synthetic_data


def target(mission="TESS"):
    star = dict(ID=123, mass=1., rad=1., Teff=5700., plx=10., Tmag=10.,
                Jmag=9., Hmag=9., Kmag=9., tdepth=.002, fluxratio=.8)
    neighbor = dict(star, ID=456, fluxratio=.2)
    return SimpleNamespace(stars=pd.DataFrame([star, neighbor]), mission=mission,
                           trilegal_fname="unused.csv")


def observations():
    blocks, spectra, covariances = [], [], []
    for offset, n in ((0., 90), (11., 71)):
        t = offset + np.arange(n)*.05
        mask = np.arange(n)%7 != 0
        mask[15:23] = False
        power = n*.1**2*(1+2/(1+np.arange(n//2+1))**2)
        cov = observed_covariance(power, n, np.flatnonzero(mask))
        flux = 1+.0001*np.sin(t[mask])
        blocks.append(GaussianBlock.from_covariance(t[mask], flux, cov, .02))
        spectra.append(dict(time=t, observed=mask, total_fft_power=power))
        covariances.append(cov)
    return blocks, spectra, covariances


def test_folded_preparation_matches_dense_operator_and_saved_map(tmp_path):
    blocks, spectra, covariances = observations()
    folded = prepare_folded_fourier(blocks, spectra=spectra, period=2., epoch=.13, bin_days=.12)
    a, centers, bins, weights = full_orbit_fold(folded.native_time, 2., .13, .12)
    expected = np.asarray(a@block_diag(*covariances)@a.T)
    np.testing.assert_allclose(folded.block.precision@expected, np.eye(len(centers)), atol=3e-13)
    np.testing.assert_allclose(folded.block.flux, a@np.concatenate([b.flux for b in blocks]), atol=4e-16)
    without_spectra = prepare_folded_fourier(blocks, period=2., epoch=.13, bin_days=.12)
    np.testing.assert_allclose(without_spectra.block.precision, folded.block.precision, atol=1e-11)
    directory = tmp_path/"fold"
    folded.save(directory)
    loaded = FoldedFourierData.load(directory)
    assert isinstance(loaded.block.precision, np.memmap)
    np.testing.assert_array_equal(loaded.bin_index, bins)
    np.testing.assert_array_equal(loaded.weights, weights)
    np.testing.assert_array_equal(loaded.block.flux, folded.block.flux)
    assert loaded.adapter(mission="Kepler").mission == "Kepler"
    with pytest.raises(FileExistsError):
        folded.save(directory)
    spectra[0]["observed"] = ~spectra[0]["observed"]
    with pytest.raises(ValueError, match="ordering"):
        prepare_folded_fourier(blocks, spectra=spectra, period=2., epoch=.13, bin_days=.12)


@pytest.mark.parametrize("mission", ["TESS", "Kepler", "K2"])
@pytest.mark.parametrize("primary_only", [False, True])
def test_real_adapter_all_scenarios_preserve_mission_and_scalar_likelihood(monkeypatch, mission, primary_only):
    data = synthetic_data()
    if primary_only:
        n = len(data["even_time"])+len(data["odd_time"])
        for key in ("time", "mean_flux", "sigma"):
            data["secondary_"+key] = np.empty(0)
        for key in ("flux", "sigma", "map_flux", "factor"):
            data[key] = data[key][:n]
    adapter = real_adapter(data, mission=mission, primary_only=primary_only, nsamples=7)
    from pentaceratops.evidence import real, eclipses
    monkeypatch.setattr(real, "trilegal_results", population)
    monkeypatch.setattr(eclipses, "trilegal_results", population)
    class Capture(BaseException):
        pass
    for name, function in scenario_functions().items():
        module = importlib.import_module(function.__module__)
        kwargs = adapter.scenario_kwargs(function, target().stars.iloc[0], trilegal="unused")
        assert kwargs["mission"] == mission
        if "filt" in kwargs:
            assert kwargs["filt"] == ("TESS" if mission == "TESS" else "Kepler")
        saved = {}
        def capture(like, prior, ndim, **options):
            saved.update(like=like, prior=prior, ndim=ndim)
            raise Capture()
        with monkeypatch.context() as patch:
            patch.setattr(module, "_run_persistent_evidence", capture)
            with patched_engine(adapter), pytest.raises(Capture):
                function(**kwargs)
        rng = np.random.default_rng(902)
        cube = rng.uniform(.02, .9, (12, saved["ndim"]))
        cube[:, 1] = .99999999
        cube[:, 2] = .02
        physical = np.array([saved["prior"](u) for u in cube])
        for fraction in (1., .03):
            adapter.aperture_fraction = fraction
            with patched_engine(adapter):
                scalar = np.array([saved["like"](p) for p in physical])
                batch = adapter.likelihood_batch(saved["like"], physical)
            assert np.isfinite(scalar).any(), name
            np.testing.assert_allclose(batch, scalar, atol=2e-5, rtol=2e-12)


@pytest.mark.parametrize("domain", ["real", "fourier"])
def test_public_folded_sampler_and_pickle_free_result(domain, tmp_path):
    if domain == "real":
        data = synthetic_data()
        for panel in ("even", "odd", "secondary"):
            data[panel+"_sigma"][:] = .1
        data["sigma"][:] = .1
        function, inputs = calc_probs_folded_real, data
    else:
        blocks, spectra, _ = observations()
        function = calc_probs_folded_fourier
        inputs = prepare_folded_fourier(blocks, spectra=spectra, period=2., epoch=0., bin_days=.1)
    destination = tmp_path/(domain+".npz")
    result = function(target(), inputs, N=8, steps=2, nsamples=3, seed=307,
                      posterior_samples=11, eb_eta=.1, scenarios=("TP", "EB", "NTP"),
                      output_path=destination)
    saved = RunResult.load(destination)
    np.testing.assert_array_equal(saved.output.lnBF, result.output.lnBF)
    assert saved.output.attrs["scenario_scope"] == "conditional_subset"
    assert saved.output.attrs["flux_frame"] == "aperture"
    assert saved.output.attrs["NFPP"] == saved.output.loc[saved.output.scenario=="NTP", "prob"].sum()
    assert sum(r["kind"] == "sampler" for r in saved.sampling) == 3
    records = saved.metadata["scenario_records"]
    assert len(records) == 3
    assert "lnBF" in repr(saved.output)
    for record in records.values():
        assert record["posterior_parameters"].shape[0] == 11
        assert record["sampling_stats"]["likelihood_batches"] > 0
        assert record["bestfit"]["scalar_replay"] == pytest.approx(record["bestfit"]["log_target"], abs=2e-5)
        assert record["lnZ"] == pytest.approx(record["lnBF"]+record["null_loglike"])
    assert saved.output.loc[saved.output.scenario=="NTP", "seed"].item() == 307+1009*15
    with pytest.raises(FileExistsError):
        function(target(), inputs, output_path=destination)


def test_invalid_inputs_do_not_promote_neighbor_or_drop_secondary():
    data = synthetic_data()
    with pytest.raises(ValueError, match="secondary"):
        real_adapter(data, mission="TESS", primary_only=True)
    data["flux"] = data["flux"][::-1]
    with pytest.raises(ValueError, match="ordering"):
        real_adapter(data, mission="TESS")
    host = target()
    host.stars.loc[0, "tdepth"] = 0
    with pytest.raises(ValueError, match="promoted"):
        calc_probs_folded_real(host, synthetic_data(), scenarios=("TP",))
