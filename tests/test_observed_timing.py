"""Automatic x2P window support, preserved ordinary EBs, complete Fourier models."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from pentaceratops import calc_probs_fourier, RunResult
from pentaceratops.experimental.window_support import contact_overlap, alternating_overlap
from pentaceratops.hz import real_adapter
from pentaceratops.likelihoods import real as lk, fourier as fft
from pentaceratops.likelihoods.observed_fourier import (
    binary_flux, lnL_EB_second_observed, lnL_EB_evenodd_observed,
)
from pentaceratops.experimental.joint_fourier import JointFourierAdapter
from pentaceratops.likelihoods.joint_fourier import GaussianBlock
from test_exposure import synthetic_data, physical_columns, scalar_scores, assert_agreement
from test_primary_v2 import primary_data


def columns_at_offset(kind, offset):
    p = {key: value[:1].copy() for key, value in physical_columns(kind).items()}
    p.update(inc=np.array([89.9]), argp=np.array([90.]), R_EB=np.array([.3]),
             companion_fluxratio=np.array([.2]), companion_is_host=np.array([False]))
    period = p["P_orb"][0]
    p["ecc"] = np.array([brentq(lambda e: (lk.mean_anomaly_difference(e, np.pi/2)-.5)*period-offset, 0., .9)])
    return p


def parameters(columns):
    return {key: value[0] for key, value in columns.items() if key != "R_p"}


def test_contact_overlap_full_width_formula_exposure_gaps_and_wrap():
    # W=.4, D=.1: x2P offset limit W+D=.5 without exposure integration.
    center = np.array([.249, .251, 2.249])
    np.testing.assert_array_equal(contact_overlap(np.array([-.2,.2]), center, 2., -.05, .05, 0.),
                                  [True, False, True])
    # An exposure crossing ingress is informative even when its center misses.
    assert contact_overlap(np.array([-.2,.2]), .251, 2., -.05, .05, .01)
    # A gap is not a non-detection: keep a model inside the window envelope.
    assert contact_overlap(np.array([-.2,.2]), 0., 2., -.01, .01, .01)
    assert not contact_overlap(np.empty(0), 0., 2., -.01, .01, .01)
    # Asymmetric windows must not be replaced with max(abs(time)).
    assert not contact_overlap(np.array([.3,.5]), -.4, 2., -.02, .02, .01)
    assert contact_overlap(np.array([.3,.5]), .48, 2., -.02, .04, .01)


@pytest.mark.parametrize("primary_only", [False, True])
@pytest.mark.parametrize("parity", ["profile", "marginalize"])
def test_alternating_edge_overlap_is_not_rejected_by_center(primary_only, parity):
    data = primary_data()[0] if primary_only else synthetic_data()
    observed = real_adapter(data, mission="TESS", primary_only=primary_only, parity=parity)
    legacy = real_adapter(data, mission="TESS", primary_only=primary_only, parity=parity,
                          timing_policy="legacy")
    columns = columns_at_offset("x2p", .48)  # Centers +/- .24 lie beyond +/- .22.
    assert alternating_overlap(data["even_time"], data["odd_time"], columns, data["exptime"])[0]
    assert np.isneginf(scalar_scores(legacy, "x2p", columns)[0])
    scalar = scalar_scores(observed, "x2p", columns)
    assert np.isfinite(scalar[0])
    assert_agreement(scalar, observed.photometric_columns("x2p", columns))
    far = columns_at_offset("x2p", 1.0)
    assert not alternating_overlap(data["even_time"], data["odd_time"], far, data["exptime"])[0]
    assert np.isneginf(scalar_scores(observed, "x2p", far)[0])
    assert np.isneginf(observed.photometric_columns("x2p", far)[0])


@pytest.mark.parametrize("primary_only", [False, True])
@pytest.mark.parametrize("offset", [0., .24, .7])
def test_ordinary_binary_keeps_existing_secondary_treatment(primary_only, offset):
    data = primary_data()[0] if primary_only else synthetic_data()
    adapter = real_adapter(data, mission="TESS", primary_only=primary_only)
    legacy = real_adapter(data, mission="TESS", primary_only=primary_only,
                          timing_policy="legacy")
    columns = columns_at_offset("binary", offset)
    adapter.record = True
    scalar = scalar_scores(adapter, "binary", columns)
    snapshot = adapter.snapshot
    legacy.record = True
    np.testing.assert_array_equal(scalar, scalar_scores(legacy, "binary", columns))
    np.testing.assert_array_equal(snapshot["model"], legacy.snapshot["model"])
    assert snapshot["phantom_loglike_ratio"] == legacy.snapshot["phantom_loglike_ratio"]
    assert np.isfinite(scalar[0])
    if primary_only:
        assert snapshot["secondary_omitted"]
        assert snapshot["phantom_loglike_ratio"] == 0.
    elif offset > data["secondary_time"].max():
        assert snapshot["secondary_outside"]
        np.testing.assert_array_equal(snapshot["model"][adapter.slices["secondary"]], 1.)
        # Retain a continuous Gaussian penalty, with no 1.5-sigma rejection.
        assert snapshot["phantom_loglike_ratio"] < 0.
        hypothetical = adapter.aperture(adapter.centered_secondary(
            data["secondary_time"], **snapshot["parameters"]))
        assert snapshot["phantom_loglike_ratio"] == pytest.approx(
            adapter.secondary_metric.gain(hypothetical))
    else:
        assert not snapshot["secondary_outside"]
        assert snapshot["phantom_loglike_ratio"] == 0.
    adapter.record = False
    assert_agreement(scalar, adapter.photometric_columns("binary", columns))
    legacy.record = False
    np.testing.assert_array_equal(adapter.photometric_columns("binary", columns),
                                  legacy.photometric_columns("binary", columns))


@pytest.mark.parametrize("kind", ["binary", "x2p"])
@pytest.mark.parametrize("ecc", [0., .45, .8])
def test_full_orbit_fourier_matches_independent_native_renderer(kind, ecc):
    p = parameters(physical_columns(kind))
    p.update(inc=89.9, ecc=ecc, argp=90., R_EB=.3, exptime=.005, nsamples=7)
    period = p["P_orb"]
    first = np.linspace(-period/4, period/4, 60, endpoint=False)
    second = first.copy()
    native = np.r_[first, second+period/2]
    block = GaussianBlock.from_covariance(native, np.ones(len(native)), np.eye(len(native)), p["exptime"])
    reference = JointFourierAdapter([block], period/(2 if kind == "x2p" else 1),
                                    0., mission="TESS", nsamples=7)
    def transform(x):
        return np.fft.rfft(x)[1:-1]
    d1, d2 = transform(1+.003*np.sin(first*13)), transform(1+.003*np.cos(second*15))
    v1, v2 = np.full(len(d1), .01), np.full(len(d2), .02)
    expected = []
    for reverse in ([False, True] if kind == "x2p" else [False]):
        model = reference.native_model(kind, p, reverse=reverse)
        rendered = binary_flux(native+(period/2 if reverse else 0.), p, alternating=kind == "x2p")
        np.testing.assert_allclose(rendered, model, atol=2e-12, rtol=0)
        expected.append(np.sum(abs(d1-transform(model[:60]))**2/v1)
                        + np.sum(abs(d2-transform(model[60:]))**2/v2))
    function = lnL_EB_evenodd_observed if kind == "x2p" else lnL_EB_second_observed
    actual = function(first,d1,v1,second,d2,v2,**p)
    assert actual == pytest.approx(min(expected), abs=2e-8, rel=2e-12)
    if kind == "binary" and ecc == .8:
        # The displaced secondary is outside its nominal half; retain its flux
        # in the primary half, unlike simply removing the old max_shift gate.
        old = fft.lnL_EB_second_fourier(first,d1,v1,second,d2,v2,**p)
        assert abs(old-actual) > .01
    if ecc == 0.:
        old = (fft.lnL_EB_second_fourier if kind == "binary" else fft.lnL_EB_evenodd_fourier)
        assert actual == pytest.approx(old(first,d1,v1,second,d2,v2,**p), abs=2e-9)


def test_recorded_fourier_selects_and_restores_policy(monkeypatch, tmp_path):
    from pentaceratops import fourier as driver
    from pentaceratops.evidence import fourier_eclipses as evidence
    original = evidence.lnL_EB_second_fourier
    observed_calls = []
    def calculate(*args, max_anomaly_shift=None, **kwargs):
        observed_calls.append(evidence.lnL_EB_second_fourier is lnL_EB_second_observed)
        return pd.DataFrame([dict(ID=1, scenario="TP", lnBF=0., prob=1.)])
    monkeypatch.setattr(driver, "calc_probs_fourier", calculate)
    path = tmp_path/"observed.npz"
    calc_probs_fourier(seed=2, output_path=path)
    saved = RunResult.load(path)
    assert saved.metadata["timing_policy"] == saved.output.attrs["timing_policy"] == "observed"
    assert evidence.lnL_EB_second_fourier is original
    with pytest.raises(ValueError, match="legacy"):
        calc_probs_fourier(max_anomaly_shift=.3)
    legacy = calc_probs_fourier(timing_policy="legacy", max_anomaly_shift=.3)
    assert legacy.metadata["max_anomaly_shift"] == .3
    assert observed_calls == [True, False]
    def broken(**kwargs):
        assert evidence.lnL_EB_second_fourier is lnL_EB_second_observed
        raise RuntimeError("sampling failed")
    monkeypatch.setattr(driver, "calc_probs_fourier", broken)
    with pytest.raises(RuntimeError, match="sampling failed"):
        calc_probs_fourier()
    assert evidence.lnL_EB_second_fourier is original


@pytest.mark.parametrize("name", ["lnZ_TEB_secondary_fourier", "lnZ_TEB_evenodd_fourier"])
def test_observed_fourier_actual_sampling_keeps_complete_pools(name, monkeypatch, tmp_path):
    from pentaceratops import run_evidence
    from pentaceratops.evidence import fourier_eclipses as evidence
    from pentaceratops.likelihoods.observed_fourier import observed_fourier_engine
    from test_all_scenarios import kwargs_for
    function = getattr(evidence, name)
    kwargs = kwargs_for(function, "default", tmp_path)
    kwargs.update(N=6, steps=1, max_shift=None)
    monkeypatch.setattr(evidence, "POSTERIOR_NSAMPLES", 9)
    with observed_fourier_engine():
        result = run_evidence(function, seed=309, **kwargs)
    assert np.isfinite(result.output["lnZ"])
    assert len(result.output["M_s"]) == 9
    pools = [p for p in result.sampling if p["kind"] == "sampler"]
    assert len(pools) == 1 and len(pools[0]["weights"]) > 0
    assert pools[0]["weights"].sum() == pytest.approx(1.)
