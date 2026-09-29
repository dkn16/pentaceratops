"""Small synthetic fixed-period checks; no catalogs or production sampling."""

import numpy as np
import pytest

from pentaceratops.experimental.covariance import (
    ModelAdapter, patched_engine, scenario_functions, scenario_kwargs,
)
from pentaceratops.experimental.v2_adapter import OptimizedAdapter


def synthetic_data(exptime=10/1440):
    rng = np.random.default_rng(81)
    grids = dict(even=np.linspace(-.22, .22, 29), odd=np.linspace(-.22, .22, 27),
                 secondary=np.linspace(-.22, .22, 25))
    data = {}
    for panel, time in grids.items():
        data[panel+"_time"] = time
        data[panel+"_mean_flux"] = (
            1-.002*np.exp(-.5*(time/.035)**2) + rng.normal(0, .0001, len(time)))
        data[panel+"_sigma"] = np.full(len(time), .002)
    data["flux"] = np.concatenate([data[p+"_mean_flux"] for p in grids])
    data["map_flux"] = data["flux"].copy()
    data["sigma"] = np.concatenate([data[p+"_sigma"] for p in grids])
    data["factor"] = rng.normal(0, .0001, (len(data["flux"]), 5))
    data.update(period=2., exptime=exptime)
    return data


def physical_columns(kind):
    n = 6
    return dict(
        P_orb=np.full(n, 4. if kind == "x2p" else 2.),
        inc=np.array([89, 89.8, 89, 89.5, 89.5, 89.9]),
        ecc=np.array([0., .6, .02, .1, .1, .4]),
        argp=np.array([90, 90, 65, 240, 310, 180]),
        a=np.full(n, 8e11), R_s=np.ones(n), u1=np.full(n, .3), u2=np.full(n, .2),
        R_p=np.array([1, 2.5, 5, 10, 19, 2.]),
        R_EB=np.array([.1, .5, 1., .99999999, 1.1, .2]),
        EB_fluxratio=np.full(n, .25), companion_fluxratio=np.full(n, .2),
        companion_is_host=np.array([False, True, False, True, False, True]),
    )


def scalar_scores(adapter, kind, columns, **overrides):
    dummy = (np.zeros(1), np.ones(1), .001) * (1 if kind == "planet" else 2)
    omitted = ("R_EB", "EB_fluxratio") if kind == "planet" else ("R_p",)
    values = []
    for i in range(len(columns["P_orb"])):
        parameters = {key: value[i] for key, value in columns.items() if key not in omitted}
        values.append(-getattr(adapter, kind)(
            *dummy, **parameters, exptime=adapter.data["exptime"], **overrides))
    return np.array(values)


def assert_agreement(reference, answer):
    finite = np.isfinite(reference)
    assert finite.any()
    np.testing.assert_array_equal(np.isfinite(answer), finite)
    np.testing.assert_allclose(answer[finite], reference[finite], atol=2e-5, rtol=2e-12)


@pytest.mark.parametrize("nsamples", [1, 5, 10, 20, 50])
@pytest.mark.parametrize("kind", ["planet", "binary", "x2p"])
@pytest.mark.parametrize("exptime", [2/1440, 30/1440])
def test_scalar_fast_agree_across_exposure_resolutions(nsamples, kind, exptime):
    data = synthetic_data(exptime)
    scalar = ModelAdapter(data, nsamples=nsamples)
    fast = OptimizedAdapter(data, nsamples=nsamples, chunk_size=3)
    columns = physical_columns(kind)
    for aperture in (1., .07):
        scalar.aperture_fraction = fast.aperture_fraction = aperture
        for parity in ("profile", "marginalize"):
            scalar.parity = fast.parity = parity
            reference = scalar_scores(scalar, kind, columns)
            answer = fast.photometric_columns(kind, columns)
            assert_agreement(reference, answer)
    if kind == "binary":
        offset = (fast.lk.mean_anomaly_difference(columns["ecc"],
                  np.deg2rad(columns["argp"]))-.5)*columns["P_orb"]
        assert np.any(abs(offset) > .22) and np.any(abs(offset) < .22)
    if kind == "x2p":
        assert np.isneginf(reference).any()  # Retain the existing timing guard.


@pytest.mark.parametrize("kind", ["planet", "binary", "x2p"])
def test_zero_gp_and_explicit_override(kind):
    data = synthetic_data()
    scalar = ModelAdapter(data, include_gp=False, nsamples=50)
    fast = OptimizedAdapter(data, include_gp=False, nsamples=50)
    columns = physical_columns(kind)
    assert_agreement(scalar_scores(scalar, kind, columns, nsamples=7),
                     fast.photometric_columns(kind, columns, nsamples=7))
    assert scalar.nsamples == fast.nsamples == 50  # No lasting mutation.


@pytest.mark.parametrize("kind", ["planet", "binary", "x2p"])
def test_omitted_count_is_exactly_twenty(kind):
    data = synthetic_data()
    scalar, fast = ModelAdapter(data), OptimizedAdapter(data)
    columns = physical_columns(kind)
    np.testing.assert_array_equal(scalar_scores(scalar, kind, columns),
                                  scalar_scores(scalar, kind, columns, nsamples=20))
    np.testing.assert_array_equal(fast.photometric_columns(kind, columns),
                                  fast.photometric_columns(kind, columns, nsamples=20))


@pytest.mark.parametrize("adapter", [ModelAdapter, OptimizedAdapter])
@pytest.mark.parametrize("bad", [0, -1, True, np.bool_(True), 2.0, 1.5, "5", None])
def test_bad_subsample_count_fails_before_model_setup(adapter, bad):
    with pytest.raises(ValueError, match="nsamples must be a positive integer"):
        adapter({}, nsamples=bad)


def test_numpy_integer_is_accepted():
    assert OptimizedAdapter(synthetic_data(), nsamples=np.int64(7)).nsamples == 7


@pytest.mark.parametrize("kind", ["planet", "binary", "x2p"])
def test_period_still_fixed(kind):
    fast = OptimizedAdapter(synthetic_data(), nsamples=7)
    columns = {key: value[:1].copy() for key, value in physical_columns(kind).items()}
    columns["P_orb"] *= 1.01
    with pytest.raises(ValueError, match="Expected fixed P or 2P"):
        fast.photometric_columns(kind, columns)


def test_subsample_count_actually_changes_exposure_average():
    fast = OptimizedAdapter(synthetic_data(30/1440))
    columns = physical_columns("planet")
    a = fast.undiluted(columns, "primary", np.zeros(6), planet=True, nsamples=1)
    b = fast.undiluted(columns, "primary", np.zeros(6), planet=True, nsamples=50)
    assert np.max(abs(a-b)) > 1e-8
    again = fast.undiluted(columns, "primary", np.zeros(6), planet=True, nsamples=1)
    np.testing.assert_array_equal(a, again)  # No stale integration settings.


def test_scenario_kwargs_inherit_or_override_count(monkeypatch):
    from pentaceratops.evidence import real, eclipses
    monkeypatch.setattr(real, "POSTERIOR_NSAMPLES", real.POSTERIOR_NSAMPLES)
    monkeypatch.setattr(eclipses, "POSTERIOR_NSAMPLES", eclipses.POSTERIOR_NSAMPLES)
    adapter = OptimizedAdapter(synthetic_data(), nsamples=7)
    star = dict(mass=.8, rad=.75, Teff=4800., plx=10., Tmag=10., Jmag=9., Hmag=9., Kmag=9.)
    for function in scenario_functions().values():
        args = (function, adapter, star, 2., "unused.csv", None, 8, 2, 100)
        assert scenario_kwargs(*args)["nsamples"] == 7
        assert scenario_kwargs(*args, nsamples=50)["nsamples"] == 50
        with pytest.raises(ValueError, match="nsamples"):
            scenario_kwargs(*args, nsamples=0)
    assert adapter.nsamples == 7


@pytest.mark.parametrize("nsamples", [1, 7, 50])
@pytest.mark.parametrize("scenario", ["TP", "EB", "EBx2P"])
def test_evidence_closure_override_reaches_fast_likelihood(monkeypatch, nsamples, scenario):
    """Exercise actual evidence closures without sampling or querying catalogs."""
    from pentaceratops.evidence import real, eclipses
    monkeypatch.setattr(real, "POSTERIOR_NSAMPLES", real.POSTERIOR_NSAMPLES)
    monkeypatch.setattr(eclipses, "POSTERIOR_NSAMPLES", eclipses.POSTERIOR_NSAMPLES)
    adapter = OptimizedAdapter(synthetic_data(), nsamples=20)
    function = scenario_functions()[scenario]
    module = real if scenario == "TP" else eclipses
    star = dict(mass=.8, rad=.75, Teff=4800., plx=10., Tmag=10., Jmag=9., Hmag=9., Kmag=9.)
    kwargs = scenario_kwargs(function, adapter, star, 2., "unused.csv", None, 8, 2, 100,
                             nsamples=nsamples)
    captured = {}

    class Captured(BaseException):
        pass

    def capture(loglike, prior_transform, ndim, **kwargs):
        captured.update(loglike=loglike, prior=prior_transform, ndim=ndim)
        raise Captured()

    monkeypatch.setattr(module, "_run_persistent_evidence", capture)
    with patched_engine(adapter):
        with pytest.raises(Captured):
            function(**kwargs)
        rng = np.random.default_rng(302)
        u = rng.uniform(.01, .99, (16, captured["ndim"]))
        u[:, 1] = rng.uniform(.99, .9999, len(u))
        u[:, 2] *= .3
        theta = np.array([captured["prior"](row) for row in u])
        expected = np.array([captured["loglike"](row) for row in theta])
        actual = adapter.likelihood_batch(captured["loglike"], theta)
    assert_agreement(expected, actual)
    assert adapter.nsamples == 20


def test_invalid_direct_override_and_recorded_scalar_setting():
    adapter = OptimizedAdapter(synthetic_data(), nsamples=7)
    p = physical_columns("planet")
    with pytest.raises(ValueError, match="nsamples"):
        adapter.photometric_columns("planet", p, nsamples=False)
    adapter.record = True
    scalar_scores(adapter, "planet", p, nsamples=11)
    assert adapter.snapshot["parameters"]["nsamples"] == 11
