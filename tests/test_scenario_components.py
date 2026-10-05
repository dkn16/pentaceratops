"""Four-corner host/system pilot, compared to the preserved numerical engine."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from pentaceratops.evidence.scenario import ScenarioLikelihood, ScenarioPrior
from pentaceratops.experimental.batched_priors import make_prior
from pentaceratops.experimental.physical_batch import PhysicalBatch
from pentaceratops.models.hosts import KnownHost, BoundCompanionHost
from pentaceratops.models.systems import Planet, Binary


def modules(scenario, domain, reference=False):
    planet = scenario in {"TP", "STP"}
    module = (
        ("real" if planet else "eclipses")
        if domain == "real"
        else ("fourier" if planet else "fourier_eclipses")
    )
    old = {
        "real": "marginal_likelihoods_new",
        "eclipses": "marginal_likelihoods_eb",
        "fourier": "marginal_likelihoods_fourier",
        "fourier_eclipses": "marginal_likelihoods_eb_fourier",
    }
    package = "triceratops." + old[module] if reference else "pentaceratops.evidence." + module
    stem = "TTP" if scenario == "TP" else "TEB" if scenario == "EB" else scenario
    function = (
        "lnZ_"
        + stem
        + ("" if planet else "_secondary")
        + ("_fourier" if domain == "fourier" else "")
    )
    mod = importlib.import_module(package)
    return mod, getattr(mod, function)


def inputs(scenario, domain, length=31):
    t = np.linspace(-0.2, 0.2, length)
    kw = dict(
        time=t,
        flux=1 - 0.003 * np.exp(-0.5 * (t / 0.04) ** 2),
        sigma=0.01,
        P_orb=5.0,
        M_s=0.8,
        R_s=0.75,
        Teff=4800.0,
        Z=0.0,
        N=8,
        steps=2,
        exptime=0.002,
        nsamples=3,
    )
    if scenario.startswith("S"):
        kw["plx"] = 10.0
    if scenario in {"EB", "SEB"}:
        kw.update(
            time_secondary=np.linspace(-0.25, 0.25, 29),
            flux_secondary=np.ones(29),
            sigma_secondary=0.015,
        )
    if domain == "fourier":
        kw["var_fourier"] = np.full(length // 2, length * 0.01**2)
        if scenario in {"EB", "SEB"}:
            kw["var_fourier_secondary"] = np.full(14, 29 * 0.015**2)
            kw["max_shift"] = 0.3
    return kw


def points(ndim):
    rng = np.random.default_rng(89)
    u = rng.uniform(0.01, 0.95, (24, ndim))
    u[:16, 1] = 0.99999
    u[:16, 2] = np.linspace(0.001, 0.15, 16)
    u[-2:, 4] = [0.0, 1.0]
    return u


def capture(monkeypatch, module, function, kwargs):
    result = {}

    class Captured(BaseException):
        pass

    def sample(likelihood, prior, **settings):
        result.update(likelihood=likelihood, prior=prior, settings=settings)
        raise Captured()

    with monkeypatch.context() as patch:
        patch.setattr(module, "_run_persistent_evidence", sample)
        with pytest.raises(Captured):
            function(**kwargs)
    return result


@pytest.fixture
def population_files(tmp_path):
    molusc = tmp_path / "companions.csv"
    molusc.write_text(
        "semi-major axis(AU),eccentricity,mass ratio\n5,0,0.8\n"
        "30,0.1,0.01\n40,0,0.4\n50,0.2,0.75\n60,0,0.99\n"
    )
    empty = tmp_path / "empty_pool.csv"
    empty.write_text("semi-major axis(AU),eccentricity,mass ratio\n5,0,0.5\n")
    contrast = tmp_path / "contrast.csv"
    contrast.write_text("0.1,1\n0.5,3\n2.0,7\n")
    return dict(molusc_file=str(molusc), contrast_curve_file=str(contrast), filt="J"), str(empty)


@pytest.mark.reference
@pytest.mark.parametrize("scenario", ["STP", "EB", "SEB"])
@pytest.mark.parametrize("domain", ["real", "fourier"])
@pytest.mark.parametrize("configuration", ["default", "pool", "empty_pool", "low_mass"])
def test_prior_likelihood_and_physical_columns_match(
    reference_root, monkeypatch, population_files, scenario, domain, configuration
):
    monkeypatch.syspath_prepend(str(reference_root))
    kwargs = inputs(scenario, domain, length=32 if configuration == "pool" else 31)
    if configuration == "pool":
        kwargs.update(P_orb=(9.5, 10.5), mission="Kepler", flatpriors=True)
        if scenario.startswith("S"):
            kwargs.update(population_files[0])
    if configuration == "empty_pool" and scenario.startswith("S"):
        kwargs["molusc_file"] = population_files[1]
    if configuration == "low_mass":
        kwargs.update(M_s=0.35, R_s=0.4, Teff=3500.0)
        if domain == "real":
            kwargs["sigma"] = np.linspace(0.005, 0.015, len(kwargs["time"]))
    old = capture(monkeypatch, *modules(scenario, domain, True), kwargs)
    new = capture(monkeypatch, *modules(scenario, domain), kwargs)
    assert old["settings"] == new["settings"]
    assert isinstance(new["prior"], ScenarioPrior) and isinstance(
        new["likelihood"], ScenarioLikelihood
    )
    u = points(new["settings"]["ndim"])
    unchanged = u.copy()
    expected = np.array([old["prior"](row) for row in u])
    actual = np.array([new["prior"](row) for row in u])
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_allclose(make_prior(new["prior"])(u), expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(u, unchanged)
    before = np.array([old["likelihood"](row) for row in expected])
    after = np.array([new["likelihood"](row) for row in actual])
    assert np.isfinite(before).any() and np.isneginf(before).any()
    np.testing.assert_array_equal(before, after)
    if domain == "real":
        old_batch = PhysicalBatch(old["likelihood"]).evaluate(expected)
        new_batch = PhysicalBatch(new["likelihood"]).evaluate(actual)
        np.testing.assert_allclose(old_batch[0], new_batch[0], atol=1e-13, rtol=1e-13)
        np.testing.assert_array_equal(old_batch[1], new_batch[1])
        assert old_batch[2].keys() == new_batch[2].keys()
        for name in old_batch[2]:
            np.testing.assert_allclose(
                old_batch[2][name], new_batch[2][name], atol=1e-13, rtol=1e-13
            )


@pytest.mark.parametrize(
    "scenario,host,system",
    [
        ("TP", KnownHost, Planet),
        ("STP", BoundCompanionHost, Planet),
        ("EB", KnownHost, Binary),
        ("SEB", BoundCompanionHost, Binary),
    ],
)
def test_both_axes_share_components_without_closure_inspection(monkeypatch, scenario, host, system):
    from pentaceratops.experimental import batched_priors, physical_batch

    cb = capture(monkeypatch, *modules(scenario, "real"), inputs(scenario, "real"))
    assert isinstance(cb["likelihood"].model.host, host)
    assert isinstance(cb["likelihood"].model.system, system)

    def forbidden(*args):
        raise AssertionError("Migrated scenarios must not inspect closures")

    monkeypatch.setattr(batched_priors, "closure", forbidden)
    monkeypatch.setattr(physical_batch, "closure", forbidden)
    theta = make_prior(cb["prior"])(points(cb["settings"]["ndim"]))
    assert len(PhysicalBatch(cb["likelihood"]).evaluate(theta)[1]) > 0


@pytest.mark.reference
@pytest.mark.parametrize("scenario", ["STP", "EB", "SEB"])
@pytest.mark.parametrize("domain", ["real", "fourier"])
def test_posterior_packing_remains_exact(reference_root, monkeypatch, scenario, domain):
    monkeypatch.syspath_prepend(str(reference_root))
    kwargs = inputs(scenario, domain)
    outputs = []
    for reference in (True, False):
        module, function = modules(scenario, domain, reference)

        def sampler(likelihood, prior, ndim, **settings):
            theta = np.array([prior(u) for u in points(ndim)])
            scores = np.array([likelihood(row) for row in theta])
            weights = np.isfinite(scores).astype(float)
            weights /= weights.sum()
            return -8.0, SimpleNamespace(samples=theta, weights=weights, log_likelihoods=scores)

        with monkeypatch.context() as patch:
            patch.setattr(module, "_run_persistent_evidence", sampler)
            patch.setattr(module, "POSTERIOR_NSAMPLES", 5)
            state = np.random.get_state()
            try:
                np.random.seed(41)
                outputs.append(function(**kwargs))
            finally:
                np.random.set_state(state)
    assert outputs[0].keys() == outputs[1].keys()
    assert len(outputs[1]["M_s"]) == 5  # Do not accidentally compare fallback outputs.
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key], err_msg=key)


@pytest.mark.parametrize("scenario", ["STP", "SEB"])
def test_correlated_fast_and_scalar_likelihood_match(monkeypatch, scenario):
    from pentaceratops.experimental.covariance import patched_engine, scenario_kwargs
    from pentaceratops.experimental.v2_adapter import OptimizedAdapter
    from test_exposure import synthetic_data, assert_agreement
    from pentaceratops.evidence import real, eclipses

    for module in (real, eclipses):
        monkeypatch.setattr(module, "POSTERIOR_NSAMPLES", module.POSTERIOR_NSAMPLES)
    adapter = OptimizedAdapter(synthetic_data(), nsamples=7)
    adapter.aperture_fraction = 0.2
    module, function = modules(scenario, "real")
    star = dict(mass=0.8, rad=0.75, Teff=4800.0, plx=10.0, Tmag=10.0, Jmag=9.0, Hmag=9.0, Kmag=9.0)
    kwargs = scenario_kwargs(function, adapter, star, 2.0, "unused.csv", None, 8, 2, 10)
    with patched_engine(adapter):
        cb = capture(monkeypatch, module, function, kwargs)
        theta = np.array([cb["prior"](row) for row in points(6)])
        scalar = np.array([cb["likelihood"](row) for row in theta])
        fast = adapter.likelihood_batch(cb["likelihood"], theta)
    assert_agreement(scalar, fast)


@pytest.mark.reference
@pytest.mark.parametrize("scenario", ["STP", "EB", "SEB"])
@pytest.mark.parametrize("domain", ["real", "fourier"])
def test_tiny_seeded_evidence_and_saved_pool(
    reference_root, monkeypatch, tmp_path, scenario, domain
):
    from pentaceratops import run_evidence, RunResult

    monkeypatch.syspath_prepend(str(reference_root))
    kwargs = inputs(scenario, domain)
    # Deliberately weak synthetic observations keep this a tiny migration test.
    kwargs["sigma"] = 0.2
    if "sigma_secondary" in kwargs:
        kwargs["sigma_secondary"] = 0.2
    for name in ("var_fourier", "var_fourier_secondary"):
        if name in kwargs:
            kwargs[name] = kwargs[name] * 400
    old, old_function = modules(scenario, domain, True)
    _, new_function = modules(scenario, domain)
    pool = []
    scope = old._run_persistent_evidence.__globals__
    original = scope["persistent_sampling"]

    def record(*args, **options):
        result = original(*args, **options)
        pool.append(result)
        return result

    state = np.random.get_state()
    try:
        with monkeypatch.context() as patch:
            patch.setitem(scope, "persistent_sampling", record)
            np.random.seed(32)
            expected = old_function(**kwargs)
    finally:
        np.random.set_state(state)
    path = tmp_path / "result.npz"
    actual = run_evidence(new_function, seed=32, output_path=path, **kwargs)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual.output[key], err_msg=key)
    saved = next(p for p in RunResult.load(path).sampling if p["kind"] == "sampler")
    assert len(pool) == 1
    for index, key in enumerate(("log_evidence", "weights", "samples", "log_target")):
        np.testing.assert_array_equal(pool[0][index], saved[key], err_msg=key)


@pytest.mark.parametrize("scenario", ["STP", "EB", "SEB"])
@pytest.mark.parametrize("mode", ["scalar", "prior_only", "batch_prior"])
def test_composed_scenarios_run_through_sampler_modes(monkeypatch, scenario, mode):
    from pentaceratops import run_evidence
    from pentaceratops.evidence import real, eclipses
    from pentaceratops.experimental.covariance import patched_engine
    from pentaceratops.experimental.sampler_mode import mode_context
    from pentaceratops.experimental.v2_adapter import OptimizedAdapter
    from test_exposure import synthetic_data

    data = synthetic_data()
    # Weak data and tiny effort exercise plumbing, not evidence precision.
    data["sigma"] = np.full_like(data["sigma"], 0.2)
    for panel in ("even", "odd", "secondary"):
        data[panel + "_sigma"] = np.full_like(data[panel + "_sigma"], 0.2)
    adapter = OptimizedAdapter(data, nsamples=3)
    module, function = modules(scenario, "real")
    for patched in (real, eclipses):
        monkeypatch.setattr(patched, "POSTERIOR_NSAMPLES", patched.POSTERIOR_NSAMPLES)
    original_sampler = module._run_persistent_evidence
    original_normalizer = module._log_sigma_norm
    kwargs = inputs(scenario, "real")
    kwargs.update(P_orb=data["period"], exptime=data["exptime"])
    if scenario != "STP":
        kwargs["secondary_loglike"] = adapter.binary
    with patched_engine(adapter), mode_context(module, adapter, mode) as stats:
        result = run_evidence(function, seed=73, **kwargs)
    assert np.isfinite(result.output["lnZ"])
    assert stats["sampler_invocations"] == 1
    assert (stats["likelihood_batches"] > 0) == (mode == "batch_prior")
    assert (stats["prior_batches"] > 0) == (mode != "scalar")
    pools = [entry for entry in result.sampling if entry["kind"] == "sampler"]
    assert len(pools) == 1 and len(pools[0]["samples"]) > 0
    assert module._run_persistent_evidence is original_sampler
    assert module._log_sigma_norm is original_normalizer
