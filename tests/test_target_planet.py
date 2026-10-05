"""Shared TP model regression tests; no catalogue queries or production runs."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from pentaceratops.evidence import real, fourier
from pentaceratops.evidence.target_planet import TargetPlanetLikelihood, TargetPlanetPrior
from pentaceratops.experimental.batched_priors import make_prior
from pentaceratops.experimental.physical_batch import PhysicalBatch


def inputs(domain="real", length=31, **overrides):
    time = np.linspace(-0.12, 0.12, length)
    kwargs = dict(
        time=time,
        flux=1 - 0.0005 * np.exp(-0.5 * (time / 0.035) ** 2),
        sigma=0.003 if length % 2 else np.linspace(0.002, 0.004, length),
        P_orb=3.0,
        M_s=0.8,
        R_s=0.75,
        Teff=4800.0,
        Z=0.0,
        N=8,
        steps=2,
        mission="TESS",
        exptime=0.002,
        nsamples=3,
    )
    if domain == "fourier":
        kwargs["var_fourier"] = np.linspace(1.0, 4.0, length // 2) * length * 0.003**2
    kwargs.update(overrides)
    return kwargs


def function(module):
    return module.lnZ_TTP_fourier if "fourier" in module.__name__ else module.lnZ_TTP


def capture(monkeypatch, module, kwargs):
    """Obtain the exact production callbacks without starting a sampler."""
    captured = {}

    class Captured(BaseException):
        pass

    def sampler(loglike, prior_transform, **settings):
        captured.update(likelihood=loglike, prior=prior_transform, settings=settings)
        raise Captured()

    with monkeypatch.context() as patch:
        patch.setattr(module, "_run_persistent_evidence", sampler)
        with pytest.raises(Captured):
            function(module)(**kwargs)
    return captured


def prior_points():
    rng = np.random.default_rng(246)
    u = rng.uniform(0.001, 0.999, (32, 5))
    u[:16, 1] = 0.9999  # Include actual transits, not just rejected draws.
    u[:16, 2] *= 0.3
    u[:2, 4] = [0.0, 1.0]  # Radius-support endpoints.
    u[2:4, 2] = [0.0, 1.0]  # Eccentricity clipping endpoints.
    return u


@pytest.mark.reference
@pytest.mark.parametrize("domain", ["real", "fourier"])
@pytest.mark.parametrize("length", [31, 32])
@pytest.mark.parametrize("period", [3.0, (2.8, 3.2)])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize(
    "mission,mass,radius,teff",
    [
        ("TESS", 0.8, 0.75, 4800.0),
        ("Kepler", 0.35, 0.4, 3500.0),
    ],
)
def test_callbacks_match_preserved_tp(
    reference_root, monkeypatch, domain, length, period, flat, mission, mass, radius, teff
):
    monkeypatch.syspath_prepend(str(reference_root))
    name = "marginal_likelihoods_new" if domain == "real" else "marginal_likelihoods_fourier"
    old = importlib.import_module("triceratops." + name)
    new = real if domain == "real" else fourier
    kwargs = inputs(
        domain,
        length,
        P_orb=period,
        flatpriors=flat,
        mission=mission,
        M_s=mass,
        R_s=radius,
        Teff=teff,
    )
    before, after = (capture(monkeypatch, module, kwargs) for module in (old, new))
    assert before["settings"] == after["settings"]
    assert isinstance(after["prior"], TargetPlanetPrior)
    assert isinstance(after["likelihood"], TargetPlanetLikelihood)
    u = prior_points()
    saved = u.copy()
    expected = np.array([before["prior"](row) for row in u])
    actual = np.array([after["prior"](row) for row in u])
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_allclose(make_prior(after["prior"])(u), expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(u, saved)
    expected_values = np.array([before["likelihood"](row) for row in expected])
    actual_values = np.array([after["likelihood"](row) for row in actual])
    assert np.isfinite(expected_values).any() and np.isneginf(expected_values).any()
    np.testing.assert_array_equal(expected_values, actual_values)


@pytest.mark.reference
@pytest.mark.parametrize("domain", ["real", "fourier"])
@pytest.mark.parametrize("fallback", [False, True])
def test_result_packing_and_fallback_match(reference_root, monkeypatch, domain, fallback):
    monkeypatch.syspath_prepend(str(reference_root))
    name = "marginal_likelihoods_new" if domain == "real" else "marginal_likelihoods_fourier"
    old = importlib.import_module("triceratops." + name)
    new = real if domain == "real" else fourier
    draws = np.array(
        [[3.0, 89.0, 0.1, 90.0, 2.0], [3.0, 89.9, 0.2, 75.0, 4.0], [3.0, 88.0, 0.05, 110.0, 1.0]]
    )
    values = np.array([-3.0, -1.0, -5.0])

    def sampler(*args, **kwargs):
        return -2.0, SimpleNamespace(
            samples=draws, weights=np.array([0.1, 0.8, 0.1]), log_likelihoods=values
        )

    def fail(*args, **kwargs):
        raise ValueError("Exercise the inherited fallback schema")

    outputs = []
    for module in (old, new):
        with monkeypatch.context() as patch:
            patch.setattr(module, "_run_persistent_evidence", sampler)
            patch.setattr(module, "POSTERIOR_NSAMPLES", 2)
            if fallback:
                patch.setattr(module, "_resample_equal", fail)
            state = np.random.get_state()
            try:
                np.random.seed(17)
                outputs.append(function(module)(**inputs(domain)))
            finally:
                np.random.set_state(state)
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key], err_msg=key)
    assert outputs[1]["R_p"][0] == (1.0 if fallback else 4.0)
    assert outputs[1]["best_lnL"] == -1.0


@pytest.mark.reference
def test_explicit_physical_batch_matches_legacy(reference_root, monkeypatch):
    monkeypatch.syspath_prepend(str(reference_root))
    old = importlib.import_module("triceratops.marginal_likelihoods_new")
    before = capture(monkeypatch, old, inputs())
    after = capture(monkeypatch, real, inputs())
    theta = make_prior(after["prior"])(prior_points())
    theta = np.vstack([theta, np.full(5, np.nan), [3.0, 90.0, 1.0, 0.0, 1.0]])
    expected = PhysicalBatch(before["likelihood"]).evaluate(theta)
    actual = PhysicalBatch(after["likelihood"]).evaluate(theta)
    for left, right in zip(expected[:2], actual[:2]):
        np.testing.assert_array_equal(left, right)
    assert expected[2].keys() == actual[2].keys()
    for key in expected[2]:
        np.testing.assert_array_equal(expected[2][key], actual[2][key], err_msg=key)


def test_tp_fast_dispatch_does_not_inspect_closures(monkeypatch):
    from pentaceratops.experimental import batched_priors, physical_batch

    callbacks = capture(monkeypatch, real, inputs())

    def forbidden(*args):
        raise AssertionError("TP dispatch must use its explicit model contract")

    monkeypatch.setattr(batched_priors, "closure", forbidden)
    monkeypatch.setattr(physical_batch, "closure", forbidden)
    batch = make_prior(callbacks["prior"])
    physics = PhysicalBatch(callbacks["likelihood"])
    assert physics.kind == "planet" and physics.nsamples == 3
    assert len(physics.evaluate(batch(prior_points()))[1]) > 0


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(5),
        np.zeros((2, 6)),
        np.full((2, 5), np.nan),
        np.full((2, 5), -0.1),
        np.full((2, 5), 1.1),
    ],
)
def test_batch_prior_rejects_bad_inputs(monkeypatch, bad):
    prior = capture(monkeypatch, real, inputs())["prior"]
    with pytest.raises(ValueError):
        make_prior(prior)(bad)


def test_fourier_is_not_silently_dispatched_as_real_fast_backend(monkeypatch):
    likelihood = capture(monkeypatch, fourier, inputs("fourier"))["likelihood"]
    with pytest.raises(ValueError, match="real-space"):
        PhysicalBatch(likelihood)


@pytest.mark.parametrize("mode", ["scalar", "prior_only", "batch_prior"])
def test_shared_tp_runs_through_sampler_modes(monkeypatch, mode):
    from pentaceratops import run_evidence
    from pentaceratops.experimental.covariance import patched_engine
    from pentaceratops.experimental.sampler_mode import mode_context
    from pentaceratops.experimental.v2_adapter import OptimizedAdapter
    from test_exposure import synthetic_data

    data = synthetic_data()
    adapter = OptimizedAdapter(data, nsamples=3)
    original_sampler, original_cost = real._run_persistent_evidence, real.lnL_TP
    with patched_engine(adapter), mode_context(real, adapter, mode) as stats:
        result = run_evidence(
            real.lnZ_TTP, seed=73, **inputs(P_orb=data["period"], exptime=data["exptime"])
        )
    assert np.isfinite(result.output["lnZ"])
    assert stats["sampler_invocations"] == 1
    assert (stats["likelihood_batches"] > 0) == (mode == "batch_prior")
    assert (stats["prior_batches"] > 0) == (mode != "scalar")
    pools = [entry for entry in result.sampling if entry["kind"] == "sampler"]
    assert len(pools) == 1 and len(pools[0]["samples"]) > 0
    assert real._run_persistent_evidence is original_sampler and real.lnL_TP is original_cost
