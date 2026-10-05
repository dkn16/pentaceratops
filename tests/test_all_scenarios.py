"""Differential coverage for every public evidence recipe and both EB branches."""

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pentaceratops.evidence.scenario import ScenarioPrior, ScenarioLikelihood
from pentaceratops.experimental.batched_priors import make_prior
from pentaceratops.experimental.physical_batch import PhysicalBatch

MODULES = {
    "real": "marginal_likelihoods_new",
    "fourier": "marginal_likelihoods_fourier",
    "eclipses": "marginal_likelihoods_eb",
    "fourier_eclipses": "marginal_likelihoods_eb_fourier",
}


def recipes():
    import pentaceratops

    root = Path(pentaceratops.__file__).parent / "evidence"
    return [
        (module, node.name)
        for module in MODULES
        for node in ast.parse((root / (module + ".py")).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("lnZ_")
    ]


RECIPES = recipes()


def population(*unused):
    return tuple(
        np.array(x, float)
        for x in (
            [9.5, 10.5, 11.2, 12.0],
            [0.6, 0.9, 0.4, 1.2],
            [4.5, 4.5, 4.8, 3.0],
            [3550.5, 4875.1, 3500.0, 6100.0],
            [0.0, -0.2, 0.1, 0.3],
            [8.5, 9.5, 10.2, 11.0],
            [8.0, 9.0, 9.7, 10.5],
            [7.5, 8.5, 9.2, 10.0],
        )
    )


def cube(ndim):
    u = np.random.default_rng(112).uniform(0.02, 0.9, (32, ndim))
    u[:24, 1] = 0.999999
    u[:24, 2] = 0.03
    u[:24, 4] = np.tile([0.0, 0.1, 0.5, 0.8, 0.95, 0.99, 0.99999, 1.0], 3)
    if ndim == 6:
        u[:24, 5] = np.repeat([0.15, 0.35, 0.65], 8)
    return u


def kwargs_for(function, config, tmp_path):
    t = np.linspace(-0.2, 0.2, 31 if config != "catalog" else 32)
    other = np.linspace(-0.18, 0.18, 29)
    flux = 1 - 0.003 * np.exp(-0.5 * (t / 0.04) ** 2)
    error = 0.02 if config != "low_mass" else np.linspace(0.01, 0.03, len(t))
    values = dict(
        time=t,
        flux=flux,
        sigma=error,
        time_even=t,
        flux_even=flux,
        sigma_even=error,
        time_odd=other,
        flux_odd=np.ones(len(other)),
        sigma_odd=0.025,
        time_secondary=other,
        flux_secondary=np.ones(len(other)),
        sigma_secondary=0.025,
        var_fourier=np.full(len(t) // 2, 0.02),
        var_fourier_even=np.full(len(t) // 2, 0.02),
        var_fourier_odd=np.full(len(other) // 2, 0.03),
        var_fourier_secondary=np.full(len(other) // 2, 0.03),
        P_orb=5.0,
        M_s=0.8,
        R_s=0.75,
        Teff=4800.0,
        Z=0.0,
        plx=10.0,
        Tmag=10.0,
        Jmag=9.0,
        Hmag=8.5,
        Kmag=8.0,
        trilegal_fname="unused.csv",
        N=8,
        steps=2,
        exptime=0.002,
        nsamples=3,
        max_shift=0.3,
    )
    if config == "catalog":
        contrast = tmp_path / "contrast.csv"
        contrast.write_text("0.1,1\n0.5,3\n2.0,7\n")
        pool = tmp_path / "companions.csv"
        pool.write_text(
            "semi-major axis(AU),eccentricity,mass ratio\n30,0.1,0.01\n40,0,0.4\n50,0.2,0.75\n60,0,0.99\n"
        )
        values.update(
            P_orb=(9.5, 10.5),
            mission="Kepler",
            flatpriors=True,
            contrast_curve_file=str(contrast),
            filt="J",
            molusc_file=str(pool),
        )
    elif config == "low_mass":
        values.update(M_s=0.35, R_s=0.4, Teff=3500.0)
        if "_evolved" in function.__name__:
            # The evolved recipe infers mass from radius and logg=3; .4 Rsun
            # would imply an unphysical .006 Msun subgiant.
            values["R_s"] = 2.5
        if "EB" in function.__name__ and not any(
            suffix in function.__name__ for suffix in ("_secondary", "_evenodd")
        ):
            # Legacy hard secondary veto requires a scalar sigma. Weak data
            # ensure its low-mass branch has finite points for packing checks.
            values["sigma"] = 0.2
    if "_evenodd_fourier" in function.__name__:
        # The inherited Fourier parity swap requires matching grids, as
        # provided by the dispatcher. Unequal grids are tested in real space.
        values.update(
            time_odd=t,
            flux_odd=np.ones(len(t)),
            sigma_even=0.02,
            sigma_odd=0.025,
            var_fourier_odd=np.full(len(t) // 2, 0.03),
        )
    signature = inspect.signature(function)
    return {k: v for k, v in values.items() if k in signature.parameters}


def evaluate_recipe(monkeypatch, module, function, kwargs, population_source=None):
    calls = []

    def sampler(likelihood, prior, ndim, **settings):
        u = cube(ndim)
        samples = np.array([prior(row) for row in u])
        scores = np.array([likelihood(row) for row in samples])
        finite = np.isfinite(scores)
        assert finite.any(), function.__name__
        weights = finite.astype(float) / finite.sum()
        calls.append((prior, likelihood, samples, scores, settings))
        return -8.0, SimpleNamespace(samples=samples, weights=weights, log_likelihoods=scores)

    state = np.random.get_state()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                module,
                "trilegal_results",
                population if population_source is None else population_source,
            )
            patch.setattr(module, "_run_persistent_evidence", sampler)
            patch.setattr(module, "POSTERIOR_NSAMPLES", 5)
            np.random.seed(39)
            output = function(**kwargs)
    finally:
        np.random.set_state(state)
    return calls, output


def assert_output(before, after):
    if isinstance(before, tuple):
        assert len(before) == len(after)
        for b, a in zip(before, after):
            assert_output(b, a)
        return
    assert before.keys() == after.keys()
    assert np.size(after["M_s"]) == 5  # Do not mistake fallback output for a passing test.
    for key in before:
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)


@pytest.mark.reference
@pytest.mark.parametrize("module_name,function_name", RECIPES)
@pytest.mark.parametrize("config", ["default", "catalog", "low_mass"])
def test_every_recipe_matches_reference(
    reference_root, monkeypatch, tmp_path, module_name, function_name, config
):
    monkeypatch.syspath_prepend(str(reference_root))
    old = importlib.import_module("triceratops." + MODULES[module_name])
    new = importlib.import_module("pentaceratops.evidence." + module_name)
    oldfn, newfn = getattr(old, function_name), getattr(new, function_name)
    kw = kwargs_for(newfn, config, tmp_path)
    before, boutput = evaluate_recipe(monkeypatch, old, oldfn, kw)
    after, aoutput = evaluate_recipe(monkeypatch, new, newfn, kw)
    assert len(before) == len(after) and len(after) in (1, 2)
    for b, a in zip(before, after):
        assert isinstance(a[0], ScenarioPrior) and isinstance(a[1], ScenarioLikelihood)
        assert b[4] == a[4]
        np.testing.assert_array_equal(b[2], a[2])
        np.testing.assert_array_equal(b[3], a[3])
        u = cube(a[2].shape[1])
        saved = u.copy()
        np.testing.assert_allclose(make_prior(a[0])(u), b[2], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(u, saved)
        is_window = "_secondary" in function_name or "_evenodd" in function_name
        if (
            module_name == "eclipses"
            and is_window
            or module_name == "real"
            and function_name in {"lnZ_TTP", "lnZ_PTP", "lnZ_STP", "lnZ_DTP", "lnZ_BTP"}
        ):
            bb, ab = PhysicalBatch(b[1]).evaluate(b[2]), PhysicalBatch(a[1]).evaluate(a[2])
            np.testing.assert_allclose(bb[0], ab[0], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(bb[1], ab[1])
            for name in bb[2]:
                np.testing.assert_allclose(bb[2][name], ab[2][name], rtol=1e-12, atol=1e-12)
    assert_output(boutput, aoutput)


@pytest.mark.reference
@pytest.mark.parametrize("module_name,function_name", RECIPES)
def test_seeded_evidence_and_complete_saved_pools(
    reference_root, monkeypatch, tmp_path, module_name, function_name
):
    from pentaceratops import run_evidence, RunResult

    monkeypatch.syspath_prepend(str(reference_root))
    old = importlib.import_module("triceratops." + MODULES[module_name])
    new = importlib.import_module("pentaceratops.evidence." + module_name)
    oldfn, newfn = getattr(old, function_name), getattr(new, function_name)
    kwargs = kwargs_for(newfn, "default", tmp_path)
    kwargs.update(N=6, steps=1)
    for key in ("sigma", "sigma_even", "sigma_odd", "sigma_secondary"):
        if key in kwargs:
            kwargs[key] = 0.2
    for key in ("var_fourier", "var_fourier_even", "var_fourier_odd", "var_fourier_secondary"):
        if key in kwargs:
            kwargs[key] = np.full_like(kwargs[key], 0.5)
    for module in (old, new):
        monkeypatch.setattr(module, "trilegal_results", population)
        monkeypatch.setattr(module, "POSTERIOR_NSAMPLES", 5)
    pools = []
    scope = old._run_persistent_evidence.__globals__
    original = scope["persistent_sampling"]

    def recorded(*args, **options):
        result = original(*args, **options)
        pools.append(result)
        return result

    state = np.random.get_state()
    try:
        with monkeypatch.context() as patch:
            patch.setitem(scope, "persistent_sampling", recorded)
            np.random.seed(109)
            expected = oldfn(**kwargs)
    finally:
        np.random.set_state(state)
    path = tmp_path / "seeded.npz"
    result = run_evidence(newfn, seed=109, output_path=path, **kwargs)
    assert_output(expected, result.output)
    saved = [p for p in RunResult.load(path).sampling if p["kind"] == "sampler"]
    assert len(pools) == len(saved) and len(saved) in (1, 2)
    for before, after in zip(pools, saved):
        for i, key in enumerate(("log_evidence", "weights", "samples", "log_target")):
            np.testing.assert_array_equal(before[i], after[key], err_msg=key)


@pytest.mark.parametrize(
    "family,kind",
    [(family, kind) for family in "TPSDB" for kind in ("planet", "secondary", "evenodd")]
    + [("N_unknown", "planet"), ("N_evolved", "planet")],
)
def test_fast_covariance_and_metadata(monkeypatch, tmp_path, family, kind):
    from pentaceratops.evidence import real, eclipses
    from pentaceratops.experimental import batched_priors, physical_batch
    from pentaceratops.experimental.covariance import patched_engine, scenario_kwargs
    from pentaceratops.experimental.v2_adapter import OptimizedAdapter
    from test_exposure import synthetic_data, assert_agreement
    from test_scenario_components import capture

    module = real if kind == "planet" else eclipses
    name = f"lnZ_{family}TP" if kind == "planet" else f"lnZ_{family}EB_{kind}"
    if family.startswith("N_"):
        name = "lnZ_NTP_" + family.split("_")[1]
    function = getattr(module, name)
    for mod in (real, eclipses):
        monkeypatch.setattr(mod, "trilegal_results", population)
        monkeypatch.setattr(mod, "POSTERIOR_NSAMPLES", mod.POSTERIOR_NSAMPLES)
    adapter = OptimizedAdapter(synthetic_data(), nsamples=7)
    adapter.aperture_fraction = 0.2
    star = dict(mass=0.8, rad=0.75, Teff=4800.0, plx=10.0, Tmag=10.0, Jmag=9.0, Hmag=8.5, Kmag=8.0)
    kw = scenario_kwargs(function, adapter, star, 2.0, "unused.csv", None, 8, 2, 5)

    def forbidden(*unused):
        raise AssertionError("A composed scenario must not inspect legacy closures")

    monkeypatch.setattr(batched_priors, "closure", forbidden)
    monkeypatch.setattr(physical_batch, "closure", forbidden)
    with patched_engine(adapter):
        cb = capture(monkeypatch, module, function, kw)
        theta = make_prior(cb["prior"])(cube(cb["settings"]["ndim"]))
        scalar = np.array([cb["likelihood"](row) for row in theta])
        batch = adapter.likelihood_batch(cb["likelihood"], theta)
        invalid = np.full((1, theta.shape[1]), np.nan)
        base, ids, _ = PhysicalBatch(cb["likelihood"]).evaluate(np.vstack([theta, invalid]))
        assert np.isneginf(base[-1]) and len(theta) not in ids
    assert_agreement(scalar, batch)


@pytest.mark.reference
@pytest.mark.parametrize("module_name,function_name", [r for r in RECIPES if "_unknown" in r[1]])
def test_unknown_empty_population_unchanged(
    reference_root, monkeypatch, tmp_path, module_name, function_name
):
    monkeypatch.syspath_prepend(str(reference_root))
    outputs = []
    for prefix, name in (
        ("triceratops.", MODULES[module_name]),
        ("pentaceratops.evidence.", module_name),
    ):
        module = importlib.import_module(prefix + name)
        function = getattr(module, function_name)

        def empty(*unused):
            values = list(population())
            values[0] = np.full(4, 20.0)
            return tuple(values)

        monkeypatch.setattr(module, "trilegal_results", empty)
        outputs.append(function(**kwargs_for(function, "default", tmp_path)))
    assert outputs[0] == outputs[1]


@pytest.mark.reference
@pytest.mark.parametrize("module_name,function_name", [r for r in RECIPES if "_unknown" in r[1]])
def test_unknown_host_cuts_preserve_population_prior(
    reference_root, monkeypatch, tmp_path, module_name, function_name
):
    monkeypatch.syspath_prepend(str(reference_root))
    old = importlib.import_module("triceratops." + MODULES[module_name])
    new = importlib.import_module("pentaceratops.evidence." + module_name)

    def mixed(*unused):
        values = list(population())
        # Remains in the magnitude-selected prior, but fails likelihood.
        values[2][0] = 3.0
        return tuple(values)

    kw = kwargs_for(getattr(new, function_name), "default", tmp_path)
    before, boutput = evaluate_recipe(monkeypatch, old, getattr(old, function_name), kw, mixed)
    after, aoutput = evaluate_recipe(monkeypatch, new, getattr(new, function_name), kw, mixed)
    for b, a in zip(before, after):
        np.testing.assert_array_equal(b[2], a[2])
        np.testing.assert_array_equal(b[3], a[3])
        assert (a[2][:, 5] == 0).any() and np.isneginf(a[3][a[2][:, 5] == 0]).all()
    assert_output(boutput, aoutput)
