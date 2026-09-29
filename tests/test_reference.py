"""Opt-in comparisons to the preserved working tree, never an installed snapshot."""
import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("old,new", [
    ("likelihoods", "likelihoods.real"), ("likelihoods_fourier", "likelihoods.fourier"),
    ("priors", "priors"), ("persistent", "sampling.persistent"),
    ("marginal_likelihoods_new", "evidence.real"),
    ("marginal_likelihoods_eb", "evidence.eclipses"),
    ("marginal_likelihoods_fourier", "evidence.fourier"),
    ("marginal_likelihoods_eb_fourier", "evidence.fourier_eclipses"),
    ("calc_probs_fourier", "fourier"), ("scenario_sampling", "sampling.policy"),
])
def test_numerical_function_bodies_unchanged(reference_root, old, new):
    import pentaceratops
    source = (reference_root / "triceratops" / (old+".py")).read_text()
    if old == "persistent":
        # The only numerical-body change: fix the scalar ndim=1 covariance.
        source = source.replace(
            "cov_matrix = np.cov(pool_us, rowvar=False, aweights=weights_norm)",
            "cov_matrix = np.atleast_2d(np.cov(pool_us, rowvar=False, aweights=weights_norm))")
    old_tree = ast.parse(source)
    new_path = Path(pentaceratops.__file__).parent.joinpath(*new.split(".")).with_suffix(".py")
    new_tree = ast.parse(new_path.read_text())
    before = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    after = {node.name: node for node in new_tree.body if isinstance(node, ast.FunctionDef)}
    assert before.keys() == after.keys()
    for name, node in before.items():
        # Observation-only decorator is outside the preserved numerical body.
        assert ast.dump(ast.Module(body=node.body, type_ignores=[])) == ast.dump(
            ast.Module(body=after[name].body, type_ignores=[])), name


def test_physical_templates_match(reference_root, monkeypatch):
    monkeypatch.syspath_prepend(str(reference_root))
    old = importlib.import_module("triceratops.likelihoods")
    from pentaceratops.likelihoods import real as new
    assert Path(old.__file__).resolve().is_relative_to(reference_root)
    even = np.linspace(-.1, .1, 15)
    odd = np.linspace(-.1, .1, 12)
    tp = dict(R_p=4., P_orb=5., inc=89., a=8.5e11, R_s=1., u1=.3, u2=.2,
              ecc=.03, argp=90., companion_fluxratio=.2, exptime=.002, nsamples=3)
    np.testing.assert_array_equal(old.simulate_TP_transit(even, **tp),
                                  new.simulate_TP_transit(even, **tp))
    eb = {key: val for key, val in tp.items() if key != "R_p"}
    eb.update(R_EB=.2, EB_fluxratio=.05)
    for name in ("simulate_EB_transit_secondary", "simulate_EB_transit_evenodd"):
        expected = getattr(old, name)(even, odd, **eb)
        actual = getattr(new, name)(even, odd, **eb)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)


def test_persistent_sampling_seeded_reference(reference_root, monkeypatch):
    monkeypatch.syspath_prepend(str(reference_root))
    old = importlib.import_module("triceratops.persistent")
    from pentaceratops.sampling.persistent import persistent_sampling
    from pentaceratops import run_evidence
    args = (lambda x: -3*np.sum(x**2), lambda x: 2*x-1, 30, 2)
    np.random.seed(3)
    before = old.persistent_sampling(*args, target_ess=60, mcmc_steps=3)
    after = run_evidence(persistent_sampling, *args, target_ess=60, mcmc_steps=3, seed=3)
    for left, right in zip(before, after.output):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("domain", ["real", "fourier"])
def test_small_tp_evidence_and_posterior_match(reference_root, monkeypatch, domain, tmp_path):
    """Low-SNR synthetic TP, tiny diagnostic effort; not a precision benchmark."""
    monkeypatch.syspath_prepend(str(reference_root))
    old_name = "marginal_likelihoods_new" if domain == "real" else "marginal_likelihoods_fourier"
    old = importlib.import_module("triceratops."+old_name)
    new = importlib.import_module("pentaceratops.evidence."+domain)
    from pentaceratops import RunResult, run_evidence
    function_name = "lnZ_TTP" if domain == "real" else "lnZ_TTP_fourier"
    time = np.linspace(-.12, .12, 31)
    flux = 1 - .0005*np.exp(-.5*(time/.035)**2)
    kwargs = dict(time=time, flux=flux, sigma=.003, P_orb=3., M_s=.8, R_s=.75,
                  Teff=4800., Z=0., N=8, steps=2, mission="TESS", exptime=.002, nsamples=3)
    if domain == "fourier":
        kwargs["var_fourier"] = np.full(len(time)//2, len(time)*.003**2)
    np.random.seed(73)
    before = getattr(old, function_name)(**kwargs)
    path = tmp_path / (domain+".npz")
    after = run_evidence(getattr(new, function_name), seed=73, output_path=path, **kwargs)
    assert before.keys() == after.output.keys()
    for key in before:
        np.testing.assert_array_equal(before[key], after.output[key], err_msg=key)
    events = [entry for entry in after.sampling if entry["kind"] == "evidence"]
    pools = [entry for entry in after.sampling if entry["kind"] == "sampler"]
    assert len(events) == len(pools) == 1
    assert pools[0]["event"] == events[0]["event"]
    assert events[0]["status"] == "complete"
    assert np.isfinite(pools[0]["log_evidence"])
    loaded = RunResult.load(path)
    saved_pool = next(entry for entry in loaded.sampling if entry["kind"] == "sampler")
    np.testing.assert_array_equal(saved_pool["samples"], pools[0]["samples"])
    np.testing.assert_array_equal(saved_pool["log_target"], pools[0]["log_target"])
