"""Free transit-time offset: model shift, sampler coordinate, and public options."""
import numpy as np
import pytest

from pentaceratops.evidence.joint_fourier import (
    DEFAULT_EPOCH_SIGMA, resolve_epoch_sigma, sample_joint_scenario,
)
from pentaceratops.experimental.joint_fourier import JointFourierAdapter
from pentaceratops.likelihoods.joint_fourier import GaussianBlock
from test_joint_fourier import fixture


def test_option_resolution():
    assert resolve_epoch_sigma() is None
    assert resolve_epoch_sigma(True) == DEFAULT_EPOCH_SIGMA == pytest.approx(3/1440)
    assert resolve_epoch_sigma(True, .001) == .001
    assert resolve_epoch_sigma(False) is None
    for arguments in ((False, .001), (True, 0.), (True, -1.), (True, np.inf), (True, True), (1, None)):
        with pytest.raises(ValueError):
            resolve_epoch_sigma(*arguments)


@pytest.mark.parametrize("kind", ["planet", "binary", "x2p"])
def test_offset_equals_a_shifted_epoch_in_scalar_and_batched_models(kind):
    delta = .0031
    adapter, p = fixture(kind)
    rows = [{k: v[i] for k, v in p.items()} for i in range(6)]
    fixed = np.array([adapter.native_model(kind, row) for row in rows])
    adapter.scalar_epoch_offset = delta
    moved = np.array([adapter.native_model(kind, row) for row in rows])
    adapter.scalar_epoch_offset = 0.
    assert np.max(abs(moved-fixed)) > 1e-5
    # Batched rows carry their own offsets and must match the scalar reference.
    offsets = np.array([delta, 0., -delta, delta, 0., 2*delta])
    dense = []
    for row, value in zip(rows, offsets):
        adapter.scalar_epoch_offset = value
        dense.append(adapter.native_model(kind, row))
    adapter.scalar_epoch_offset = 0.
    sparse = adapter._project(kind, dict(p, epoch_offset=offsets))
    np.testing.assert_allclose(sparse, np.array(dense), atol=3e-9, rtol=1e-12)
    np.testing.assert_allclose(adapter.native_model(kind, rows[1]), fixed[1])
    # A later transit is the same model seen at earlier relative times.
    n = 400; t = np.linspace(-.5, 7.5, n)
    late = JointFourierAdapter([GaussianBlock.from_spectrum(t, np.ones(n), np.full(n//2+1, n*.002**2), t[1]-t[0])],
                               4., delta, nsamples=7)
    early = JointFourierAdapter([GaussianBlock.from_spectrum(t, np.ones(n), np.full(n//2+1, n*.002**2), t[1]-t[0])],
                                4., 0., nsamples=7)
    early.scalar_epoch_offset = delta
    np.testing.assert_allclose(early.native_model(kind, rows[2]), late.native_model(kind, rows[2]), atol=1e-12)


@pytest.mark.parametrize("scenario", ["TP", "EB"])
def test_sampler_returns_offsets_and_restores_the_adapter(scenario):
    t = np.linspace(-.3, 7.7, 120)
    block = GaussianBlock.from_covariance(t, np.ones(len(t)), np.eye(len(t))*.1**2, .003)
    adapter = JointFourierAdapter([block], 4., 0.)
    star = dict(mass=1., rad=1., Teff=5700., plx=10., Tmag=10., Jmag=9., Hmag=9., Kmag=9.)
    fixed = sample_joint_scenario(adapter, scenario, star, trilegal="unused", N=8, steps=2,
                                  seed=307, posterior_samples=19)
    assert "epoch_offset" not in fixed
    sigma = .002
    result = sample_joint_scenario(adapter, scenario, star, trilegal="unused", N=8, steps=2,
                                   seed=307, posterior_samples=19, epoch_sigma=sigma)
    assert np.isfinite(result["lnBF"])
    assert result["epoch_sigma"] == sigma
    assert result["epoch_offset"].shape == (len(result["physical_parameters"]),)
    assert result["physical_parameters"].shape[1] == fixed["physical_parameters"].shape[1]
    np.testing.assert_array_equal(result["posterior_epoch_offset"],
                                  result["epoch_offset"][result["posterior_source_index"]])
    # Uninformative data: the offsets keep the scale of their prior.
    assert np.all(abs(result["epoch_offset"]) < 8*sigma)
    assert result["bestfit"]["scalar_replay"] == pytest.approx(result["bestfit"]["log_target"], abs=2e-5)
    assert "epoch_offset" in result["bestfit"]["parameters"]
    assert adapter.epoch_sigma is None and adapter.scalar_epoch_offset == 0.


def test_public_options_default_off_for_both_likelihoods():
    from pentaceratops.prepared import _options
    for domain in ("fourier", "real"):
        options = _options(domain, {})
        assert options["free_epoch"] is False and options["epoch_sigma"] is None
        assert _options(domain, dict(free_epoch=True))["free_epoch"] is True
        assert _options(domain, dict(free_epoch=True, epoch_sigma=.001))["epoch_sigma"] == .001
        for settings in (dict(epoch_sigma=.001), dict(free_epoch=True, epoch_sigma=-1.), dict(free_epoch="yes")):
            with pytest.raises(ValueError):
                _options(domain, settings)


@pytest.mark.parametrize("primary_only", [False, True])
def test_subtracted_likelihood_batched_offsets_match_the_scalar_reference(monkeypatch, primary_only):
    import importlib
    from pentaceratops.hz import real_adapter
    from pentaceratops.experimental.covariance import scenario_functions, patched_engine
    from pentaceratops.evidence import real, eclipses
    from test_all_scenarios import population
    from test_exposure import synthetic_data
    from test_hz import target
    data = synthetic_data()
    if primary_only:
        n = len(data["even_time"])+len(data["odd_time"])
        for key in ("time", "mean_flux", "sigma"):
            data["secondary_"+key] = np.empty(0)
        for key in ("flux", "sigma", "map_flux", "factor"):
            data[key] = data[key][:n]
    adapter = real_adapter(data, mission="TESS", primary_only=primary_only, nsamples=7)
    monkeypatch.setattr(real, "trilegal_results", population)
    monkeypatch.setattr(eclipses, "trilegal_results", population)
    class Capture(BaseException):
        pass
    moved = 0
    for name, function in scenario_functions().items():
        module = importlib.import_module(function.__module__)
        kwargs = adapter.scenario_kwargs(function, target().stars.iloc[0], trilegal="unused")
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
        offsets = rng.normal(0., .004, len(physical))
        offsets[0] = 0.
        with patched_engine(adapter):
            fixed = adapter.likelihood_batch(saved["like"], physical)
            scalar = []
            for row, offset in zip(physical, offsets):
                adapter.scalar_epoch_offset = float(offset)
                try: scalar.append(saved["like"](row))
                finally: adapter.scalar_epoch_offset = 0.
            adapter.epoch_sigma = .004
            try: batch = adapter.likelihood_batch(saved["like"], np.column_stack([physical, offsets]))
            finally: adapter.epoch_sigma = None
            again = np.array([saved["like"](row) for row in physical])
        scalar = np.array(scalar)
        assert np.isfinite(scalar).any(), name
        np.testing.assert_allclose(batch, scalar, atol=2e-5, rtol=2e-12)
        assert batch[0] == pytest.approx(fixed[0], abs=1e-9) or not np.isfinite(fixed[0])
        # The scalar reference is unchanged once the offset is cleared (no stale model times).
        np.testing.assert_allclose(again, fixed, atol=2e-5, rtol=2e-12)
        both = np.isfinite(batch) & np.isfinite(fixed)
        moved += int(np.sum(abs(batch[both]-fixed[both]) > 1e-6))
    assert moved > 0


def test_public_subtracted_run_with_a_free_epoch(tmp_path):
    from pentaceratops import calc_probs_folded_real
    from test_exposure import synthetic_data
    from test_hz import target
    data = synthetic_data()
    for panel in ("even", "odd", "secondary"):
        data[panel+"_sigma"][:] = .1
    data["sigma"][:] = .1
    common = dict(N=8, steps=2, nsamples=3, seed=307, posterior_samples=11, eb_eta=.1,
                  scenarios=("TP", "EB", "EBx2P"))
    fixed = calc_probs_folded_real(target(), data, **common)
    assert "epoch_offset" not in fixed.output and fixed.output.attrs["free_epoch"] is False
    result = calc_probs_folded_real(target(), data, free_epoch=True, epoch_sigma=.002, **common)
    table = result.output
    assert table.attrs["free_epoch"] is True and table.attrs["epoch_sigma"] == .002
    assert np.isfinite(table.epoch_offset).all() and np.all(table.epoch_offset_err > 0)
    for record in result.metadata["scenario_records"].values():
        assert record["posterior_epoch_offset"].shape == (11,)
        assert np.all(abs(record["epoch_offset"]) < 8*.002)
        assert "epoch_offset" in record["bestfit"]["parameters"]
        assert record["bestfit"]["scalar_replay"] == pytest.approx(record["bestfit"]["log_target"], abs=2e-5)


def test_paper_likelihood_names_and_their_earlier_aliases():
    from pentaceratops.prepared import _options, likelihood_domain
    assert likelihood_domain("subtracted") == likelihood_domain("real") == "real"
    assert likelihood_domain("marginalized") == likelihood_domain("fourier") == "fourier"
    for name in ("Subtracted", "marg", "", None, 3, ["real"]):
        with pytest.raises(ValueError, match="subtracted"):
            likelihood_domain(name)
    assert _options("subtracted", {}) == _options("real", {})
    assert _options("marginalized", {}) == _options("fourier", {})
    assert _options("subtracted", {})["nsamples"] == 20 and _options("marginalized", {})["nsamples"] == 7
    with pytest.raises(ValueError):
        _options("marginalized", dict(primary_only=True))


@pytest.mark.parametrize("name,alias", [("subtracted", "real"), ("marginalized", "fourier")])
def test_evidence_records_the_paper_name_for_either_spelling(monkeypatch, tmp_path, name, alias):
    from pentaceratops import hz
    from pentaceratops.evidence import evidence
    from test_prepared_command import prepared
    from test_evidence_api import field
    saved, _, _ = prepared(monkeypatch)
    host = field(tmp_path)

    def sample(adapter, scenario, star, **options):
        return dict(lnBF=1., lnZ=2., null_loglike=1., N=options["N"], steps=options["steps"])
    monkeypatch.setattr(hz, "sample_joint_scenario", sample)
    tables = []
    for spelling in (name, alias):
        result = evidence(saved, target=host, likelihood=spelling)
        assert result.metadata["evidence_call"]["likelihood"] == name
        tables.append(result.output[["ID", "scenario", "lnBF", "prob"]])
    assert tables[0].equals(tables[1])
    with pytest.raises(ValueError, match="subtracted"):
        evidence(saved, target=host, likelihood="windowed")
