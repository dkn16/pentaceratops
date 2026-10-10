"""Saved-preparation covariance bridge and an actual offline evidence command."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix

from pentaceratops import RunResult, run_prepared, calc_probs_folded_real, calc_probs_folded_fourier
from pentaceratops.cli import main
from pentaceratops.preprocessing import candidate as prep
from pentaceratops.preprocessing.evidence_inputs import candidate_folded_real, candidate_folded_fourier
from pentaceratops.preprocessing.posterior import conditional_posterior
from pentaceratops.likelihoods.joint_fourier import observed_covariance, total_fft_power
from test_candidate_examples import fixtures, fake_fgp
from test_hz import target


def prepared(monkeypatch, *, two_sectors=False):
    ephem, block, options = fixtures()
    block["flux"][15:20] = np.nan
    blocks = [block]
    if two_sectors:
        other = dict(block, time=block["time"]+15., error=block["error"]*1.6, segment=2)
        blocks.append(other)
    monkeypatch.setattr(prep, "detrend_regular_segment", fake_fgp)
    output, _ = prep.preprocess_blocks(blocks, ephem, options=options)
    saved = RunResult(output, metadata=dict(artifact="candidate_preparation", likelihood_ready=False,
        settings=dict(candidate=asdict(ephem), options=asdict(options))))
    # Constructing either likelihood must use the saved state exclusively.
    monkeypatch.setattr(prep, "detrend_regular_segment", lambda *a, **k: pytest.fail("Do not refit"))
    return saved, ephem, options


def configuration(tmp_path, saved):
    saved.save(tmp_path/"candidate.npz")
    target().stars.to_csv(tmp_path/"stars.csv", index=False)
    (tmp_path/"population.csv").write_text("offline fixture; TP/EB/NTP do not read this population\n")
    config = dict(schema_version=1, mission="TESS", target_id=123, stars="stars.csv",
        trilegal="population.csv", input=dict(format="candidate", path="candidate.npz"),
        likelihood="real", output="result.npz", settings=dict(N=6, steps=1, nsamples=3,
            seed=307, posterior_samples=9, eb_eta=.1, scenarios=["TP", "EB", "NTP"]))
    path = tmp_path/"target.json"
    path.write_text(json.dumps(config))
    return path, config


def test_real_covariance_matches_dense_conditional_reference(monkeypatch):
    saved, ephem, options = prepared(monkeypatch, two_sectors=True)
    data = candidate_folded_real(saved)
    windows, segments = saved.output["windows"], saved.output["segments"]
    m = len(windows["time_even"])
    means, covariances, raw_weights = [], [], []
    for sector in segments:
        t, obs = sector["time"], sector["observed"]
        mapping, _ = prep.assignments(t, obs, ephem, windows["time_even"], windows["bin_days"])
        a = np.zeros((3*m, len(t)))
        for i, panel in enumerate(prep.PANELS):
            ids, use = mapping[panel]
            a[i*m+ids[use], np.flatnonzero(use)] = 1/sector["relative_error"][use]**2
        raw_weights.append(a)
        post = conditional_posterior(sector["fit_values"], sector["fit_mask"], sector["stellar_psd"],
            sector["white_variance"], windows["bin_days"], options.fgp_cutoff_per_day)
        scale = sector["scale"]/sector["center"]
        # Independent dense posterior covariance at native samples.
        factor = post.flux_factor*scale
        covariances.append(factor @ factor.T)
        means.append(np.where(obs, sector["cleaned_flux"], 0.)-scale*post.flux_mean)
    weights = np.hstack(raw_weights)
    a = weights/weights.sum(axis=1)[:, None]
    expected_cov = a @ block_diag(*covariances) @ a.T
    np.testing.assert_allclose(data["factor"]@data["factor"].T, expected_cov, atol=1e-24, rtol=2e-11)
    np.testing.assert_allclose(data["flux"], a@np.concatenate(means), atol=2e-15, rtol=0)
    assert np.max(abs(expected_cov[:m, m:2*m])) > 1e-15
    np.testing.assert_allclose(data["map_flux"], np.concatenate([windows["flux_"+p] for p in prep.PANELS]),
                               atol=2e-15, rtol=0)
    for panel in prep.PANELS:
        np.testing.assert_allclose(data[panel+"_sigma"], np.median(windows["err_"+panel]), rtol=1e-14)


def test_full_orbit_covariance_matches_dense_saved_psd(monkeypatch):
    saved, _, _ = prepared(monkeypatch)
    folded = candidate_folded_fourier(saved)
    sector = saved.output["segments"][0]
    obs, length = sector["observed"], len(sector["time"])
    scale = sector["scale"]/sector["center"]
    dc = (sector["stellar_psd"][0]+length*sector["white_variance"])*scale**2
    power = total_fft_power(sector["stellar_psd"], sector["white_variance"], length, scale, dc_power=dc)
    covariance = observed_covariance(power, length, np.flatnonzero(obs))
    a = csr_matrix((folded.weights, (folded.bin_index, np.arange(obs.sum()))),
                   shape=(len(folded.block.time), obs.sum()))
    expected = (a @ covariance) @ a.T
    np.testing.assert_allclose(folded.block.precision @ expected, np.eye(len(expected)), atol=3e-13, rtol=0)
    np.testing.assert_allclose(folded.block.flux,
        1+a@(sector["raw_flux"][obs]/sector["center"]-1), atol=2e-16, rtol=0)
    np.testing.assert_array_equal(folded.native_time, sector["time"][obs])
    assert obs.sum() > sum(saved.output["stream"]["protected"])


@pytest.mark.parametrize("domain", ["real", "fourier"])
def test_one_command_runs_sampler_and_matches_python_api(monkeypatch, tmp_path, capsys, domain):
    saved, _, _ = prepared(monkeypatch)
    path, config = configuration(tmp_path, saved)
    # Relative input paths follow the config even from another working directory.
    elsewhere = tmp_path/"work"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["evidence", str(path), "--likelihood", domain, "--output", "cli.npz", "--N", "8"]) == 0
    result = RunResult.load(elsewhere/"cli.npz")
    assert result.output.attrs["N"] == 8
    assert result.output.attrs["scenario_scope"] == "conditional_subset"
    assert len(result.metadata["scenario_records"]) == 3
    assert sum(r["kind"] == "sampler" for r in result.sampling) == 3
    assert all(len(r["weights"]) for r in result.sampling if r["kind"] == "sampler")
    options = dict(config["settings"], N=8)
    prepare = candidate_folded_real if domain == "real" else candidate_folded_fourier
    calculate = calc_probs_folded_real if domain == "real" else calc_probs_folded_fourier
    reference = calculate(target(), prepare(saved), trilegal_fname=str(tmp_path/"population.csv"), **options)
    np.testing.assert_array_equal(result.output.lnBF, reference.output.lnBF)
    assert result.metadata["prepared_run"]["likelihood"] == domain
    assert all(len(r["sha256"]) == 64 for r in result.metadata["prepared_run"]["files"])
    assert '"FPP"' in capsys.readouterr().out
    monkeypatch.setattr("pentaceratops.prepared.hz.calc_probs_folded_real", lambda *a, **k: pytest.fail("No overwrite"))
    with pytest.raises(FileExistsError):
        run_prepared(path, output_path=elsewhere/"cli.npz")


@pytest.mark.parametrize("damage", ["mask", "frame", "ephemeris", "fold", "duplicate"])
def test_preparation_mismatch_is_rejected(monkeypatch, damage):
    saved, _, _ = prepared(monkeypatch)
    if damage == "mask":
        saved.output["segments"][0]["fit_mask"][:] = True
    elif damage == "frame":
        saved.output["windows"]["flux_frame"] = "unknown"
    elif damage == "ephemeris":
        saved.output["windows"]["period"] += .1
    elif damage == "fold":
        saved.output["windows"]["flux_even"][0] -= .1
    else:
        saved.output["segments"].append(saved.output["segments"][0])
    with pytest.raises(ValueError):
        candidate_folded_real(saved)


@pytest.mark.parametrize("damage", ["unknown", "target", "missing", "settings", "wrong_domain", "primary"])
def test_bad_config_fails_before_covariance_or_sampler(monkeypatch, tmp_path, damage):
    saved, _, _ = prepared(monkeypatch)
    path, config = configuration(tmp_path, saved)
    if damage == "unknown":
        config["typo"] = True
    elif damage == "target":
        config["target_id"] = 999
    elif damage == "missing":
        config["trilegal"] = "absent.csv"
    elif damage == "settings":
        config["settings"]["N"] = 0
    elif damage == "wrong_domain":
        config["input"] = dict(format="folded_fourier", directory="missing")
    else:
        config["settings"]["primary_only"] = True
    path.write_text(json.dumps(config))
    monkeypatch.setattr("pentaceratops.preprocessing.evidence_inputs.candidate_folded_real",
                        lambda *a, **k: pytest.fail("Fail before covariance building"))
    with pytest.raises((ValueError, FileNotFoundError)):
        run_prepared(path)


def test_load_existing_hz_real_and_fourier_inputs(monkeypatch, tmp_path):
    saved, _, _ = prepared(monkeypatch)
    path, config = configuration(tmp_path, saved)
    expected = candidate_folded_real(saved)
    prepared_arrays, posterior_arrays = {}, {}
    for panel in prep.PANELS:
        for prefix, suffix in (("time", "time"), ("flux", "map_flux"), ("err", "input_error")):
            prepared_arrays[prefix+"_"+panel] = expected[panel+"_"+suffix]
        for suffix, field in (("time", "time"), ("map_flux", "map_flux"),
                              ("conditional_mean", "mean_flux"), ("input_error", "input_error")):
            posterior_arrays[panel+"_"+suffix] = expected[panel+"_"+field]
    n = len(expected["even_time"])
    for i, panel in enumerate(prep.PANELS):
        posterior_arrays[panel+"_covariance_factor"] = expected["factor"][i*n:(i+1)*n]
    prepared_arrays.update(time=expected["even_time"], exptime_days=expected["exptime"], period=expected["period"])
    posterior_arrays["primary_conditional_mean"] = expected["even_mean_flux"]
    np.savez(tmp_path/"reference.npz", **prepared_arrays)
    np.savez(tmp_path/"posterior.npz", **posterior_arrays)
    folded = candidate_folded_fourier(saved)
    folded.save(tmp_path/"fourier")
    def calculate(host, data, **options):
        if isinstance(data, dict):
            np.testing.assert_array_equal(data["factor"], expected["factor"])
            np.testing.assert_array_equal(data["flux"], expected["flux"])
        else:
            np.testing.assert_array_equal(data.block.precision, folded.block.precision)
        return RunResult(pd.DataFrame())
    monkeypatch.setattr("pentaceratops.prepared.hz.calc_probs_folded_real", calculate)
    monkeypatch.setattr("pentaceratops.prepared.hz.calc_probs_folded_fourier", calculate)
    for domain, inputs in (("real", dict(format="folded_real", prepared="reference.npz", posterior="posterior.npz")),
                            ("fourier", dict(format="folded_fourier", directory="fourier"))):
        config.update(likelihood=domain, input=inputs, output=domain+".npz")
        path.write_text(json.dumps(config))
        result = run_prepared(path)
        assert result.metadata["prepared_run"]["input_format"] == "folded_"+domain


@pytest.mark.parametrize("domain", ["real", "fourier"])
@pytest.mark.parametrize("settings,overrides,expected", [
    ({}, {}, {"TP": (100, 20), "EB": (200, 30), "NTP": (100, 20)}),
    ({"N": None, "steps": None}, {}, {"TP": (100, 20), "EB": (200, 30), "NTP": (100, 20)}),
    ({"N": 17, "steps": None}, {"steps": 4}, {"TP": (17, 4), "EB": (17, 4), "NTP": (17, 4)}),
])
def test_prepared_config_uses_shared_defaults(monkeypatch, tmp_path, domain, settings, overrides, expected):
    saved, _, _ = prepared(monkeypatch)
    path, config = configuration(tmp_path, saved)
    config["likelihood"] = domain
    config["settings"].pop("N")
    config["settings"].pop("steps")
    config["settings"].update(settings)
    path.write_text(json.dumps(config))
    calls = {}

    def sample(adapter, scenario, star, **options):
        calls[scenario] = (options["N"], options["steps"])
        return dict(lnBF=1., lnZ=2., null_loglike=1.)

    monkeypatch.setattr("pentaceratops.hz.sample_joint_scenario", sample)
    result = run_prepared(path, **overrides)
    assert calls == expected
    assert result.metadata["prepared_run"]["resolved_settings"]["N"] == settings.get("N")
    assert result.output.attrs["sampling_config"]["NTP"]["steps"] == expected["NTP"][1]
