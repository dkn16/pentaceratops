"""Direct evidence(prepared, target=...) without config files or implicit saving."""
import inspect
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pentaceratops import RunResult, calc_probs_folded_real, calc_probs_folded_fourier
from pentaceratops.evidence import evidence
from pentaceratops.preprocessing.evidence_inputs import candidate_folded_real, candidate_folded_fourier
from test_prepared_command import prepared
from test_hz import target


def field(tmp_path):
    host = target()
    host.trilegal_fname = tmp_path/"population.csv"
    host.trilegal_fname.write_text("fixture; selected TP/EB/NTP do not use this population\n")
    # The target remains first even when a caller retained a custom table index.
    host.stars.index = [7, 12]
    return host


@pytest.mark.parametrize("domain", ["real", "fourier"])
@pytest.mark.parametrize("input_form", ["object", "path", "folded"])
def test_direct_evidence_matches_existing_api_without_json(monkeypatch, tmp_path, domain, input_form):
    saved, _, _ = prepared(monkeypatch)
    host = field(tmp_path)
    prepare = candidate_folded_real if domain == "real" else candidate_folded_fourier
    data = prepare(saved)
    if input_form == "object":
        argument = saved
    elif input_form == "path":
        argument = saved.save(tmp_path/"candidate.npz")
    else:
        argument = data
    before = set(tmp_path.iterdir())
    original_stars = host.stars.copy(deep=True)
    options = dict(N=8, steps=1, nsamples=3, seed=307, posterior_samples=9,
                    eb_eta=.1, scenarios=["TP", "EB", "NTP"])
    results = evidence(argument, target=host, likelihood=domain, **options)
    assert isinstance(results, RunResult)
    assert set(tmp_path.iterdir()) == before  # No output/config/export was required.
    assert len(results.output) == 3 and np.isfinite(results.output.lnBF).all()
    assert sum(r["kind"] == "sampler" for r in results.sampling) == 3
    assert results.output.attrs["scenario_scope"] == "conditional_subset"
    calculate = calc_probs_folded_real if domain == "real" else calc_probs_folded_fourier
    expected = calculate(host, data, **options)
    np.testing.assert_array_equal(results.output.lnBF, expected.output.lnBF)
    pd.testing.assert_frame_equal(host.stars, original_stars)
    info = results.metadata["evidence_call"]
    assert info["resolved_settings"]["N"] == 8 and info["resolved_settings"]["eb_eta"] == .1
    if input_form == "object":
        info["preparation_metadata"]["settings"]["candidate"]["host_id"] = 999
        assert saved.metadata["settings"]["candidate"]["host_id"] == 123


def test_optional_output_directory_input_and_real_options(monkeypatch, tmp_path):
    saved, _, _ = prepared(monkeypatch)
    host = field(tmp_path)
    folded = candidate_folded_fourier(saved)
    folded.save(tmp_path/"folded")
    calls = []
    def calculate(target, data, **options):
        calls.append(options)
        if isinstance(data, dict):
            assert options["include_gp"] is False
            assert options["primary_only"] is False
            assert options["timing_policy"] == "legacy"
        else:
            np.testing.assert_array_equal(data.block.precision, folded.block.precision)
            assert options["nsamples"] == 7
        return RunResult(pd.DataFrame({"lnBF": [1.]}))
    monkeypatch.setattr("pentaceratops.prepared.hz.calc_probs_folded_fourier", calculate)
    monkeypatch.setattr("pentaceratops.prepared.hz.calc_probs_folded_real", calculate)
    destination = tmp_path/"result.npz"
    evidence(tmp_path/"folded", target=host, likelihood="fourier", output_path=destination)
    result = RunResult.load(destination)
    assert result.metadata["evidence_call"]["input_format"] == "folded_fourier"
    evidence(candidate_folded_real(saved), target=host, include_gp=False, timing_policy="legacy")
    assert len(calls) == 2 and calls[1]["nsamples"] == 20
    with pytest.raises(FileExistsError):
        evidence(saved, target=host, output_path=destination)
    assert len(calls) == 2


@pytest.mark.parametrize("options", [dict(N=0), dict(steps=0), dict(nsamples=0),
    dict(likelihood="fourier", include_gp=False), dict(likelihood="fourier", timing_policy="legacy"),
    dict(primary_only=True), dict(unknown_option=True)])
def test_invalid_direct_options_fail_before_covariance(monkeypatch, tmp_path, options):
    saved, _, _ = prepared(monkeypatch)
    host = field(tmp_path)
    monkeypatch.setattr("pentaceratops.preprocessing.evidence_inputs.candidate_folded_real",
                        lambda *a, **k: pytest.fail("No covariance for invalid options"))
    with pytest.raises((ValueError, TypeError)):
        evidence(saved, target=host, **options)


def test_wrong_target_or_input_type_is_rejected(monkeypatch, tmp_path):
    saved, _, _ = prepared(monkeypatch)
    host = field(tmp_path)
    host.ID = 999
    with pytest.raises(ValueError, match="target"):
        evidence(saved, target=host)
    host.ID = 123
    with pytest.raises(TypeError, match="prepared"):
        evidence((saved, "a-path"), target=host)
    with pytest.raises(TypeError, match="matching"):
        evidence(candidate_folded_real(saved), target=host, likelihood="fourier")


@pytest.mark.parametrize("first_import", ["import pentaceratops.hz", "from pentaceratops.evidence import evidence"])
def test_evidence_function_and_existing_submodules_coexist(first_import):
    code = first_import + "\n" + """
import inspect
from pentaceratops.evidence import evidence, real, fourier, eclipses
from pentaceratops import evidence as evidence_package
assert inspect.isfunction(evidence)
assert inspect.ismodule(evidence_package)
assert evidence_package.evidence is evidence
assert 'prepared' in inspect.signature(evidence).parameters
assert 'target' in inspect.signature(evidence).parameters
assert callable(real.lnZ_TTP) and callable(fourier.lnZ_TTP_fourier)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
