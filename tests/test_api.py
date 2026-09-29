import importlib
import inspect
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def test_import_does_not_load_catalog_clients():
    code = """
import sys
import pentaceratops
from pentaceratops import Target, RunResult, run_evidence
import pentaceratops.companions
assert 'lightkurve' not in sys.modules
assert not any(name.startswith('astroquery') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_all_evidence_functions_instrumented_once():
    for name in ("real", "eclipses", "fourier", "fourier_eclipses"):
        module = importlib.import_module("pentaceratops.evidence."+name)
        functions = [func for key, func in vars(module).items()
                     if key.startswith("lnZ_") and inspect.isfunction(func)
                     and func.__module__ == module.__name__]
        assert functions
        for function in functions:
            assert hasattr(function, "__wrapped__")
            assert not hasattr(function.__wrapped__, "__wrapped__")


def test_target_returns_record_without_changing_legacy_sampling_options(monkeypatch):
    from pentaceratops import Target
    from pentaceratops._target import target
    from pentaceratops.evidence import real, eclipses
    counts = real.POSTERIOR_NSAMPLES, eclipses.POSTERIOR_NSAMPLES

    def fake(self, *args, **kwargs):
        assert "output_posteriors" not in kwargs  # No additional random draws.
        real.POSTERIOR_NSAMPLES = 55
        eclipses.POSTERIOR_NSAMPLES = 55
        self.probs = pd.DataFrame({"scenario": ["TP"], "prob": [1.]})
        self.FPP = self.NFPP = 0.
        for name in ("star_num", "u1", "u2", "fluxratio_EB", "fluxratio_comp"):
            setattr(self, name, np.array([1.]))

    monkeypatch.setattr(target, "calc_probs", fake)
    instance = Target.__new__(Target)  # Deliberately bypass network constructor.
    instance.ID = 1
    with pytest.warns(UserWarning, match="Compatibility"):
        result = instance.calc_probs(seed=4)
    assert result is instance.last_run
    assert result.output.attrs["FPP"] == 0.
    assert (real.POSTERIOR_NSAMPLES, eclipses.POSTERIOR_NSAMPLES) == counts


def test_cli_policy(capsys):
    from pentaceratops.cli import main
    assert main(["sampling-policy"]) == 0
    assert '"STP"' in capsys.readouterr().out


def test_existing_destination_rejected_before_computation(tmp_path):
    from pentaceratops import RunResult, run_evidence
    path = RunResult(1).save(tmp_path / "existing.npz")
    def must_not_run():
        pytest.fail("An existing output should stop before any sampling")
    with pytest.raises(FileExistsError):
        run_evidence(must_not_run, output_path=path)
    with pytest.raises(ValueError, match=".npz"):
        run_evidence(must_not_run, output_path=tmp_path / "bad.txt")


def test_recorded_dataframe_does_not_alias_callers_data():
    from pentaceratops import run_evidence
    original = pd.DataFrame({"a": [1.]})
    result = run_evidence(lambda: original)
    original.loc[0, "a"] = 2.
    assert result.output.loc[0, "a"] == 1.


def test_callable_instances_and_partials_are_recordable():
    from functools import partial
    from pentaceratops import run_evidence
    class Callable:
        def __call__(self, value):
            return value + 1
    result = run_evidence(partial(Callable(), 3))
    assert result.output == 4
    assert "Callable" in result.metadata["function"]


def test_transit_dependency_versions_are_recorded(tmp_path):
    from importlib.metadata import version
    from pentaceratops import run_evidence, RunResult
    path = tmp_path / "versions.npz"
    run_evidence(lambda: 1., output_path=path)
    result = RunResult.load(path)
    for name in ("pytransit", "meepmeep", "numba"):
        assert result.metadata[name] == version(name)


def test_doctor_does_not_require_runtime_setuptools(monkeypatch, capsys):
    from pentaceratops import cli
    def installed(name):
        if name == "setuptools":
            raise cli.PackageNotFoundError(name)
        return "test-version"
    monkeypatch.setattr(cli, "version", installed)
    assert cli.main(["doctor"]) == 0
    assert '"setuptools": "not installed"' in capsys.readouterr().out
