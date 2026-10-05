"""Recorded original-grid Fourier inference with updated folded covariance."""
import importlib

import numpy as np
import pytest

from pentaceratops import run_folded_baseline, RunResult
from pentaceratops.likelihoods.joint_fourier import GaussianBlock
from test_all_scenarios import kwargs_for


@pytest.mark.parametrize("module_name,name", [
    ("fourier", "lnZ_TTP_fourier"),
    ("fourier_eclipses", "lnZ_TEB_secondary_fourier"),
    ("fourier_eclipses", "lnZ_TEB_evenodd_fourier"),
])
def test_original_grid_sampling_replay_and_records(tmp_path, module_name, name):
    module = importlib.import_module("pentaceratops.evidence."+module_name)
    function = getattr(module, name)
    kwargs = kwargs_for(function, "default", tmp_path)
    t = kwargs.get("time", kwargs.get("time_even"))
    if "time_secondary" in kwargs:
        kwargs.update(time_secondary=t.copy(), flux_secondary=np.ones(len(t)),
                      var_fourier_secondary=np.full(len(t)//2, .03))
    # Match the original two equal grids; retain a gapped, nonlocal subset.
    length = 2*len(t)
    ids = np.flatnonzero(np.arange(length)%7 != 0)
    time = np.arange(length)*kwargs["exptime"]
    flux = 1+1e-4*np.sin(time)
    covariance = .1**2*np.eye(len(ids)) + 1e-5*np.ones((len(ids), len(ids)))
    block = GaussianBlock.from_covariance(time[ids], flux[ids], covariance, kwargs["exptime"])
    sampler, count = module._run_persistent_evidence, module.POSTERIOR_NSAMPLES
    destination = tmp_path/"baseline.npz"
    result = run_folded_baseline(
        function, block=block, fold_indices=ids, fold_length=length, fold_roll=3,
        period=kwargs["P_orb"], epoch=.01, aperture_fraction=.17,
        posterior_samples=9, seed=39, output_path=destination, **kwargs,
    )
    assert module._run_persistent_evidence is sampler
    assert module.POSTERIOR_NSAMPLES == count
    saved = RunResult.load(destination)
    output = saved.output
    np.testing.assert_array_equal(output["physical_parameters"], result.output["physical_parameters"])
    assert output["lnZ"] == pytest.approx(output["lnBF"]+block.null_loglike)
    assert output["bestfit"]["scalar_replay"] == pytest.approx(output["bestfit"]["log_target"], abs=2e-5)
    assert output["bestfit"]["model"].shape == block.flux.shape
    assert output["parity"] == "profile"
    assert saved.metadata["max_shift"] == kwargs.get("max_shift")
    assert saved.metadata["posterior_samples"] == 9
    assert len(output["physical_result"]["M_s"]) == 9
    assert sum(r["kind"] == "sampler" for r in saved.sampling) == 1
    # This API deliberately retains the scalar historical sampling path.
    np.testing.assert_allclose(output["bestfit"]["aperture_fraction"], .17)


def test_invalid_original_grid_fails_before_sampling():
    from pentaceratops.evidence.fourier import lnZ_TTP_fourier
    block = GaussianBlock.from_covariance(np.arange(2.), np.ones(2), np.eye(2), .02)
    with pytest.raises(ValueError, match="fold_length"):
        run_folded_baseline(lnZ_TTP_fourier, block=block, fold_indices=np.arange(2),
                            fold_length=3, fold_roll=0, period=2., epoch=0., aperture_fraction=1.)
