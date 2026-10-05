"""Recorded covariance-only updates of the original folded Kepler recipes."""
import importlib
import inspect
from numbers import Integral
from types import SimpleNamespace

import numpy as np

from ..api import run_evidence
from ..experimental.folded_baseline import FoldedBaselineObservation
from ..likelihoods import real as lk
from ..likelihoods.joint_fourier import GaussianBlock, JointFourierMetric
from ..results import RunResult


def run_folded_baseline(function, *, block, fold_indices, fold_length, fold_roll,
                        period, epoch, aperture_fraction, seed=42,
                        posterior_samples=2000, output_path=None, **kwargs):
    """Retain an original Fourier recipe while changing data and covariance.

    ``function`` is a single-output evidence.fourier/fourier_eclipses recipe.
    ``kwargs`` are its original physical/data-grid and sampler arguments,
    including the original max_shift. The supplied Gaussian block contains the
    actual selected fold and any *explicitly* marginalized original FFT modes.
    fold_indices maps its observations to the full P or 2P model grid after
    applying the original fold_roll (P) or parity concatenation (2P).

    This preserves the original scalar sampler, priors, half-grid renderers,
    timing limits and maximum-over-parity convention used for KOI-2719.02.
    It does not replace them with the newer native-exposure Fourier renderer.
    Weighted pools and scalar best-fit replay are saved in a RunResult.
    """
    if output_path is not None:
        RunResult.check_destination(output_path)
    if not isinstance(block, GaussianBlock):
        raise TypeError("Require a prepared GaussianBlock")
    if (isinstance(fold_length, (bool, np.bool_)) or not isinstance(fold_length, Integral)
            or fold_length < 2 or fold_length % 2
            or isinstance(fold_roll, (bool, np.bool_)) or not isinstance(fold_roll, Integral)):
        raise ValueError("Require an even positive fold_length and integer fold_roll")
    indices = np.asarray(fold_indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or len(indices) != len(block.time) or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= fold_length)):
        raise ValueError("Invalid retained indices on the original folded model grid")
    if (not np.isfinite([period, epoch, aperture_fraction]).all() or period <= 0
            or not 0 < aperture_fraction <= 1):
        raise ValueError("Invalid period, epoch or aperture flux fraction")
    if (isinstance(posterior_samples, (bool, np.bool_))
            or not isinstance(posterior_samples, Integral)
            or posterior_samples < 1):
        raise ValueError("posterior_samples must be a positive integer")
    module = importlib.import_module(function.__module__)
    if module.__name__ not in ("pentaceratops.evidence.fourier",
                              "pentaceratops.evidence.fourier_eclipses"):
        raise ValueError("Require a Pentaceratops historical Fourier evidence recipe")
    bound = inspect.signature(function).bind(**kwargs)
    bound.apply_defaults()
    adapter = SimpleNamespace(
        metric=JointFourierMetric([block]), period=float(period), epoch=float(epoch),
        exptime=bound.arguments["exptime"], nsamples=bound.arguments["nsamples"],
        fold_indices=indices, fold_length=fold_length, fold_roll=fold_roll,
        aperture_fraction=float(aperture_fraction), record=False, snapshot=None, lk=lk,
    )
    if not np.isclose(adapter.exptime, block.exptime, rtol=1e-12, atol=0):
        raise ValueError("Exposure duration differs from the prepared block")

    def calculate():
        original = module._run_persistent_evidence
        count = module.POSTERIOR_NSAMPLES
        result = {}
        def sample(observation, prior, **options):
            if result:
                raise ValueError("Use one physical branch per folded baseline call")
            bridge = FoldedBaselineObservation(observation, adapter)
            evidence, pool = original(bridge, prior, **options)
            result.update(lnBF=float(evidence), physical_parameters=np.asarray(pool.samples).copy(),
                          weights=np.asarray(pool.weights).copy(),
                          log_target=np.asarray(pool.log_likelihoods).copy())
            if np.isfinite(evidence):
                best = int(np.argmax(pool.log_likelihoods))
                adapter.record = True
                bridge.reference = True
                try:
                    replay = float(bridge(pool.samples[best]))
                finally:
                    adapter.record = False
                    bridge.reference = False
                if not np.isclose(replay, pool.log_likelihoods[best], atol=2e-5, rtol=2e-12):
                    raise RuntimeError("Historical scalar best-fit replay changed")
                result["bestfit"] = dict(adapter.snapshot, theta=np.asarray(pool.samples[best]).copy(),
                                         scalar_replay=replay, log_target=float(pool.log_likelihoods[best]))
            return evidence, pool
        module._run_persistent_evidence = sample
        module.POSTERIOR_NSAMPLES = int(posterior_samples)
        try:
            physical = function(**kwargs)
        finally:
            module._run_persistent_evidence = original
            module.POSTERIOR_NSAMPLES = count
        if not result:
            if not isinstance(physical, dict) or physical.get("lnZ") != -np.inf:
                raise RuntimeError("Unexpected historical sampler bypass")
            result.update(lnBF=-np.inf, empty_support=True)
        elif not np.isfinite(result["lnBF"]):
            raise RuntimeError("Nonfinite baseline evidence without explicit empty support")
        result.update(lnZ=result["lnBF"]+block.null_loglike, null_loglike=block.null_loglike,
                      physical_result=physical, parity="profile",
                      backend="original_fourier_models_priors_scalar_sampler_folded_covariance")
        return result

    return run_evidence(calculate, seed=seed, output_path=output_path, metadata=dict(
        likelihood="historical_folded_fourier", timing_policy="legacy",
        recipe=function.__module__+"."+function.__name__,
        period=float(period), epoch=float(epoch), aperture_fraction=float(aperture_fraction),
        fold_indices=indices, fold_length=fold_length, fold_roll=fold_roll,
        exptime=adapter.exptime, nsamples=adapter.nsamples, N=bound.arguments["N"],
        steps=bound.arguments["steps"], max_shift=bound.arguments.get("max_shift"),
        observation_time=block.time, observation_flux=block.flux,
        posterior_samples=int(posterior_samples),
    ))
