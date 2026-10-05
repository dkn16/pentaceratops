"""Recorded HZ inference on explicitly prepared, aperture-frame observations.

These entry points integrate the production adapters. They do not download data,
select transits, estimate a PSD, or replace the explicitly selected archive-reproduction recipes.
All likelihoods compare models in the same aperture frame and against the same
data null; the demographic EB odds are applied once when reporting probabilities.
"""
from numbers import Integral

import numpy as np
import pandas as pd

from .api import run_evidence
from .evidence.joint_fourier import FAMILIES, sample_joint_scenario
from .experimental.covariance import PANELS, scenario_kwargs
from .experimental.v2_adapter import OptimizedAdapter
from .experimental.primary_v2 import PrimaryOnlyAdapter
from .results import RunResult, scenario_probabilities


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _mission(target):
    mission = getattr(target, "mission", None)
    if mission not in ("TESS", "Kepler", "K2"):
        raise ValueError("Set target.mission to TESS, Kepler, or K2 explicitly")
    return mission


class _RealArguments:
    def scenario_kwargs(self, function, star, *, trilegal, molusc=None, N=500, steps=50):
        from .evidence import real, eclipses
        counts = real.POSTERIOR_NSAMPLES, eclipses.POSTERIOR_NSAMPLES
        try:
            values = scenario_kwargs(function, self, star, self.data["period"], trilegal,
                                     molusc, N, steps, counts[0])
        finally:
            real.POSTERIOR_NSAMPLES, eclipses.POSTERIOR_NSAMPLES = counts
        values["mission"] = self.mission
        if "filt" in values:
            values["filt"] = self.filt
        return values


class FoldedRealAdapter(_RealArguments, OptimizedAdapter):
    """Production folded even/odd/secondary conditional-FGP likelihood."""
    backend = "folded_real_strict_batch"


class PrimaryFoldedRealAdapter(_RealArguments, PrimaryOnlyAdapter):
    """Observed primary parities only; no synthetic secondary constraint."""
    backend = "folded_real_primary_only_strict_batch"


def real_adapter(data, *, mission, filt=None, primary_only=False, include_gp=True,
                 parity="profile", nsamples=20, timing_policy="observed"):
    """Validate folded data ordering before constructing the optimized metric.

    ``factor`` columns must share one coefficient ordering across all panels.
    Measurement errors and the trend factor are separate inputs: the covariance
    is diag(sigma**2) + factor @ factor.T. Do not pre-add trend variances to sigma.
    """
    if mission not in ("TESS", "Kepler", "K2"):
        raise ValueError("mission must be TESS, Kepler, or K2")
    if not np.isfinite(data["period"]) or data["period"] <= 0:
        raise ValueError("Require a positive fixed period")
    if not np.isfinite(data["exptime"]) or data["exptime"] <= 0:
        raise ValueError("Require a positive exposure duration in days")
    for panel in PANELS:
        time, flux, sigma = (np.asarray(data[panel + suffix], float)
                             for suffix in ("_time", "_mean_flux", "_sigma"))
        empty_secondary = primary_only and panel == "secondary"
        if (time.ndim != 1 or flux.shape != time.shape or sigma.shape != time.shape
                or not np.isfinite(time).all() or not np.isfinite(flux).all()
                or not np.all(np.isfinite(sigma) & (sigma > 0))
                or (len(time) == 0) != empty_secondary):
            raise ValueError(f"Invalid {panel} panel; primary_only must match observed coverage")
    for field, suffix in (("flux", "_mean_flux"), ("sigma", "_sigma")):
        expected = np.concatenate([data[p + suffix] for p in PANELS])
        if not np.array_equal(np.asarray(data[field]), expected):
            raise ValueError(f"{field} must follow the even, odd, secondary ordering")
    cls = PrimaryFoldedRealAdapter if primary_only else FoldedRealAdapter
    adapter = cls(data, include_gp=include_gp, parity=parity, nsamples=nsamples,
                  timing_policy=timing_policy)
    adapter.mission = mission
    adapter.filt = ("TESS" if mission == "TESS" else "Kepler") if filt is None else filt
    return adapter


def _dispatch(target, adapter, *, trilegal_fname, molusc_file, N, steps, seed,
              posterior_samples, eb_eta, scenarios, missing_host_policy):
    N, steps = _positive_integer(N, "N"), _positive_integer(steps, "steps")
    posterior_samples = _positive_integer(posterior_samples, "posterior_samples")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(eb_eta) or eb_eta < 0:
        raise ValueError("eb_eta must be finite and nonnegative")
    if missing_host_policy not in ("error", "solar"):
        raise ValueError("missing_host_policy must be error or solar")
    stars = target.stars.copy(deep=True)
    if stars.empty or not stars.iloc[0].tdepth > 0:
        raise ValueError("The first star must be the eligible target, not a promoted neighbor")
    if stars.ID.duplicated().any():
        raise ValueError("Star IDs must be unique")
    population = trilegal_fname or getattr(target, "trilegal_fname", None)
    if population is None:
        raise ValueError("Supply the shared TRILEGAL population")
    selected = None if scenarios is None else tuple(scenarios)
    allowed = (*FAMILIES, "NTP", "NEB", "NEBx2P")
    if selected is not None and (not selected or len(set(selected)) != len(selected)
                                 or set(selected) - set(allowed)):
        raise ValueError("scenarios must contain unique supported scenario labels")
    rows, records, replacements, tasks = [], {}, [], []
    ordinal = 0
    for index, (_, original_star) in enumerate(stars.iterrows()):
        if not original_star.tdepth > 0:
            continue
        star = original_star.copy()
        for key, fallback in (("mass", 1.), ("rad", 1.), ("Teff", 5777.)):
            if not np.isfinite(star[key]) or star[key] <= 0:
                if index == 0 or missing_host_policy == "error":
                    raise ValueError(f"Invalid {key} for host {star.ID}; supply stellar inputs")
                replacements.append(dict(ID=int(star.ID), field=key, value=fallback))
                star[key] = fallback
        fraction = float(star.fluxratio)
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError(f"Invalid aperture flux fraction for host {star.ID}")
        for scenario in FAMILIES if index == 0 else ("NTP", "NEB", "NEBx2P"):
            scenario_seed = int(seed) + 1009 * ordinal
            ordinal += 1  # Subsetting must not change the seeds of retained scenarios.
            if selected is not None and scenario not in selected:
                continue
            if scenario_seed >= 2**32:
                raise ValueError("Scenario seeds must fit the inherited 32-bit RNG")
            tasks.append((star, scenario, scenario_seed, molusc_file if index == 0 else None))
    if not tasks:
        raise ValueError("No eligible hosts for the requested scenarios")
    # Check every eligible host before the first potentially expensive fit.
    for star, scenario, scenario_seed, molusc in tasks:
        adapter.aperture_fraction = float(star.fluxratio)
        result = sample_joint_scenario(
            adapter, scenario, star, trilegal=population, molusc=molusc,
            N=N, steps=steps, seed=scenario_seed, posterior_samples=posterior_samples,
        )
        if not np.isfinite(result["lnBF"]) and not result.get("empty_support", False):
            raise RuntimeError(f"Unfinished/nonfinite evidence: {star.ID}/{scenario}")
        records[f"{int(star.ID)}:{scenario}"] = result
        rows.append(dict(ID=int(star.ID), scenario=scenario, lnBF=result["lnBF"],
                         lnZ=result["lnZ"], null_loglike=result["null_loglike"],
                         aperture_fraction=adapter.aperture_fraction, seed=scenario_seed))
    table = scenario_probabilities(pd.DataFrame(rows), evidence_column="lnBF", eb_eta=eb_eta)
    table.attrs.update(posterior_records=records, backend=adapter.backend,
                       scenario_scope="all" if selected is None else "conditional_subset",
                       missing_host_policy=missing_host_policy, stellar_replacements=replacements,
                       N=N, steps=steps, nsamples=adapter.nsamples, parity=adapter.parity,
                       posterior_samples=posterior_samples, mission=adapter.mission,
                       timing_policy=getattr(adapter, "timing_policy", "observed"),
                       flux_frame="aperture", evidence_frame="same_data_null_log_bayes_factor")
    return table


def _run(target, adapter, *, output_path, seed, metadata, **options):
    records = {}
    metadata.update(stars=target.stars.copy(deep=True), filt=adapter.filt,
                    trilegal_path=str(options["trilegal_fname"] or getattr(target, "trilegal_fname", "")),
                    molusc_path=None if options["molusc_file"] is None else str(options["molusc_file"]))
    def calculate():
        table = _dispatch(target, adapter, seed=seed, **options)
        # Keep array-valued records outside DataFrame.attrs: pandas compares
        # attrs during slicing/concatenation, where ndarray equality is ambiguous.
        records.update(table.attrs.pop("posterior_records"))
        return table
    result = run_evidence(calculate, seed=seed, metadata=metadata)
    result.metadata["scenario_records"] = records
    if output_path is not None:
        result.save(output_path)
    return result


def calc_probs_folded_real(target, data, *, trilegal_fname=None, molusc_file=None,
                           N=500, steps=50, nsamples=20, seed=42, posterior_samples=2000,
                           parity="profile", eb_eta=1., primary_only=False, include_gp=True,
                           filt=None, scenarios=None, missing_host_policy="error", output_path=None,
                           timing_policy="observed"):
    """Return a RunResult for folded Real data using the optimized HZ likelihood.

    Inputs are the dictionaries returned by load_folded_real. The loader retains the
    HZ per-panel median measurement errors and cross-panel trend covariance.
    Set primary_only explicitly when no secondary observations exist.
    The default derives x2P support from window/exposure boundaries and model
    contact times. timing_policy="legacy" restores the archived x2P center-only
    gate. Both policies preserve the existing ordinary-EB flat-secondary penalty.
    With scenarios supplied, the returned FPP is conditional on that subset.
    """
    if output_path is not None:
        RunResult.check_destination(output_path)
    adapter = real_adapter(data, mission=_mission(target), filt=filt,
                           primary_only=primary_only, include_gp=include_gp,
                           parity=parity, nsamples=nsamples, timing_policy=timing_policy)
    metadata = dict(likelihood="folded_real", period=float(data["period"]),
                    exptime=float(data["exptime"]), primary_only=bool(primary_only),
                    include_gp=bool(include_gp), time_convention="panel offsets in days",
                    timing_policy=timing_policy,
                    observation_time=np.concatenate([data[p + "_time"] for p in PANELS]),
                    observation_flux=adapter.metric.flux.copy())
    return _run(target, adapter, output_path=output_path, seed=seed, metadata=metadata,
                trilegal_fname=trilegal_fname, molusc_file=molusc_file, N=N, steps=steps,
                posterior_samples=posterior_samples, eb_eta=eb_eta, scenarios=scenarios,
                missing_host_policy=missing_host_policy)


def calc_probs_folded_fourier(target, folded, *, trilegal_fname=None, molusc_file=None,
                              N=500, steps=50, nsamples=7, seed=42, posterior_samples=2000,
                              parity="profile", eb_eta=1., filt=None, scenarios=None,
                              missing_host_policy="error", output_path=None):
    """Return a RunResult for the full-orbit folded PSD likelihood.

    ``folded`` is FoldedFourierData, prepared once then optionally saved and
    memory-mapped. Physical predictions use every contributing native exposure.
    This is the latest full-orbit HZ convention; archive-reproduction recipes are
    available separately and are not silently substituted.
    """
    from .preprocessing.folded import FoldedFourierData
    if output_path is not None:
        RunResult.check_destination(output_path)
    if not isinstance(folded, FoldedFourierData):
        raise TypeError("Require FoldedFourierData with native times and folding weights")
    adapter = folded.adapter(mission=_mission(target), filt=filt, parity=parity, nsamples=nsamples)
    adapter.backend = "folded_fourier_native_exposure_strict_batch"
    metadata = dict(likelihood="folded_fourier", period=folded.period, epoch=folded.epoch,
                    timing_policy="observed", max_anomaly_shift=None,
                    exptime=folded.block.exptime, local_windows=False,
                    covariance="sum_s A_s C_s A_s.T", model="A m(native exposures)",
                    native_time=folded.native_time, bin_index=folded.bin_index,
                    weights=folded.weights, observation_time=folded.block.time,
                    observation_flux=folded.block.flux)
    return _run(target, adapter, output_path=output_path, seed=seed, metadata=metadata,
                trilegal_fname=trilegal_fname, molusc_file=molusc_file, N=N, steps=steps,
                posterior_samples=posterior_samples, eb_eta=eb_eta, scenarios=scenarios,
                missing_host_policy=missing_host_policy)
