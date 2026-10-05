"""Conditional trend-posterior propagation into a common-window likelihood.

This is a cut-posterior sensitivity model, not a joint raw-photometry fit.
The frozen trend posterior is integrated analytically, keeping all its
within/between-panel correlations. No production engine source is modified.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
import os
from numbers import Integral
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve

PANELS = ("even", "odd", "secondary")


def _validate_nsamples(value):
    """Number of integration points within one exposure, not sampler effort."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError("nsamples must be a positive integer (subsamples per exposure)")
    return int(value)


class GaussianMetric:
    """Log Gaussian likelihood relative to the same-data flat-flux null."""
    def __init__(self, flux, sigma, factor=None):
        self.flux = np.asarray(flux, float)
        sigma = np.asarray(sigma, float)
        if self.flux.ndim != 1 or sigma.shape != self.flux.shape:
            raise ValueError("Flux/error shapes differ")
        if not np.all(np.isfinite(self.flux)) or not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError("Require finite flux and positive errors")
        self.covariance = np.diag(sigma**2)
        if factor is not None:
            factor = np.asarray(factor, float)
            if factor.ndim != 2 or factor.shape[0] != len(sigma) or not np.all(np.isfinite(factor)):
                raise ValueError("Invalid trend covariance factor")
            self.covariance += factor @ factor.T
        chol = cho_factor(self.covariance, lower=True, check_finite=False)
        self.precision = cho_solve(chol, np.eye(len(sigma)), check_finite=False)
        self.precision = .5 * (self.precision + self.precision.T)
        self.residual = self.flux - 1
        self.projected_residual = self.precision @ self.residual
        self.logdet = float(2 * np.log(np.diag(chol[0])).sum())
        self.null_loglike = float(-.5 * (self.residual @ self.projected_residual
                                         + self.logdet + len(sigma) * np.log(2 * np.pi)))

    def gain(self, model):
        signal = np.asarray(model, float) - 1
        if signal.shape != self.flux.shape:
            raise ValueError("Model has wrong data ordering/shape")
        if not np.all(np.isfinite(signal)):
            return -np.inf
        return float(self.projected_residual @ signal - .5 * signal @ (self.precision @ signal))


def load_inputs(row, folded_root):
    path = Path(row["prepared_path"])
    folded_path = Path(row.get("folded_path") or Path(folded_root) / row["group"] / row["tag"] / "folded_posterior.npz")
    result = {}
    with np.load(path, allow_pickle=False) as original, np.load(folded_path, allow_pickle=False) as folded:
        factors = []
        for panel in PANELS:
            for key, source in (("time", "time"), ("map_flux", "map_flux"),
                                ("mean_flux", "conditional_mean"), ("input_error", "input_error")):
                result[f"{panel}_{key}"] = np.asarray(folded[f"{panel}_{source}"], float)
            for key, source in (("time", "time"), ("map_flux", "flux"), ("input_error", "err")):
                np.testing.assert_allclose(result[f"{panel}_{key}"], original[f"{source}_{panel}"],
                                           atol=1e-10, rtol=0, err_msg=f"{panel}: mismatched FGP/reference cache")
            factor = np.asarray(folded[f"{panel}_covariance_factor"], float)
            factors.append(factor)
            # Preserve the published benchmark's scalar median error in each
            # fold. Do not simultaneously change its measurement-noise model.
            result[f"{panel}_sigma"] = np.full(len(factor), np.median(result[f"{panel}_input_error"]))
        if len({f.shape[1] for f in factors}) != 1:
            raise ValueError("Panel coefficient columns do not share an ordering")
        result["factor"] = np.vstack(factors)
        result["exptime"] = float(original["exptime_days"])
        result["period"] = float(original["period"])
        result["primary_time"] = np.asarray(original["time"], float)
        result["primary_flux"] = np.asarray(folded["primary_conditional_mean"], float)
    result["flux"] = np.concatenate([result[p + "_mean_flux"] for p in PANELS])
    result["map_flux"] = np.concatenate([result[p + "_map_flux"] for p in PANELS])
    result["sigma"] = np.concatenate([result[p + "_sigma"] for p in PANELS])
    return result


class ModelAdapter:
    def __init__(self, data, include_gp=True, mean="conditional", parity="profile", *,
                 nsamples=20, timing_policy="legacy"):
        self.nsamples = _validate_nsamples(nsamples)
        if timing_policy not in ("observed", "legacy"):
            raise ValueError("timing_policy must be observed or legacy")
        self.timing_policy = timing_policy
        from ..likelihoods import real as lk
        from .._target import _centered_secondary_model_new
        self.lk = lk
        # Do not inherit stale ID-based grid caches from a previous target
        # executed in the same process-pool worker.
        lk._tm_cache["id_time"] = None
        lk._tm_sec_cache["id_time"] = None
        self.centered_secondary = _centered_secondary_model_new
        self.data = data
        self.parity = parity
        if parity not in ("profile", "marginalize"):
            raise ValueError(parity)
        flux = data["flux"] if mean == "conditional" else data["map_flux"]
        factor = data["factor"] if include_gp else None
        self.metric = GaussianMetric(flux, data["sigma"], factor)
        self.times = {p: np.array(data[p + "_time"], copy=True) for p in PANELS}
        self.slices = {}
        start = 0
        for p in PANELS:
            stop = start + len(self.times[p])
            self.slices[p] = slice(start, stop)
            start = stop
        self.eo_time = np.concatenate([self.times["even"], self.times["odd"]])
        self.order = np.argsort(self.eo_time, kind="stable")
        self.sorted_eo_time = self.eo_time[self.order]
        self.inverse_order = np.argsort(self.order)
        self.secondary_metric = GaussianMetric(np.ones(len(self.times["secondary"])),
            data["secondary_sigma"], None if factor is None else factor[self.slices["secondary"]])
        self.aperture_fraction = 1.
        self.record = False
        self.snapshot = None
        self.calls = dict(planet=0, binary=0, x2p=0)

    def integration_samples(self, nsamples=None):
        """Resolve an optional per-call override without changing adapter defaults."""
        return _validate_nsamples(self.nsamples if nsamples is None else nsamples)

    def aperture(self, flux):
        return 1 + self.aperture_fraction * (np.asarray(flux) - 1)

    def join_primary_secondary(self, primary, secondary):
        return np.concatenate([np.asarray(primary)[self.inverse_order], secondary])

    def save_snapshot(self, kind, parameters, model, gain, **extra):
        if not self.record:
            return
        clean = {}
        for k, v in parameters.items():
            a = np.asarray(v)
            clean[k] = a.item() if a.size == 1 else a.tolist()
        self.snapshot = dict(kind=kind, parameters=clean,
                             aperture_fraction=float(self.aperture_fraction),
                             photometric_loglike_ratio=float(gain), **extra)
        self.snapshot["model"] = np.asarray(model).copy()

    def planet(self, time, flux, sigma, R_p, P_orb, inc, a, R_s, u1, u2,
               ecc, argp, companion_fluxratio=0., companion_is_host=False,
               exptime=.00139, nsamples=None):
        nsamples = self.integration_samples(nsamples)
        self.calls["planet"] += 1
        parameters = dict(R_p=R_p, P_orb=P_orb, inc=inc, a=a, R_s=R_s,
                          u1=u1, u2=u2, ecc=ecc, argp=argp,
                          companion_fluxratio=companion_fluxratio, companion_is_host=companion_is_host,
                          exptime=exptime, nsamples=nsamples)
        pri = self.aperture(self.lk.simulate_TP_transit(self.sorted_eo_time, **parameters))
        model = self.join_primary_secondary(pri, np.ones(len(self.times["secondary"])))
        gain = self.metric.gain(model)
        self.save_snapshot("planet", parameters, model, gain)
        return -gain

    def binary(self, time, flux, sigma, time_secondary, flux_secondary, sigma_secondary,
               R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp,
               companion_fluxratio=0., companion_is_host=False, exptime=.00139, nsamples=None):
        nsamples = self.integration_samples(nsamples)
        self.calls["binary"] += 1
        parameters = dict(R_EB=R_EB, EB_fluxratio=EB_fluxratio, P_orb=P_orb, inc=inc, a=a,
                          R_s=R_s, u1=u1, u2=u2, ecc=ecc, argp=argp,
                          companion_fluxratio=companion_fluxratio, companion_is_host=companion_is_host,
                          exptime=exptime, nsamples=nsamples)
        ts = self.times["secondary"]
        offset = float((self.lk.mean_anomaly_difference(ecc, np.deg2rad(argp)) - .5) * P_orb)
        pri, sec = self.lk.simulate_EB_transit_secondary(self.sorted_eo_time, ts, **parameters)
        outside = not ts.min() <= offset <= ts.max()
        phantom_gain = 0.
        if outside:
            # Preserve the established conservative no-secondary-elsewhere
            # assumption, as an independent same-length synthetic constraint.
            sec = np.ones_like(ts)
            phantom = self.aperture(self.centered_secondary(ts, **parameters))
            phantom_gain = self.secondary_metric.gain(phantom)
        model = self.join_primary_secondary(self.aperture(pri), self.aperture(sec))
        gain = self.metric.gain(model) + phantom_gain
        self.save_snapshot("binary", parameters, model, gain, secondary_offset_days=offset,
                           secondary_outside=outside, phantom_loglike_ratio=float(phantom_gain))
        return -gain

    def x2p(self, time, flux, sigma, time_secondary, flux_secondary, sigma_secondary,
            R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp,
            companion_fluxratio=0., companion_is_host=False, exptime=.00139, nsamples=None):
        nsamples = self.integration_samples(nsamples)
        self.calls["x2p"] += 1
        parameters = dict(R_EB=R_EB, EB_fluxratio=EB_fluxratio, P_orb=P_orb, inc=inc, a=a,
                          R_s=R_s, u1=u1, u2=u2, ecc=ecc, argp=argp,
                          companion_fluxratio=companion_fluxratio, companion_is_host=companion_is_host,
                          exptime=exptime, nsamples=nsamples)
        offset = float((self.lk.mean_anomaly_difference(ecc, np.deg2rad(argp)) - .5) * P_orb)
        te, to = self.times["even"], self.times["odd"]
        if self.timing_policy == "legacy":
            half = max(np.max(np.abs(te)), np.max(np.abs(to)))
            allowed = abs(offset)/2 <= half
        else:
            from .window_support import alternating_overlap
            allowed = bool(alternating_overlap(te, to, parameters, exptime)[0])
        if not allowed:
            return np.inf
        e1, o2 = self.lk.simulate_EB_transit_evenodd(te, to, **parameters)
        o1, e2 = self.lk.simulate_EB_transit_evenodd(to, te, **parameters)
        secondary = np.ones(len(self.times["secondary"]))
        forward = np.concatenate([self.aperture(e1), self.aperture(o2), secondary])
        reverse = np.concatenate([self.aperture(e2), self.aperture(o1), secondary])
        gains = np.array([self.metric.gain(forward), self.metric.gain(reverse)])
        if self.parity == "profile":  # preserve the existing engine choice in both arms
            gain = float(np.max(gains))
        else:
            gain = float(np.logaddexp(*gains) - np.log(2))
        reverse_best = bool(gains[1] > gains[0])
        self.save_snapshot("x2p", parameters, reverse if reverse_best else forward, gain,
                           reversed_parity=reverse_best, parity_loglike_ratios=gains.tolist(),
                           secondary_offset_days=offset)
        return -gain


@contextmanager
def patched_engine(adapter):
    """Isolated workers only. Priors/samplers/source files are unchanged."""
    from ..evidence import real as mn
    from ..evidence import eclipses as me
    originals = {}
    for module in (mn, me):
        for name, replacement in (("lnL_TP", adapter.planet), ("lnL_EB_evenodd", adapter.x2p),
                                  ("_log_sigma_norm", lambda *args: -.5 * np.log(2 * np.pi))):
            if hasattr(module, name):
                originals[module, name] = getattr(module, name)
                setattr(module, name, replacement)
    try:
        yield
    finally:
        for (module, name), original in originals.items():
            setattr(module, name, original)


def scenario_functions():
    from ..evidence import real as mn
    from ..evidence import eclipses as me
    result = {}
    for prefix, stem in (("", "T"), ("P", "P"), ("S", "S"), ("D", "D"), ("B", "B")):
        result[prefix + "TP"] = getattr(mn, f"lnZ_{stem}TP")
        result[prefix + "EB"] = getattr(me, f"lnZ_{stem}EB_secondary")
        result[prefix + "EBx2P"] = getattr(me, f"lnZ_{stem}EB_evenodd")
    return result


def scenario_kwargs(function, adapter, star, period, trilegal, molusc, n, steps, posterior_n,
                    *, nsamples=None):
    integration_samples = adapter.integration_samples(nsamples)
    from ..evidence import real as mn
    from ..evidence import eclipses as me
    mn.POSTERIOR_NSAMPLES = me.POSTERIOR_NSAMPLES = posterior_n
    d = adapter.data
    f = adapter.aperture_fraction
    host_flux = lambda x: 1 + (np.asarray(x) - 1) / f
    primary_flux = np.concatenate([d["even_mean_flux"], d["odd_mean_flux"]])[adapter.order]
    primary_sigma = np.concatenate([d["even_sigma"], d["odd_sigma"]])[adapter.order]
    values = dict(time=adapter.sorted_eo_time, flux=host_flux(primary_flux), sigma=primary_sigma / f,
                  time_secondary=adapter.times["secondary"], flux_secondary=host_flux(d["secondary_mean_flux"]),
                  sigma_secondary=d["secondary_sigma"] / f,
                  time_even=adapter.times["even"], flux_even=host_flux(d["even_mean_flux"]), sigma_even=d["even_sigma"] / f,
                  time_odd=adapter.times["odd"], flux_odd=host_flux(d["odd_mean_flux"]), sigma_odd=d["odd_sigma"] / f,
                  P_orb=float(period), M_s=float(star["mass"]), R_s=float(star["rad"]), Teff=float(star["Teff"]),
                  Z=0., plx=float(star["plx"]), Tmag=float(star["Tmag"]), Jmag=float(star["Jmag"]),
                  Hmag=float(star["Hmag"]), Kmag=float(star["Kmag"]), trilegal_fname=str(trilegal),
                  contrast_curve_file=None, filt="TESS", N=n, steps=steps, mission="TESS", flatpriors=False,
                  exptime=d["exptime"], nsamples=integration_samples, molusc_file=str(molusc) if molusc else None,
                  secondary_loglike=adapter.binary)
    signature = inspect.signature(function)
    missing = [name for name, p in signature.parameters.items()
               if p.default is inspect.Parameter.empty and name not in values]
    if missing:
        raise ValueError(f"Missing {function.__name__} arguments: {missing}")
    return {name: values[name] for name in signature.parameters if name in values}
