"""Optimized full-orbit models for the joint, observed-data Fourier likelihood.

No eclipse-window split, interpolation, secondary non-detection veto, or
window-based timing rejection. Both parity orientations of 2P binaries use
exactly the same observed data and covariance. Fixed-period sampling only.
"""
import inspect
import time

import numpy as np
from pytransit import QuadraticModel

from ..likelihoods import real as lk
from ..likelihoods.joint_fourier import JointFourierMetric
from .covariance import _validate_nsamples
from .physical_batch import PhysicalBatch
from .sparse_kernels import orbit_setup, project_strict


class JointFourierAdapter:
    def __init__(self, blocks, period, epoch, *, parity="profile", nsamples=7, chunk_size=64,
                 mission="TESS", filt=None):
        if not np.all(np.isfinite([period, epoch])) or period <= 0:
            raise ValueError("Require finite epoch and positive fixed period")
        if parity not in ("profile", "marginalize"):
            raise ValueError("Unknown parity treatment")
        if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if mission not in ("TESS", "Kepler", "K2"):
            raise ValueError("mission must be TESS, Kepler, or K2")
        self.mission = mission
        self.filt = ("TESS" if mission == "TESS" else "Kepler") if filt is None else filt
        self.metric = JointFourierMetric(blocks)
        exposures = np.array([b.exptime for b in self.metric.blocks])
        if not np.allclose(exposures, exposures[0], atol=0, rtol=1e-12):
            raise ValueError("This adapter requires a common exposure duration")
        self.period, self.epoch = float(period), float(epoch)
        self.nsamples, self.chunk_size = _validate_nsamples(nsamples), int(chunk_size)
        self.parity, self.lk = parity, lk
        self.native_time = np.ascontiguousarray(self.metric.time - epoch)
        self.exptime = float(exposures[0])
        self.aperture_fraction = 1.
        self.record, self.snapshot = False, None
        self.calls = dict(planet=0, binary=0, x2p=0)
        self.physical, self.grids = {}, {}
        self.primary_model = QuadraticModel(interpolate=False)
        self.secondary_model = QuadraticModel(interpolate=False)
        for model in (self.primary_model, self.secondary_model):
            model.set_data(self.native_time, exptimes=self.exptime, nsamples=self.nsamples)
        for mult in (1, 2):
            phases = self.native_time % (mult * self.period)
            order = np.argsort(phases, kind="stable")
            self.grids[mult] = (np.ascontiguousarray(phases[order]),
                                np.ascontiguousarray(order))
        self.identity = np.arange(len(self.native_time), dtype=np.int64)
        self.weights = np.ones(len(self.native_time))
        self.reset_profile()

    def integration_samples(self, nsamples=None):
        value = self.nsamples if nsamples is None else _validate_nsamples(nsamples)
        if value != self.nsamples:
            raise ValueError("Exposure integration must match this adapter")
        return value

    def reset_profile(self):
        self.profile = dict(physical_setup_seconds=0., physics_seconds=0., exposure_seconds=0.,
                            covariance_seconds=0., native_evaluations=0, model_batches=0)

    def _check(self, kind, p):
        expected = self.period * (2 if kind == "x2p" else 1)
        if not np.allclose(p["P_orb"], expected, atol=0, rtol=1e-12):
            raise ValueError("Require P for planets/EBs and exactly 2P for alternating EBs")
        if "exptime" in p and not np.allclose(p["exptime"], self.exptime, atol=0, rtol=1e-12):
            raise ValueError("Exposure duration changed")
        if "nsamples" in p and not np.all(np.asarray(p["nsamples"]) == self.nsamples):
            raise ValueError("Exposure integration changed")
        if not np.isfinite(self.aperture_fraction) or not 0 < self.aperture_fraction <= 1:
            raise ValueError("Invalid aperture flux fraction")

    def _components(self, kind, p, reverse=False):
        """Orbital centers and light fractions shared by scalar/batched models."""
        comp = p["companion_fluxratio"] / (1-p["companion_fluxratio"])
        host = np.where(p["companion_is_host"], comp, 1.)
        if kind == "planet":
            return p["R_p"]*lk.Rearth/(p["R_s"]*lk.Rsun), 0., 0., host/(1+comp), 0.
        k = p["R_EB"] / p["R_s"]
        k = np.where(abs(k-1.) < 1e-6, k*.999, k)
        delta = lk.mean_anomaly_difference(p["ecc"], np.deg2rad(p["argp"])) * p["P_orb"]
        primary = -(delta-p["P_orb"]/2)/2 if kind == "x2p" else np.zeros_like(delta)
        if reverse:
            primary = primary+p["P_orb"]/2
        eb = p["EB_fluxratio"] / (1-p["EB_fluxratio"])
        return k, primary, primary+delta, host/(1+comp+eb), eb/(1+comp+eb)

    def native_model(self, kind, p, *, reverse=False):
        """Independent dense PyTransit reference at the actual observed times."""
        self._check(kind, p)
        k, primary, secondary, amp_primary, amp_secondary = self._components(kind, p, reverse)
        common = dict(ldc=[np.asarray(p["u1"]).item(), np.asarray(p["u2"]).item()], p=float(p["P_orb"]),
                      i=float(np.deg2rad(p["inc"])), e=float(p["ecc"]))
        flux = self.primary_model.evaluate_ps(k=float(k), t0=float(primary),
            a=float(p["a"]/(p["R_s"]*lk.Rsun)), w=float(np.deg2rad(90-p["argp"])), **common)
        signal = (flux-1)*amp_primary
        if kind != "planet":
            sec = self.secondary_model.evaluate_ps(k=float(1/k), t0=float(secondary),
                a=float(p["a"]/(k*p["R_s"]*lk.Rsun)),
                w=float(np.deg2rad(270-p["argp"])), **common)
            signal = signal+(sec-1)*amp_secondary
        return 1+self.aperture_fraction*signal

    def _scalar(self, kind, p):
        self.calls[kind] += 1
        model = self.native_model(kind, p)
        gain = self.metric.gain(model)
        extra = dict(phantom_loglike_ratio=0.)
        if kind == "x2p":
            reverse = self.native_model(kind, p, reverse=True)
            second = self.metric.gain(reverse)
            extra.update(reversed_parity=bool(second > gain), parity_loglike_ratios=[gain, second])
            if second > gain:
                model = reverse
            gain = max(gain, second) if self.parity == "profile" else np.logaddexp(gain, second)-np.log(2)
        if self.record:
            clean = {k: np.asarray(v).item() for k, v in p.items()}
            self.snapshot = dict(kind=kind, parameters=clean, model=model.copy(),
                photometric_loglike_ratio=float(gain), aperture_fraction=float(self.aperture_fraction), **extra)
        return -float(gain)

    def planet(self, time, flux, sigma, R_p, P_orb, inc, a, R_s, u1, u2, ecc, argp,
               companion_fluxratio=0., companion_is_host=False, exptime=.00139, nsamples=7):
        p = dict(R_p=R_p, P_orb=P_orb, inc=inc, a=a, R_s=R_s, u1=u1, u2=u2, ecc=ecc,
            argp=argp, companion_fluxratio=companion_fluxratio, companion_is_host=companion_is_host,
            exptime=exptime, nsamples=nsamples)
        return self._scalar("planet", p)

    def binary(self, time, flux, sigma, time_secondary, flux_secondary, sigma_secondary,
               R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp,
               companion_fluxratio=0., companion_is_host=False, exptime=.00139, nsamples=7):
        p = dict(R_EB=R_EB, EB_fluxratio=EB_fluxratio, P_orb=P_orb, inc=inc, a=a, R_s=R_s,
            u1=u1, u2=u2, ecc=ecc, argp=argp, companion_fluxratio=companion_fluxratio,
            companion_is_host=companion_is_host, exptime=exptime, nsamples=nsamples)
        return self._scalar("binary", p)

    def x2p(self, time, flux, sigma, time_secondary, flux_secondary, sigma_secondary,
               R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp,
               companion_fluxratio=0., companion_is_host=False, exptime=.00139, nsamples=7):
        p = dict(R_EB=R_EB, EB_fluxratio=EB_fluxratio, P_orb=P_orb, inc=inc, a=a, R_s=R_s,
            u1=u1, u2=u2, ecc=ecc, argp=argp, companion_fluxratio=companion_fluxratio,
            companion_is_host=companion_is_host, exptime=exptime, nsamples=nsamples)
        return self._scalar("x2p", p)

    def _project(self, kind, p, *, reverse=False):
        self._check(kind, p)
        start = time.perf_counter()
        n = len(p["P_orb"])
        k, tpri, tsec, apri, asec = self._components(kind, p, reverse)
        signal = np.zeros((n, len(self.native_time)))
        phase, order = self.grids[2 if kind == "x2p" else 1]
        for secondary in (False, True) if kind != "planet" else (False,):
            pars = np.empty((n, 10))
            pars[:,0] = 1/k if secondary else k
            pars[:,1] = tsec if secondary else tpri
            pars[:,2] = p["P_orb"]
            pars[:,3] = p["a"]/((k if secondary else 1)*p["R_s"]*lk.Rsun)
            pars[:,4] = np.deg2rad(p["inc"])
            pars[:,5] = p["ecc"]
            pars[:,6] = np.deg2rad((270 if secondary else 90)-p["argp"])
            pars[:,7], pars[:,8] = p["u1"], p["u2"]
            pars[:,9] = self.aperture_fraction*(asec if secondary else apri)
            coeff, windows = orbit_setup(pars)
            value, counts = project_strict(pars, coeff, windows, self.native_time, phase, order,
                self.identity, self.weights, len(self.native_time), self.exptime, self.nsamples)
            signal += value
            self.profile["native_evaluations"] += int(counts.sum())
        self.profile["exposure_seconds"] += time.perf_counter()-start
        return 1+signal

    def photometric_columns(self, kind, p):
        models = self._project(kind, p)
        start = time.perf_counter()
        scores = self.metric.gains(models)
        self.profile["covariance_seconds"] += time.perf_counter()-start
        if kind == "x2p":
            second = self.metric.gains(self._project(kind, p, reverse=True))
            scores = np.maximum(scores, second) if self.parity == "profile" else np.logaddexp(scores, second)-np.log(2)
        self.calls[kind] += len(scores)
        self.profile["model_batches"] += 1
        return scores

    def make_physics(self, scalar):
        if scalar not in self.physical:
            start = time.perf_counter()
            self.physical[scalar] = PhysicalBatch(scalar)
            self.profile["physical_setup_seconds"] += time.perf_counter()-start
        return self.physical[scalar]

    def likelihood_batch(self, scalar, theta):
        if self.record:
            raise RuntimeError("Best-fit replay must use the scalar reference")
        physical = self.make_physics(scalar)
        start = time.perf_counter()
        values, ids, columns = physical.evaluate(theta)
        self.profile["physics_seconds"] += time.perf_counter()-start
        for lo in range(0, len(ids), self.chunk_size):
            sl = slice(lo, lo+self.chunk_size)
            values[ids[sl]] += self.photometric_columns(physical.kind, {k:v[sl] for k,v in columns.items()})
        values[~np.isfinite(values)] = -np.inf
        return values

    def scenario_kwargs(self, function, star, *, trilegal, molusc=None, N=500, steps=50):
        """Use existing population/geometry recipes with the joint callbacks.

        Empty secondary arguments are API placeholders only. Every photometric
        callback scores self.metric, i.e. every observed sample, exactly once.
        """
        f = self.aperture_fraction
        t, flux, sigma = self.native_time, 1+(self.metric.flux-1)/f, self.metric.sigma/f
        values = dict(time=t, flux=flux, sigma=sigma,
            time_even=t, flux_even=flux, sigma_even=sigma,
            time_odd=np.empty(0), flux_odd=np.empty(0), sigma_odd=np.empty(0),
            time_secondary=np.empty(0), flux_secondary=np.empty(0), sigma_secondary=np.empty(0),
            P_orb=self.period, M_s=float(star["mass"]), R_s=float(star["rad"]),
            Teff=float(star["Teff"]), Z=0., plx=float(star["plx"]),
            Tmag=float(star["Tmag"]), Jmag=float(star["Jmag"]), Hmag=float(star["Hmag"]), Kmag=float(star["Kmag"]),
            trilegal_fname=str(trilegal), contrast_curve_file=None, filt=self.filt, N=N, steps=steps,
            mission=self.mission, flatpriors=False, exptime=self.exptime, nsamples=self.nsamples,
            molusc_file=str(molusc) if molusc else None, secondary_loglike=self.binary)
        signature = inspect.signature(function)
        missing = [k for k,p in signature.parameters.items() if p.default is inspect.Parameter.empty and k not in values]
        if missing:
            raise ValueError(f"Missing scenario arguments: {missing}")
        return {k:values[k] for k in signature.parameters if k in values}
