"""TP compatibility facade over the composable host/system implementation."""

import numpy as np

from ..models.hosts import KnownHost
from ..models.systems import Planet, Scenario, RSUN, period_range, semimajor_axis
from .scenario import ScenarioPrior as TargetPlanetPrior
from .scenario import ScenarioLikelihood as TargetPlanetLikelihood


class TargetPlanet:
    """Construction shim retained for the existing real/Fourier TP wrappers."""

    @staticmethod
    def prepare(period, mass, radius, teff, metallicity, flatpriors, ldc):
        host = KnownHost.prepare(mass, radius, teff, metallicity, ldc)
        return Scenario(host, Planet(flatpriors), period_range(period))


def target_planet_result(model, lnz, samples, posterior_count, resample):
    """Keep historical resampling order, best-fit placement, and fallback schema."""
    try:
        draws = resample(samples.samples, samples.weights, log_likelihoods=samples.log_likelihoods)
        period, inc, ecc, argp, rp = (draws[:, i] for i in range(5))
        a = semimajor_axis(period, model.host.target.mass)
        r = a * (1 - ecc**2) / (1 + ecc * np.sin(argp * np.pi / 180))
        b = r * np.cos(inc * np.pi / 180) / (model.host.target.radius * RSUN)
        count = min(posterior_count, draws.shape[0])
        selected = slice(0, count)
        result = dict(
            M_s=np.full(count, model.host.target.mass),
            R_s=np.full(count, model.host.target.radius),
            u1=np.full(count, model.host.target.u1),
            u2=np.full(count, model.host.target.u2),
            P_orb=period[selected],
            inc=inc[selected],
            b=b[selected],
            R_p=rp[selected],
            ecc=ecc[selected],
            argp=argp[selected],
            M_EB=np.zeros(count),
            R_EB=np.zeros(count),
            fluxratio_EB=np.zeros(count),
            fluxratio_comp=np.zeros(count),
            lnZ=float(lnz),
        )
    except Exception:
        result = dict(
            M_s=np.array([model.host.target.mass]),
            R_s=np.array([model.host.target.radius]),
            u1=np.array([model.host.target.u1]),
            u2=np.array([model.host.target.u2]),
            P_orb=np.array([model.period_range[0]]),
            inc=np.array([90.0]),
            b=np.array([0.0]),
            R_p=np.array([1.0]),
            ecc=np.array([0.0]),
            argp=np.array([90.0]),
            M_EB=np.array([0.0]),
            R_EB=np.array([0.0]),
            fluxratio_EB=np.array([0.0]),
            fluxratio_comp=np.array([0.0]),
            lnZ=float(lnz),
        )
    result["best_lnL"] = float(np.max(samples.log_likelihoods))
    return result


def run_target_planet(
    model, observation, inverse_eccentricity, *, sampler, resample, n_active, steps, posterior_count
):
    """Single shared TP evidence pipeline; sampler hooks remain explicit."""
    prior = TargetPlanetPrior(model, inverse_eccentricity)
    lnz, samples = sampler(
        observation, prior, ndim=5, n_active=n_active, target_ess=2 * n_active, mcmc_steps=steps
    )
    return target_planet_result(model, lnz, samples, posterior_count, resample)
