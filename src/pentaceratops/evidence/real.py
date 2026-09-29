"""Scenario evidences computed with persistent sampling.

Extracted from the corrected research engine; numerical logic is unchanged.
N is the active particle count and steps is the MCMC effort.
See docs/architecture.md for migration limitations.
"""

import numpy as np
from pandas import read_csv
from astropy import constants
import os
from typing import Callable, Tuple, Optional


from ..sampling.persistent import persistent_sampling
from ..likelihoods.real import *
from ..priors import *
from ..stellar import (
    stellar_relations, flux_relation, nearest_ldc_coefficients
)

np.seterr(divide='ignore')

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi
ln2pi = np.log(2*pi)


def _log_sigma_norm(*sigmas):
    """Scalar noise normalization for scalar or per-point uncertainties."""
    values = np.concatenate([
        np.atleast_1d(np.asarray(sigma, dtype=float)).reshape(-1)
        for sigma in sigmas
    ])
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("flux uncertainties must be finite and positive")
    # Preserve the historical one-factor likelihood normalization while
    # allowing chi-square to use the exact uncertainty of every observation.
    return float(np.log(np.mean(values)))

# load TESS limb darkening coefficients
LDC_FILE = os.path.join(os.path.dirname(__file__), '../data/ldc_tess.csv')
ldc_T = read_csv(LDC_FILE)
ldc_T_Zs = np.array(ldc_T.Z, dtype=float)
ldc_T_Teffs = np.array(ldc_T.Teff, dtype=int)
ldc_T_loggs = np.array(ldc_T.logg, dtype=float)
ldc_T_u1s = np.array(ldc_T.aLSM, dtype=float)
ldc_T_u2s = np.array(ldc_T.bLSM, dtype=float)

# load Kepler limb darkening coefficients
LDC_FILE = os.path.join(os.path.dirname(__file__), '../data/ldc_kepler.csv')
ldc_K = read_csv(LDC_FILE)
ldc_K_Zs = np.array(ldc_K.Z, dtype=float)
ldc_K_Teffs = np.array(ldc_K.Teff, dtype=int)
ldc_K_loggs = np.array(ldc_K.logg, dtype=float)
ldc_K_u1s = np.array(ldc_K.a, dtype=float)
ldc_K_u2s = np.array(ldc_K.b, dtype=float)


# ------------------------------
# Nested sampling utilities
# ------------------------------
# Number of equal-weight posterior draws retained per scenario in the res
# dicts built by the lnZ_* functions. Default 100 leaves the best-fit
# behaviour unchanged; calc_probs raises it (via posterior_nsamples) when
# called with output_posteriors=True, so full posteriors can be exported.
POSTERIOR_NSAMPLES = 100


def _resample_equal(samples, weights, n_samples=None, log_likelihoods=None):
    """
    Resample samples with replacement according to weights.
    Similar to dynesty.utils.resample_equal.
    """
    if n_samples is None:
        n_samples = len(samples)
    indices = np.random.choice(len(samples), size=n_samples, p=weights)
    out = samples[indices]
    # Put the maximum-LIKELIHOOD (best-fit-to-data) sample first so the stored
    # best-fit parameters (res[...][0]) reproduce the observed light curve
    # rather than a random posterior draw. (Max-weight/posterior can sit at
    # extreme values in poorly constrained directions.)
    if (log_likelihoods is not None and len(out) > 0
            and len(log_likelihoods) > 0):
        out[0] = samples[int(np.argmax(log_likelihoods))]
    return out

def _run_persistent_evidence(
    loglike: Callable[[np.ndarray], float],
    prior_transform: Callable[[np.ndarray], np.ndarray],
    ndim: int,
    n_active: int = 100,
    target_ess: Optional[float] = None,
    mcmc_steps: int = 20,
) -> Tuple[float, object]:
    """
    Run persistent sampling to estimate ln(Z).
    Returns (lnZ, results_object).
    """
    if target_ess is None:
        target_ess = 2 * n_active

    log_Z, weights, positions, log_likelihoods = persistent_sampling(
        log_likelihood_function=loglike,
        prior_transform=prior_transform,
        n_active=n_active,
        ndim=ndim,
        target_ess=target_ess,
        mcmc_steps=mcmc_steps
    )
    
    class MockResults:
        def __init__(self, samples, weights, logz, log_likelihoods):
            self.samples = samples
            self.weights = weights
            self.logz = np.array([logz])  # Make it an array like nestle
            self.log_likelihoods = log_likelihoods

    results = MockResults(positions, weights, log_Z, log_likelihoods)
    return log_Z, results


def _inv_sample_ecc(u: float, planet: bool, P_orb: float) -> float:
    """
    Deterministic inverse-CDF sampler for eccentricity matching priors used in sample_ecc.
    For planets: Beta(0.867, 3.030) (Kipping 2013).
    For binaries: powerlaw with exponent depending on period (Moe & Di Stefano 2017).
    """
    from scipy.stats import beta as _beta
    from scipy.stats import powerlaw as _powerlaw

    u = np.clip(u, 1e-12, 1-1e-12)
    if planet:
        return float(_beta.ppf(u, 0.867, 3.030))
    else:
        a_shape = 0.2 if P_orb <= 10 else 0.6
        return float(_powerlaw.ppf(u, a_shape))

def lnZ_TTP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the TTP scenario.
    Persistent sampling usage:
      - nlive: number of active particles (default 20). Increase for better sampling.
      - dlogz: not used, target_ess set to nlive.
      - dynamic: not used.
    Period handling: if P_orb is a 2-tuple, we assume Uniform(P_min, P_max); if scalar, P is fixed.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of draws for MC.
        steps (int): Number of steps for MCMC.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # For evidence, we'll treat either fixed P_orb or a uniform range
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)
    # a depends on P_orb; will be computed per-sample inside loglike
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # Build dynesty prior transform and log-likelihood
    # Parameterization: theta = [u_P, u_inc, u_ecc, u_argp, u_rp]
    # where u_* are in [0,1]. We transform to physical values using
    # the existing prior samplers.

    def prior_transform(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, urp = u
        # P_orb
        if P_orb_range[0] == P_orb_range[1]:
            P = P_orb_range[0]
        else:
            P = P_orb_range[0] + uP * (P_orb_range[1] - P_orb_range[0])
        # Eccentricity and argp
        ecc = _inv_sample_ecc(float(uecc), planet=True, P_orb=P)
        #ecc = float(sample_ecc(np.array([uecc]), planet=True, P_orb=P)[0])
        argp = float(sample_w(np.array([uargp]))[0])
        # Sample rp
        rp = float(sample_rp(np.array([urp]), np.array([M_s]), flatpriors)[0])
        # Inclination prior on [0, 90] degrees.
        # Transit geometry (inc >= inc_min) is enforced in the likelihood
        # so the evidence includes the geometric transit probability.
        inc = float(sample_inc(np.array([uinc]), lower=0.0, upper=90.0)[0])
        return np.array([P, inc, ecc, argp, rp])

    def loglike(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, rp = theta
        a = ((G*M_s*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        # Transit probability / geometry checks
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (rp*Rearth + R_s*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        # collision check
        if (rp*Rearth + R_s*Rsun) > a*(1-ecc):
            return -np.inf
        # min inclination to transit
        #if Ptra <= 1.0:
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        #else:
        #    inc_min = 0.0
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_TP(
            time, flux, sigma, rp, P, inc, a, R_s, u1, u2,
            ecc, argp_, exptime=exptime, nsamples=nsamples
        )
        return float(lnL)

    #u = np.array([0.5,  0.9964872937794484, 0.09987218797188613, 0.045427036113811825, 0.17504388466655085])
    #print(prior_transform(u))
    #print(loglike(prior_transform(u)))

    #u = np.array([P_orb_range[0], 0.9946931766323459, 0.5734859411160013, 0.9182614005552816, 0.9863205269412394])
    #print(loglike(prior_transform(u)))

    # run persistent sampling
    lnZ, results = _run_persistent_evidence(
        loglike, prior_transform, ndim=5,
        n_active=N, target_ess=2 * N, mcmc_steps=steps,
    )

    #print(lnZ)
    # Best samples: take top N from weighted samples as representatives
    N_samples = POSTERIOR_NSAMPLES
    try:
        wsamps = _resample_equal(results.samples, results.weights, log_likelihoods=results.log_likelihoods)
        # compute b for these
        P = wsamps[:,0]
        inc = wsamps[:,1]
        ecc = wsamps[:,2]
        argp_ = wsamps[:,3]
        rp = wsamps[:,4]
        a_arr = ((G*M_s*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        r = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
        b = r*np.cos(inc*pi/180)/(R_s*Rsun)
        # Select up to N_samples
        sel = slice(0, min(N_samples, wsamps.shape[0]))
        res = {
            'M_s': np.full(min(N_samples, wsamps.shape[0]), M_s),
            'R_s': np.full(min(N_samples, wsamps.shape[0]), R_s),
            'u1': np.full(min(N_samples, wsamps.shape[0]), u1),
            'u2': np.full(min(N_samples, wsamps.shape[0]), u2),
            'P_orb': P[sel],
            'inc': inc[sel],
            'b': b[sel],
            'R_p': rp[sel],
            'ecc': ecc[sel],
            'argp': argp_[sel],
            'M_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'R_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_comp': np.zeros(min(N_samples, wsamps.shape[0])),
            'lnZ': float(lnZ)
        }
    except Exception:
        res = {
            'M_s': np.array([M_s]),
            'R_s': np.array([R_s]),
            'u1': np.array([u1]),
            'u2': np.array([u2]),
            'P_orb': np.array([P_orb_range[0]]),
            'inc': np.array([90.0]),
            'b': np.array([0.0]),
            'R_p': np.array([1.0]),
            'ecc': np.array([0.0]),
            'argp': np.array([90.0]),
            'M_EB': np.array([0.0]),
            'R_EB': np.array([0.0]),
            'fluxratio_EB': np.array([0.0]),
            'fluxratio_comp': np.array([0.0]),
            'lnZ': float(lnZ)
        }
    res['best_lnL'] = float(np.max(results.log_likelihoods))
    return res


def lnZ_TEB(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            N_twin: int = None, steps_twin: int = None):
    """
    Calculates the marginal likelihood of the TEB scenario with dynesty.
    Two branches are evaluated separately:
      - Single: q < 0.95 using lnL_EB.
      - Twin: q >= 0.95 using lnL_EB_twin with 2×P.
    Evidence (lnZ) is returned for each branch along with representative samples.
    Tuning: nlive, dlogz, dynamic control dynesty's behavior.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of particles.
        steps (int): Number of steps for MCMC.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # dynesty prior transforms
    # theta = [uP, uinc, uecc, uargp, uq, umode]
    # umode is ignored in transform but we'll compute both single and twin

    def _q_from_u(uq: float) -> float:
        # Use sample_q via inverse transform by searching u in [0,1]
        # We approximate using direct sampler on a single value
        return float(sample_q(np.array([uq]), M_s)[0])

    def prior_transform_single(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, uq = u
        if P_orb_range[0] == P_orb_range[1]:
            P = P_orb_range[0]
        else:
            P = P_orb_range[0] + uP * (P_orb_range[1] - P_orb_range[0])
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=False, P_orb=P)
        argp = float(sample_w(np.array([uargp]))[0])
        q = _q_from_u(float(uq))
        return np.array([P, inc, ecc, argp, q])

    def loglike_single(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q = theta
        # only q < 0.95 branch
        if q >= 0.95:
            return -np.inf
        masses = q*M_s
        radii, _ = stellar_relations(np.array([masses]), np.array([R_s]), np.array([Teff]))
        radii = float(radii[0])
        fluxratio = float(
            flux_relation(np.array([masses]))
            / (flux_relation(np.array([masses])) + flux_relation(np.array([M_s])))
        )
        a = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        # geometry checks
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (radii*Rsun + R_s*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        if (radii*Rsun + R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB(
            time, flux, sigma, radii, fluxratio, P, inc, a, R_s, u1, u2,
            ecc, argp_, exptime=exptime, nsamples=nsamples
        )
        return float(lnL)

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        # twin branch uses 2*P
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q = theta
        if q < 0.95:
            return -np.inf
        masses = q*M_s
        radii, _ = stellar_relations(np.array([masses]), np.array([R_s]), np.array([Teff]))
        radii = float(radii[0])
        fluxratio = float(
            flux_relation(np.array([masses]))
            / (flux_relation(np.array([masses])) + flux_relation(np.array([M_s])))
        )
        a = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (radii*Rsun + R_s*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        if (2*R_s*Rsun) > a*(1-ecc):  # conservative collision for twin
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB_twin(
            time, flux, sigma, radii, fluxratio, P, inc, a, R_s, u1, u2,
            ecc, argp_, exptime=exptime, nsamples=nsamples
        )
        return float(lnL)

    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=5,
        n_active=N, target_ess=2 * N, mcmc_steps=steps,
    )
    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=5,
        n_active=N if N_twin is None else N_twin,
        target_ess=2 * (N if N_twin is None else N_twin),
        mcmc_steps=steps if steps_twin is None else steps_twin,
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights, log_likelihoods=res_obj.log_likelihoods)
            sel = slice(0, min(N_samples, wsamps.shape[0]))
            P = wsamps[:,0]
            inc = wsamps[:,1]
            ecc = wsamps[:,2]
            argp_ = wsamps[:,3]
            q = wsamps[:,4]
            masses = q*M_s
            radii, _ = stellar_relations(masses, np.full_like(masses, R_s), np.full_like(masses, Teff))
            fluxratios = (
                flux_relation(masses)
                / (flux_relation(masses) + flux_relation(np.array([M_s])))
            )
            a_arr = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
            r = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
            b = r*np.cos(inc*pi/180)/(R_s*Rsun)
            out = {
                'M_s': np.full(min(N_samples, wsamps.shape[0]), M_s),
                'R_s': np.full(min(N_samples, wsamps.shape[0]), R_s),
                'u1': np.full(min(N_samples, wsamps.shape[0]), u1),
                'u2': np.full(min(N_samples, wsamps.shape[0]), u2),
                # The twin prior transform already changes P_fold to 2P.
                'P_orb': P[sel],
                'inc': inc[sel],
                'b': b[sel],
                'R_p': np.zeros(min(N_samples, wsamps.shape[0])),
                'ecc': ecc[sel],
                'argp': argp_[sel],
                'M_EB': masses[sel],
                'R_EB': radii[sel],
                'fluxratio_EB': fluxratios[sel],
                'fluxratio_comp': np.zeros(min(N_samples, wsamps.shape[0])),
            }
        except Exception:
            out = {
                'M_s': np.array([M_s]),
                'R_s': np.array([R_s]),
                'u1': np.array([u1]),
                'u2': np.array([u2]),
                'P_orb': np.array([2*P_orb_range[0] if twin else P_orb_range[0]]),
                'inc': np.array([90.0]),
                'b': np.array([0.0]),
                'R_p': np.array([0.0]),
                'ecc': np.array([0.0]),
                'argp': np.array([90.0]),
                'M_EB': np.array([0.0]),
                'R_EB': np.array([0.0]),
                'fluxratio_EB': np.array([0.0]),
                'fluxratio_comp': np.array([0.0]),
            }
        return out

    res = _build_output(res_s, twin=False)
    res['lnZ'] = float(lnZ_single)
    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    return res, res_twin


def lnZ_PTP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None):
    """
    Calculates the marginal likelihood of the PTP scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of particles.
        steps (int): Number of steps for MCMC.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)
    # a depends on P in loglike
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # Precompute companion mass-ratio sampling if MOLUSC provided
    molusc_qs = None
    if molusc_file is not None:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a*(1-molusc_e) > 10]
        molusc_qs = molusc_df2["mass ratio"].values
        molusc_qs[molusc_qs < 0.1/M_s] = 0.1/M_s

    # Build prior transform: [uP, uinc, uecc, uargp, urp, uqcomp]
    def _qcomp_from_u(uq: float) -> float:
        if molusc_qs is not None and molusc_qs.size > 0:
            # discrete selection
            idx = int(np.floor(np.clip(uq, 1e-12, 1-1e-12) * molusc_qs.size))
            idx = min(idx, molusc_qs.size-1)
            return float(molusc_qs[idx])
        else:
            return float(sample_q_companion(np.array([uq]), M_s)[0])

    def prior_transform(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, urp, uq = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=True, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        rp = float(sample_rp(np.array([urp]), np.array([M_s]), flatpriors)[0])
        qcomp = _qcomp_from_u(float(uq))
        return np.array([P, inc, ecc, argp_, rp, qcomp])

    # Preload contrast curve data if provided
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = np.array([2.2]), np.array([1.0])

    def companion_lnprior(mass_comp: float) -> float:
        # flux ratio and delta mag depend on chosen filter
        if contrast_curve_file is None:
            fr_cc = flux_relation(np.array([mass_comp]))/(flux_relation(np.array([mass_comp]))+flux_relation(np.array([M_s])))
        else:
            fr_cc = flux_relation(np.array([mass_comp]), filt)/(flux_relation(np.array([mass_comp]), filt)+flux_relation(np.array([M_s]), filt))
        delta_mag = float(2.5*np.log10(fr_cc/(1-fr_cc)))
        if mission.upper() in ("TESS", "K2", "KEPLER"):
            lnpr = float(lnprior_bound_TP(M_s, plx, np.array([abs(delta_mag)]), separations, contrasts)[0])
        else:
            lnpr = float(lnprior_bound_TP(M_s, plx, np.array([abs(delta_mag)]), separations, contrasts)[0])
        if lnpr > 0.0 or delta_mag > 0.0:
            return -np.inf
        return lnpr

    def loglike(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, rp, qcomp = theta
        if qcomp <= 0.0:
            return -np.inf
        M_comp = qcomp*M_s
        R_comp, _ = stellar_relations(np.array([M_comp]), np.array([R_s]), np.array([Teff]))
        R_comp = float(R_comp[0])
        # flux ratios in TESS band for dilution in transit
        fr = float(
            flux_relation(np.array([M_comp]))
            / (flux_relation(np.array([M_comp])) + flux_relation(np.array([M_s])))
        )
        a = ((G*M_s*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (rp*Rearth + R_s*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        if (rp*Rearth + R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_TP(
            time, flux, sigma, rp, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        # add companion prior factor
        lnL += companion_lnprior(M_comp)
        return float(lnL)

    lnZ, results = _run_persistent_evidence(
        loglike, prior_transform, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # Build representative outputs
    N_samples = POSTERIOR_NSAMPLES
    try:
        wsamps = _resample_equal(results.samples, results.weights, log_likelihoods=results.log_likelihoods)
        P = wsamps[:,0]
        inc = wsamps[:,1]
        ecc = wsamps[:,2]
        argp_ = wsamps[:,3]
        rp = wsamps[:,4]
        qcomp = wsamps[:,5]
        a_arr = ((G*M_s*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        r = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
        b = r*np.cos(inc*pi/180)/(R_s*Rsun)
        M_comp = qcomp*M_s
        fr_comp = (
            flux_relation(M_comp)
            / (flux_relation(M_comp) + flux_relation(np.array([M_s])))
        )
        sel = slice(0, min(N_samples, wsamps.shape[0]))
        res = {
            'M_s': np.full(min(N_samples, wsamps.shape[0]), M_s),
            'R_s': np.full(min(N_samples, wsamps.shape[0]), R_s),
            'u1': np.full(min(N_samples, wsamps.shape[0]), u1),
            'u2': np.full(min(N_samples, wsamps.shape[0]), u2),
            'P_orb': P[sel],
            'inc': inc[sel],
            'b': b[sel],
            'R_p': rp[sel],
            'ecc': ecc[sel],
            'argp': argp_[sel],
            'M_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'R_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_comp': fr_comp[sel],
            'lnZ': float(lnZ)
        }
    except Exception:
        res = {
            'M_s': np.array([M_s]), 'R_s': np.array([R_s]), 'u1': np.array([u1]), 'u2': np.array([u2]),
            'P_orb': np.array([P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
            'R_p': np.array([1.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
            'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]),
            'fluxratio_comp': np.array([0.0]), 'lnZ': float(lnZ)
        }
    res['best_lnL'] = float(np.max(results.log_likelihoods))
    return res


def lnZ_PEB(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None,
            N_twin: int = None, steps_twin: int = None):
    """
    Calculates the marginal likelihood of the PEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of particles.
        steps (int): Number of steps for MCMC.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # Precompute MOLUSC q_comp pool, if provided
    molusc_qs = None
    if molusc_file is not None:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a*(1-molusc_e) > 10]
        molusc_qs = molusc_df2["mass ratio"].values
        molusc_qs[molusc_qs < 0.1/M_s] = 0.1/M_s

    def _qcomp_from_u(uq: float) -> float:
        if molusc_qs is not None and molusc_qs.size > 0:
            idx = int(np.floor(np.clip(uq, 1e-12, 1-1e-12) * molusc_qs.size))
            idx = min(idx, molusc_qs.size-1)
            return float(molusc_qs[idx])
        else:
            return float(sample_q_companion(np.array([uq]), M_s)[0])

    # Preload contrast curve data if provided
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = np.array([2.2]), np.array([1.0])

    def companion_lnprior(mass_comp: float) -> float:
        # Compute delta mag in chosen filter and bound companion prior
        if contrast_curve_file is None:
            fr_cc = flux_relation(np.array([mass_comp]))/(flux_relation(np.array([mass_comp]))+flux_relation(np.array([M_s])))
        else:
            fr_cc = flux_relation(np.array([mass_comp]), filt)/(flux_relation(np.array([mass_comp]), filt)+flux_relation(np.array([M_s]), filt))
        delta_mag = float(2.5*np.log10(fr_cc/(1-fr_cc)))
        lnpr = float(lnprior_bound_EB(M_s, plx, np.array([abs(delta_mag)]), separations, contrasts)[0])
        if lnpr > 0.0 or delta_mag > 0.0:
            return -np.inf
        return lnpr

    # dynesty prior/likelihood for single and twin branches
    # theta = [uP, uinc, uecc, uargp, uq, uqcomp]
    def prior_transform_single(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, uq, uqcomp = u
        # P_orb
        if P_orb_range[0] == P_orb_range[1]:
            P = P_orb_range[0]
        else:
            P = P_orb_range[0] + uP * (P_orb_range[1] - P_orb_range[0])
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=False, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        q = float(sample_q(np.array([uq]), M_s)[0])
        qcomp = _qcomp_from_u(float(uqcomp))
        return np.array([P, inc, ecc, argp_, q, qcomp])

    def loglike_single(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, qcomp = theta
        if q >= 0.95:
            return -np.inf
        masses = q*M_s
        radii, _ = stellar_relations(np.array([masses]), np.array([R_s]), np.array([Teff]))
        radii = float(radii[0])
        fluxratio = float(
            flux_relation(np.array([masses]))
            / (flux_relation(np.array([masses])) + flux_relation(np.array([M_s])))
        )
        M_comp = qcomp*M_s
        fr_comp = float(
            flux_relation(np.array([M_comp]))
            / (flux_relation(np.array([M_comp])) + flux_relation(np.array([M_s])))
        )
        a = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (radii*Rsun + R_s*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        if (radii*Rsun + R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB(
            time, flux, sigma, radii, fluxratio, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr_comp, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        lnL += companion_lnprior(M_comp)
        return float(lnL)

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, qcomp = theta
        if q < 0.95:
            return -np.inf
        masses = q*M_s
        radii, _ = stellar_relations(np.array([masses]), np.array([R_s]), np.array([Teff]))
        radii = float(radii[0])
        fluxratio = float(
            flux_relation(np.array([masses]))
            / (flux_relation(np.array([masses])) + flux_relation(np.array([M_s])))
        )
        M_comp = qcomp*M_s
        fr_comp = float(
            flux_relation(np.array([M_comp]))
            / (flux_relation(np.array([M_comp])) + flux_relation(np.array([M_s])))
        )
        a = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (radii*Rsun + R_s*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        if (2*R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB_twin(
            time, flux, sigma, radii, fluxratio, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr_comp, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        lnL += companion_lnprior(M_comp)
        return float(lnL)

    # Run nested sampling for both branches
    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6, n_active=N, target_ess=2 * N, mcmc_steps=steps
    )
    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6, n_active=N if N_twin is None else N_twin,
        target_ess=2 * (N if N_twin is None else N_twin),
        mcmc_steps=steps if steps_twin is None else steps_twin
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights, log_likelihoods=res_obj.log_likelihoods)
            sel = slice(0, min(N_samples, wsamps.shape[0]))
            P = wsamps[:,0]
            inc = wsamps[:,1]
            ecc = wsamps[:,2]
            argp_ = wsamps[:,3]
            q = wsamps[:,4]
            qcomp = wsamps[:,5]
            masses = q*M_s
            radii, _ = stellar_relations(masses, np.full_like(masses, R_s), np.full_like(masses, Teff))
            fluxratios = (
                flux_relation(masses)
                / (flux_relation(masses) + flux_relation(np.array([M_s])))
            )
            a_arr = ((G*(M_s+masses)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
            r = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
            b = r*np.cos(inc*pi/180)/(R_s*Rsun)
            M_comp = qcomp*M_s
            fr_comp = (
                flux_relation(M_comp)
                / (flux_relation(M_comp) + flux_relation(np.array([M_s])))
            )
            out = {
                'M_s': np.full(min(N_samples, wsamps.shape[0]), M_s),
                'R_s': np.full(min(N_samples, wsamps.shape[0]), R_s),
                'u1': np.full(min(N_samples, wsamps.shape[0]), u1),
                'u2': np.full(min(N_samples, wsamps.shape[0]), u2),
                # The twin prior transform already changes P_fold to 2P.
                'P_orb': P[sel],
                'inc': inc[sel],
                'b': b[sel],
                'R_p': np.zeros(min(N_samples, wsamps.shape[0])),
                'ecc': ecc[sel],
                'argp': argp_[sel],
                'M_EB': masses[sel],
                'R_EB': radii[sel],
                'fluxratio_EB': fluxratios[sel],
                'fluxratio_comp': fr_comp[sel],
            }
        except Exception:
            out = {
                'M_s': np.array([M_s]),
                'R_s': np.array([R_s]),
                'u1': np.array([u1]),
                'u2': np.array([u2]),
                'P_orb': np.array([2*P_orb_range[0] if twin else P_orb_range[0]]),
                'inc': np.array([90.0]),
                'b': np.array([0.0]),
                'R_p': np.array([0.0]),
                'ecc': np.array([0.0]),
                'argp': np.array([90.0]),
                'M_EB': np.array([0.0]),
                'R_EB': np.array([0.0]),
                'fluxratio_EB': np.array([0.0]),
                'fluxratio_comp': np.array([0.0]),
            }
        return out

    res = _build_output(res_s, twin=False)
    res['lnZ'] = float(lnZ_single)
    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    return res, res_twin


def lnZ_STP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float, Z: float,
            plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None):
    """
    Calculates the marginal likelihood of the STP scenario using nested sampling.
    Planet transits the bound stellar companion (companion is the host).
    Returns representative samples and lnZ.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)

    # MOLUSC q_comp sampling (optional)
    molusc_qs = None
    if molusc_file is not None:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a*(1-molusc_e) > 10]
        molusc_qs = molusc_df2["mass ratio"].values
        molusc_qs[molusc_qs < 0.1/M_s] = 0.1/M_s

    def _qcomp_from_u(uq: float) -> float:
        if molusc_qs is not None and molusc_qs.size > 0:
            idx = int(np.floor(np.clip(uq, 1e-12, 1-1e-12) * molusc_qs.size))
            idx = min(idx, molusc_qs.size-1)
            return float(molusc_qs[idx])
        else:
            return float(sample_q_companion(np.array([uq]), M_s)[0])

    # Preload contrast curve if provided
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = np.array([2.2]), np.array([1.0])

    # Limb-darkening grid accessors
    if mission == "TESS":
        _ldc_Zs, _ldc_Teffs, _ldc_loggs = ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs
        _ldc_u1s, _ldc_u2s = ldc_T_u1s, ldc_T_u2s
        ldc_at_Z = ldc_T[(_ldc_Zs == _ldc_Zs[np.abs(_ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.aLSM, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.bLSM, dtype=float)
    else:
        _ldc_Zs, _ldc_Teffs, _ldc_loggs = ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs
        _ldc_u1s, _ldc_u2s = ldc_K_u1s, ldc_K_u2s
        ldc_at_Z = ldc_K[(_ldc_Zs == _ldc_Zs[np.abs(_ldc_Zs - Z).argmin()])]
        Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
        loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
        u1s_at_Z = np.array(ldc_at_Z.a, dtype=float)
        u2s_at_Z = np.array(ldc_at_Z.b, dtype=float)

    def _comp_ldc(M_comp: float) -> tuple:
        R_comp, Teff_comp = stellar_relations(np.array([M_comp]), np.array([R_s]), np.array([Teff]))
        R_comp = float(R_comp[0])
        Teff_comp = float(Teff_comp[0])
        logg_comp = float(np.log10(G*(M_comp*Msun)/(R_comp*Rsun)**2))
        u1, u2 = nearest_ldc_coefficients(
            Teff_comp, logg_comp, Teffs_at_Z, loggs_at_Z,
            u1s_at_Z, u2s_at_Z
        )
        return R_comp, u1, u2

    def companion_lnprior(mass_comp: float) -> float:
        # Flux ratio in chosen filter for contrast curve prior
        if contrast_curve_file is None:
            fr_cc = flux_relation(np.array([mass_comp]))/(flux_relation(np.array([mass_comp]))+flux_relation(np.array([M_s])))
        else:
            fr_cc = flux_relation(np.array([mass_comp]), filt)/(flux_relation(np.array([mass_comp]), filt)+flux_relation(np.array([M_s]), filt))
        delta_mag = float(2.5*np.log10(fr_cc/(1-fr_cc)))
        lnpr = float(lnprior_bound_TP(M_s, plx, np.array([abs(delta_mag)]), separations, contrasts)[0])
        if lnpr > 0.0 or delta_mag > 0.0:
            return -np.inf
        return lnpr

    # Prior transform: [uP, uinc, uecc, uargp, urp, uqcomp]
    def prior_transform(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, urp, uq = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=True, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        qcomp = _qcomp_from_u(float(uq))
        # rp prior depends on host mass which is M_comp (companion)
        M_comp = qcomp*M_s
        rp = float(sample_rp(np.array([urp]), np.array([M_comp]), flatpriors)[0])
        return np.array([P, inc, ecc, argp_, rp, qcomp])

    def loglike(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, rp, qcomp = theta
        if qcomp <= 0.0:
            return -np.inf
        M_comp = qcomp*M_s
        R_comp, u1c, u2c = _comp_ldc(M_comp)
        # Dilution flux ratio in TESS band
        fr_comp = float(
            flux_relation(np.array([M_comp]))
            / (flux_relation(np.array([M_comp])) + flux_relation(np.array([M_s])))
        )
        a = ((G*M_comp*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (rp*Rearth + R_comp*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        if (rp*Rearth + R_comp*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_TP(
            time, flux, sigma, rp, P, inc, a, R_comp, u1c, u2c,
            ecc, argp_, companion_fluxratio=fr_comp, companion_is_host=True,
            exptime=exptime, nsamples=nsamples
        )
        lnL += companion_lnprior(M_comp)
        return float(lnL)

    lnZ, results = _run_persistent_evidence(
        loglike, prior_transform, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # Representative outputs
    N_samples = POSTERIOR_NSAMPLES
    try:
        wsamps = _resample_equal(results.samples, results.weights, log_likelihoods=results.log_likelihoods)
        P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; rp = wsamps[:,4]; qcomp = wsamps[:,5]
        M_comp = qcomp*M_s
        # Compute companion radius and LDCs vectorized
        R_comp, Teff_comp = stellar_relations(M_comp, np.full_like(M_comp, R_s), np.full_like(M_comp, Teff))
        logg_comp = np.log10(G*(M_comp*Msun)/(R_comp*Rsun)**2)
        if mission == "TESS":
            ldc_at_Z = ldc_T[(ldc_T_Zs == ldc_T_Zs[np.abs(ldc_T_Zs - Z).argmin()])]
            Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
            loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
            u1s_at_Z = np.array(ldc_at_Z.aLSM, dtype=float)
            u2s_at_Z = np.array(ldc_at_Z.bLSM, dtype=float)
        else:
            ldc_at_Z = ldc_K[(ldc_K_Zs == ldc_K_Zs[np.abs(ldc_K_Zs - Z).argmin()])]
            Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
            loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
            u1s_at_Z = np.array(ldc_at_Z.a, dtype=float)
            u2s_at_Z = np.array(ldc_at_Z.b, dtype=float)
        u1c, u2c = nearest_ldc_coefficients(
            Teff_comp, logg_comp, Teffs_at_Z, loggs_at_Z,
            u1s_at_Z, u2s_at_Z
        )
        a_arr = ((G*M_comp*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        r = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
        b = r*np.cos(inc*pi/180)/(R_comp*Rsun)
        fr_comp = (
            flux_relation(M_comp)
            / (flux_relation(M_comp) + flux_relation(np.array([M_s])))
        )
        sel = slice(0, min(N_samples, wsamps.shape[0]))
        res = {
            'M_s': M_comp[sel],
            'R_s': R_comp[sel],
            'u1': u1c[sel],
            'u2': u2c[sel],
            'P_orb': P[sel],
            'inc': inc[sel],
            'b': b[sel],
            'R_p': rp[sel],
            'ecc': ecc[sel],
            'argp': argp_[sel],
            'M_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'R_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_comp': fr_comp[sel],
            'lnZ': float(lnZ)
        }
    except Exception:
        res = {
            'M_s': np.array([M_s]), 'R_s': np.array([R_s]), 'u1': np.array([0.3]), 'u2': np.array([0.2]),
            'P_orb': np.array([P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
            'R_p': np.array([1.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
            'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]),
            'fluxratio_comp': np.array([0.0]), 'lnZ': float(lnZ)
        }
    res['best_lnL'] = float(np.max(results.log_likelihoods))
    return res


def lnZ_SEB(time: np.ndarray, flux: np.ndarray, sigma: float,
        P_orb: float, M_s: float, R_s: float, Teff: float,
        Z: float, plx: float, contrast_curve_file: str = None,
        filt: str = "TESS",
        N: int = 10, steps: int = 20,
        mission: str = "TESS", flatpriors: bool = False,
        exptime: float = 0.00139, nsamples: int = 20,
        molusc_file: str = None,
            N_twin: int = None, steps_twin: int = None):
    """
    Calculates the marginal likelihood of the SEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        plx (float): Target star parallax [mas].
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of particles.
        steps (int): Number of steps for MCMC.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)

    # Preload contrast curve data if provided
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = np.array([2.2]), np.array([1.0])

    # MOLUSC q_comp optional pool
    molusc_qs = None
    if molusc_file is not None:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a*(1-molusc_e) > 10]
        molusc_qs = molusc_df2["mass ratio"].values
        molusc_qs[molusc_qs < 0.1/M_s] = 0.1/M_s

    def _qcomp_from_u(uq: float) -> float:
        if molusc_qs is not None and molusc_qs.size > 0:
            idx = int(np.floor(np.clip(uq, 1e-12, 1-1e-12) * molusc_qs.size))
            idx = min(idx, molusc_qs.size-1)
            return float(molusc_qs[idx])
        else:
            return float(sample_q_companion(np.array([uq]), M_s)[0])

    # limb darkening accessor based on companion mass
    def _comp_props_and_ldc(M_comp: float) -> tuple:
        R_comp, Teff_comp = stellar_relations(np.array([M_comp]), np.array([R_s]), np.array([Teff]))
        R_comp = float(R_comp[0]); Teff_comp = float(Teff_comp[0])
        logg_comp = float(np.log10(G*(M_comp*Msun)/(R_comp*Rsun)**2))
        if mission == "TESS":
            ldc_at_Z = ldc_T[(ldc_T_Zs == ldc_T_Zs[np.abs(ldc_T_Zs - Z).argmin()])]
            Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
            loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
            u1s_at_Z = np.array(ldc_at_Z.aLSM, dtype=float)
            u2s_at_Z = np.array(ldc_at_Z.bLSM, dtype=float)
        else:
            ldc_at_Z = ldc_K[(ldc_K_Zs == ldc_K_Zs[np.abs(ldc_K_Zs - Z).argmin()])]
            Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
            loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
            u1s_at_Z = np.array(ldc_at_Z.a, dtype=float)
            u2s_at_Z = np.array(ldc_at_Z.b, dtype=float)
        u1, u2 = nearest_ldc_coefficients(
            Teff_comp, logg_comp, Teffs_at_Z, loggs_at_Z,
            u1s_at_Z, u2s_at_Z
        )
        return R_comp, u1, u2

    def companion_lnprior(M_comp: float, M_eb: float) -> float:
        # Build combined delta-mag for EB+host system
        if contrast_curve_file is None:
            fr_host = flux_relation(np.array([M_comp]))/(flux_relation(np.array([M_comp]))+flux_relation(np.array([M_s])))
            fr_eb = flux_relation(np.array([M_eb]))/(flux_relation(np.array([M_eb]))+flux_relation(np.array([M_s])))
        else:
            fr_host = flux_relation(np.array([M_comp]), filt)/(flux_relation(np.array([M_comp]), filt)+flux_relation(np.array([M_s]), filt))
            fr_eb = flux_relation(np.array([M_eb]), filt)/(flux_relation(np.array([M_eb]), filt)+flux_relation(np.array([M_s]), filt))
        delta_mag = float(2.5*np.log10((fr_host/(1-fr_host)) + (fr_eb/(1-fr_eb))))
        lnpr = float(lnprior_bound_EB(M_s, plx, np.array([abs(delta_mag)]), separations, contrasts)[0])
        if lnpr > 0.0 or delta_mag > 0.0:
            return -np.inf
        return lnpr

    # theta = [uP, uinc, uecc, uargp, uq, uqcomp]
    def prior_transform_single(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, uq, uqcomp = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=False, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        q = float(sample_q(np.array([uq]), M_s)[0])
        qcomp = _qcomp_from_u(float(uqcomp))
        return np.array([P, inc, ecc, argp_, q, qcomp])

    def loglike_single(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, qcomp = theta
        if q >= 0.95:
            return -np.inf
        M_comp = qcomp*M_s
        R_comp, u1c, u2c = _comp_props_and_ldc(M_comp)
        M_eb = q*M_comp
        R_eb, _ = stellar_relations(np.array([M_eb]), np.array([R_comp]), np.array([Teff]))
        R_eb = float(R_eb[0])
        fr_eb = float(
            flux_relation(np.array([M_eb]))
            / (flux_relation(np.array([M_eb])) + flux_relation(np.array([M_s])))
        )
        fr_host = float(
            flux_relation(np.array([M_comp]))
            / (flux_relation(np.array([M_comp])) + flux_relation(np.array([M_s])))
        )
        a = ((G*(M_comp+M_eb)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R_eb*Rsun + R_comp*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        if (R_eb*Rsun + R_comp*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB(
            time, flux, sigma, R_eb, fr_eb, P, inc, a, R_comp, u1c, u2c,
            ecc, argp_, companion_fluxratio=fr_host, companion_is_host=True,
            exptime=exptime, nsamples=nsamples
        )
        lnL += companion_lnprior(M_comp, M_eb)
        return float(lnL)

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, qcomp = theta
        if q < 0.95:
            return -np.inf
        M_comp = qcomp*M_s
        R_comp, u1c, u2c = _comp_props_and_ldc(M_comp)
        M_eb = q*M_comp
        R_eb, _ = stellar_relations(np.array([M_eb]), np.array([R_comp]), np.array([Teff]))
        R_eb = float(R_eb[0])
        fr_eb = float(
            flux_relation(np.array([M_eb]))
            / (flux_relation(np.array([M_eb])) + flux_relation(np.array([M_s])))
        )
        fr_host = float(
            flux_relation(np.array([M_comp]))
            / (flux_relation(np.array([M_comp])) + flux_relation(np.array([M_s])))
        )
        a = ((G*(M_comp+M_eb)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R_eb*Rsun + R_comp*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        # twin collision criterion
        if (2*R_comp*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB_twin(
            time, flux, sigma, R_eb, fr_eb, P, inc, a, R_comp, u1c, u2c,
            ecc, argp_, companion_fluxratio=fr_host, companion_is_host=True,
            exptime=exptime, nsamples=nsamples
        )
        lnL += companion_lnprior(M_comp, M_eb)
        return float(lnL)

    # Run nested sampling for both branches
    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )
    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N if N_twin is None else N_twin,
        target_ess=2 * (N if N_twin is None else N_twin),
        mcmc_steps=steps if steps_twin is None else steps_twin
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights, log_likelihoods=res_obj.log_likelihoods)
            sel = slice(0, min(N_samples, wsamps.shape[0]))
            P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; q = wsamps[:,4]; qcomp = wsamps[:,5]
            M_comp = qcomp*M_s
            R_comp, Teff_comp = stellar_relations(M_comp, np.full_like(M_comp, R_s), np.full_like(M_comp, Teff))
            # LDCs per sample
            logg_comp = np.log10(G*(M_comp*Msun)/(R_comp*Rsun)**2)
            if mission == "TESS":
                ldc_at_Z = ldc_T[(ldc_T_Zs == ldc_T_Zs[np.abs(ldc_T_Zs - Z).argmin()])]
                Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
                loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
                u1s_at_Z = np.array(ldc_at_Z.aLSM, dtype=float)
                u2s_at_Z = np.array(ldc_at_Z.bLSM, dtype=float)
            else:
                ldc_at_Z = ldc_K[(ldc_K_Zs == ldc_K_Zs[np.abs(ldc_K_Zs - Z).argmin()])]
                Teffs_at_Z = np.array(ldc_at_Z.Teff, dtype=int)
                loggs_at_Z = np.array(ldc_at_Z.logg, dtype=float)
                u1s_at_Z = np.array(ldc_at_Z.a, dtype=float)
                u2s_at_Z = np.array(ldc_at_Z.b, dtype=float)
            u1c, u2c = nearest_ldc_coefficients(
                Teff_comp, logg_comp, Teffs_at_Z, loggs_at_Z,
                u1s_at_Z, u2s_at_Z
            )
            M_eb = q*M_comp
            R_eb, _ = stellar_relations(M_eb, R_comp, np.full_like(M_comp, Teff))
            fr_eb = (
                flux_relation(M_eb)
                / (flux_relation(M_eb) + flux_relation(np.array([M_s])))
            )
            fr_host = (
                flux_relation(M_comp)
                / (flux_relation(M_comp) + flux_relation(np.array([M_s])))
            )
            # The twin prior transform already changes P_fold to 2P.
            P_eff = P
            a_arr = ((G*(M_comp+M_eb)*Msun)/(4*pi**2)*(P_eff*86400)**2)**(1/3)
            r = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
            b = r*np.cos(inc*pi/180)/(R_comp*Rsun)
            out = {
                'M_s': M_comp[sel],
                'R_s': R_comp[sel],
                'u1': u1c[sel],
                'u2': u2c[sel],
                'P_orb': P_eff[sel],
                'inc': inc[sel],
                'b': b[sel],
                'R_p': np.zeros(min(N_samples, wsamps.shape[0])),
                'ecc': ecc[sel],
                'argp': argp_[sel],
                'M_EB': M_eb[sel],
                'R_EB': R_eb[sel],
                'fluxratio_EB': fr_eb[sel],
                'fluxratio_comp': fr_host[sel],
            }
        except Exception:
            out = {
                'M_s': np.array([M_s]),
                'R_s': np.array([R_s]),
                'u1': np.array([0.3]),
                'u2': np.array([0.2]),
                'P_orb': np.array([2*P_orb_range[0] if twin else P_orb_range[0]]),
                'inc': np.array([90.0]),
                'b': np.array([0.0]),
                'R_p': np.array([0.0]),
                'ecc': np.array([0.0]),
                'argp': np.array([90.0]),
                'M_EB': np.array([0.0]),
                'R_EB': np.array([0.0]),
                'fluxratio_EB': np.array([0.0]),
                'fluxratio_comp': np.array([0.0]),
            }
        return out

    res = _build_output(res_s, twin=False)
    res['lnZ'] = float(lnZ_single)
    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    return res, res_twin


def lnZ_DTP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, Tmag: float, Jmag: float, Hmag: float,
            Kmag: float, trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the DTP scenario using dynesty.
    Background host selection is modeled through TRILEGAL-based priors and
    optional contrast-curve constraints, which are added to the log-likelihood
    as lnprior terms. Evidence lnZ is estimated via nested sampling.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        trilegal_fname (string): File containing trilegal query results.
        contrast_curve_file (string): Contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of particles.
        steps (int): Number of steps for MCMC.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # determine target star limb darkening coefficients
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs
        ldc_Teffs = ldc_T_Teffs
        ldc_loggs = ldc_T_loggs
        ldc_u1s = ldc_T_u1s
        ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs
        ldc_Teffs = ldc_K_Teffs
        ldc_loggs = ldc_K_loggs
        ldc_u1s = ldc_K_u1s
        ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = (
        (ldc_Zs == this_Z)
        & (ldc_Teffs == this_Teff)
        & (ldc_loggs == this_logg)
        )
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # determine background star population properties
    # Background population
    (Tmags_comp, masses_comp, loggs_comp, Teffs_comp, Zs_comp,
        Jmags_comp, Hmags_comp, Kmags_comp) = trilegal_results(trilegal_fname, Tmag)
    delta_T = Tmag - Tmags_comp
    delta_J = Jmag - Jmags_comp
    delta_H = Hmag - Hmags_comp
    delta_K = Kmag - Kmags_comp
    fluxratios_comp_T = 10**(delta_T/2.5) / (1 + 10**(delta_T/2.5))
    N_comp = Tmags_comp.shape[0]

    # contrast curve
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = None, None

    # prior transform [uP, uinc, uecc, uargp, urp, uidx]
    def prior_transform(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, urp, uidx = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=True, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        rp = float(sample_rp(np.array([urp]), np.array([M_s]), flatpriors)[0])
        idx = int(np.floor(np.clip(uidx, 1e-12, 1-1e-12) * N_comp))
        idx = min(idx, N_comp-1)
        return np.array([P, inc, ecc, argp_, rp, float(idx)])

    # delta mag per filter function
    def _delta_mag_for_idx(idx: int) -> float:
        if filt == "J":
            return float(delta_J[idx])
        elif filt == "H":
            return float(delta_H[idx])
        elif filt == "K":
            return float(delta_K[idx])
        else:
            return float(delta_T[idx])

    def lnprior_background_idx(idx: int) -> float:
        dm = _delta_mag_for_idx(idx)
        if dm > 0.0:
            return -np.inf
        if separations is None:
            # approximate prior used previously
            lnpr = np.log10((N_comp/0.1) * (1/3600)**2 * 2.2**2)
            return float(0.0 if lnpr > 0.0 else lnpr)
        else:
            lnpr = float(lnprior_background(N_comp, np.array([abs(dm)]), separations, contrasts)[0])
            return float(0.0 if lnpr > 0.0 else lnpr)

    def loglike(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, rp, idx_f = theta
        idx = int(idx_f)
        a = ((G*M_s*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (rp*Rearth + R_s*Rsun)/a * e_corr
        if Ptra > 1.0:
            return -np.inf
        if (rp*Rearth + R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        fr_bg = float(fluxratios_comp_T[idx])
        lnL = -0.5*ln2pi - lnsigma - lnL_TP(
            time, flux, sigma, rp, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr_bg, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        lnL += lnprior_background_idx(idx)
        return float(lnL)

    lnZ, results = _run_persistent_evidence(
        loglike, prior_transform, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative outputs
    N_samples = POSTERIOR_NSAMPLES
    try:
        wsamps = _resample_equal(results.samples, results.weights, log_likelihoods=results.log_likelihoods)
        sel = slice(0, min(N_samples, wsamps.shape[0]))
        P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; rp = wsamps[:,4]; idxf = wsamps[:,5]
        idx = np.clip(np.floor(idxf).astype(int), 0, N_comp-1)
        a_arr = ((G*M_s*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        r = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
        b = r*np.cos(inc*pi/180)/(R_s*Rsun)
        fr_bg = fluxratios_comp_T[idx]
        res = {
            'M_s': np.full(min(N_samples, wsamps.shape[0]), M_s),
            'R_s': np.full(min(N_samples, wsamps.shape[0]), R_s),
            'u1': np.full(min(N_samples, wsamps.shape[0]), u1),
            'u2': np.full(min(N_samples, wsamps.shape[0]), u2),
            'P_orb': P[sel],
            'inc': inc[sel],
            'b': b[sel],
            'R_p': rp[sel],
            'ecc': ecc[sel],
            'argp': argp_[sel],
            'M_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'R_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_comp': fr_bg[sel],
            'lnZ': float(lnZ)
        }
    except Exception:
        res = {
            'M_s': np.array([M_s]), 'R_s': np.array([R_s]), 'u1': np.array([u1]), 'u2': np.array([u2]),
            'P_orb': np.array([P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
            'R_p': np.array([1.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
            'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]),
            'fluxratio_comp': np.array([0.0]), 'lnZ': float(lnZ)
        }
    res['best_lnL'] = float(np.max(results.log_likelihoods))
    return res


def lnZ_DEB(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, Tmag: float, Jmag: float, Hmag: float,
            Kmag: float, trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            N_twin: int = None, steps_twin: int = None):
    """
    Calculates the marginal likelihood of the DEB scenario.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        M_s (float): Target star mass [Solar masses].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        Tmag (float): Target star TESS magnitude.
        Jmag (float): Target star J magnitude.
        Hmag (float): Target star H magnitude.
        Kmag (float): Target star K magnitude.
        trilegal_fname (string): File containing trilegal query results.
        contrast_curve_file (string): Path to contrast curve file.
        filt (string): Photometric filter of contrast curve. Options
                         are TESS, Vis, J, H, and K.
        N (int): Number of particles.
        steps (int): Number of steps for MCMC.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)
    logg = np.log10(G*(M_s*Msun)/(R_s*Rsun)**2)
    # limb darkening for target (host star for DEB)
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs; ldc_Teffs = ldc_T_Teffs; ldc_loggs = ldc_T_loggs; ldc_u1s = ldc_T_u1s; ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs; ldc_Teffs = ldc_K_Teffs; ldc_loggs = ldc_K_loggs; ldc_u1s = ldc_K_u1s; ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = ((ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg))
    u1, u2 = ldc_u1s[mask], ldc_u2s[mask]

    # Background population (for dilution prior and flux ratio)
    (Tmags_comp, masses_comp, loggs_comp, Teffs_comp, Zs_comp, Jmags_comp, Hmags_comp, Kmags_comp) = trilegal_results(trilegal_fname, Tmag)
    delta_T = Tmag - Tmags_comp
    delta_J = Jmag - Jmags_comp
    delta_H = Hmag - Hmags_comp
    delta_K = Kmag - Kmags_comp
    fluxratios_comp_T = 10**(delta_T/2.5) / (1 + 10**(delta_T/2.5))
    N_comp = Tmags_comp.shape[0]

    # contrast curve data (optional)
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = None, None

    # helpers
    def _delta_mag_for_idx(idx: int) -> float:
        if filt == "J":
            return float(delta_J[idx])
        elif filt == "H":
            return float(delta_H[idx])
        elif filt == "K":
            return float(delta_K[idx])
        else:
            return float(delta_T[idx])

    def lnprior_background_idx(idx: int) -> float:
        dm = _delta_mag_for_idx(idx)
        if dm > 0.0:
            return -np.inf
        if separations is None:
            lnpr = np.log10((N_comp/0.1) * (1/3600)**2 * 2.2**2)
            return float(0.0 if lnpr > 0.0 else lnpr)
        lnpr = float(lnprior_background(N_comp, np.array([abs(dm)]), separations, contrasts)[0])
        return float(0.0 if lnpr > 0.0 else lnpr)

    # theta = [uP, uinc, uecc, uargp, uq, uidx]
    def prior_transform_single(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, uq, uidx = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=False, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        q = float(sample_q(np.array([uq]), M_s)[0])
        idx = int(np.floor(np.clip(uidx, 1e-12, 1-1e-12) * N_comp))
        idx = min(idx, N_comp-1)
        return np.array([P, inc, ecc, argp_, q, float(idx)])

    def loglike_single(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, idxf = theta
        if q >= 0.95:
            return -np.inf
        idx = int(idxf)
        M2 = q*M_s
        R2, _ = stellar_relations(np.array([M2]), np.array([R_s]), np.array([Teff]))
        R2 = float(R2[0])
        fr_eb = float(flux_relation(np.array([M2]))/(flux_relation(np.array([M2]))+flux_relation(np.array([M_s]))))
        fr_bg = float(fluxratios_comp_T[idx])
        a = ((G*(M_s+M2)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R2*Rsun + R_s*Rsun)/a * e_corr
        if Ptra > 1.0 or (R2*Rsun + R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB(
            time, flux, sigma, R2, fr_eb, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr_bg, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        lnL += lnprior_background_idx(idx)
        return float(lnL)

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, idxf = theta
        if q < 0.95:
            return -np.inf
        idx = int(idxf)
        M2 = q*M_s
        R2, _ = stellar_relations(np.array([M2]), np.array([R_s]), np.array([Teff]))
        R2 = float(R2[0])
        fr_eb = float(flux_relation(np.array([M2]))/(flux_relation(np.array([M2]))+flux_relation(np.array([M_s]))))
        fr_bg = float(fluxratios_comp_T[idx])
        a = ((G*(M_s+M2)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R2*Rsun + R_s*Rsun)/a * e_corr
        # twin collision criterion uses 2*R_s
        if Ptra > 1.0 or (2*R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB_twin(
            time, flux, sigma, R2, fr_eb, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr_bg, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        lnL += lnprior_background_idx(idx)
        return float(lnL)

    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )
    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N if N_twin is None else N_twin,
        target_ess=2 * (N if N_twin is None else N_twin),
        mcmc_steps=steps if steps_twin is None else steps_twin
    )

    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights, log_likelihoods=res_obj.log_likelihoods)
            sel = slice(0, min(N_samples, wsamps.shape[0]))
            P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; q = wsamps[:,4]; idxf = wsamps[:,5]
            idx = np.clip(np.floor(idxf).astype(int), 0, N_comp-1)
            M2 = q*M_s
            R2, _ = stellar_relations(M2, np.full_like(M2, R_s), np.full_like(M2, Teff))
            fr_eb = (
                flux_relation(M2)
                / (flux_relation(M2) + flux_relation(np.array([M_s])))
            )
            fr_bg = fluxratios_comp_T[idx]
            # The twin prior transform already changes P_fold to 2P.
            P_eff = P
            a_arr = ((G*(M_s+M2)*Msun)/(4*pi**2)*(P_eff*86400)**2)**(1/3)
            r = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
            b = r*np.cos(inc*pi/180)/(R_s*Rsun)
            out = {
                'M_s': np.full(min(N_samples, wsamps.shape[0]), M_s),
                'R_s': np.full(min(N_samples, wsamps.shape[0]), R_s),
                'u1': np.full(min(N_samples, wsamps.shape[0]), u1),
                'u2': np.full(min(N_samples, wsamps.shape[0]), u2),
                'P_orb': P_eff[sel],
                'inc': inc[sel],
                'b': b[sel],
                'R_p': np.zeros(min(N_samples, wsamps.shape[0])),
                'ecc': ecc[sel],
                'argp': argp_[sel],
                'M_EB': M2[sel],
                'R_EB': R2[sel],
                'fluxratio_EB': fr_eb[sel],
                'fluxratio_comp': fr_bg[sel],
            }
        except Exception:
            out = {
                'M_s': np.array([M_s]), 'R_s': np.array([R_s]), 'u1': np.array([u1]), 'u2': np.array([u2]),
                'P_orb': np.array([2*P_orb_range[0] if twin else P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
                'R_p': np.array([0.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
                'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]),
                'fluxratio_comp': np.array([0.0]),
            }
        return out

    res = _build_output(res_s, twin=False)
    res['lnZ'] = float(lnZ_single)
    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    return res, res_twin


def lnZ_BTP(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Tmag: float, Jmag: float, Hmag: float, Kmag: float,
            trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the BTP scenario (planet transits a background star).
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)

    # Background population and flux ratios (companion vs target)
    (Tmags_comp, masses_comp, loggs_comp, Teffs_comp, Zs_comp,
        Jmags_comp, Hmags_comp, Kmags_comp) = trilegal_results(trilegal_fname, Tmag)
    delta_T = Tmag - Tmags_comp
    delta_J = Jmag - Jmags_comp
    delta_H = Hmag - Hmags_comp
    delta_K = Kmag - Kmags_comp
    fluxratios_comp_T = 10**(delta_T/2.5) / (1 + 10**(delta_T/2.5))
    N_comp = Tmags_comp.shape[0]

    # Contrast curve (optional)
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = None, None

    # Preload LDC grids
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs; ldc_Teffs = ldc_T_Teffs; ldc_loggs = ldc_T_loggs; ldc_u1s = ldc_T_u1s; ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs; ldc_Teffs = ldc_K_Teffs; ldc_loggs = ldc_K_loggs; ldc_u1s = ldc_K_u1s; ldc_u2s = ldc_K_u2s

    # helpers
    def _delta_mag_for_idx(idx: int) -> float:
        if filt == "J":
            return float(delta_J[idx])
        elif filt == "H":
            return float(delta_H[idx])
        elif filt == "K":
            return float(delta_K[idx])
        else:
            return float(delta_T[idx])

    def lnprior_background_idx(idx: int) -> float:
        dm = _delta_mag_for_idx(idx)
        if dm > 0.0:
            return -np.inf
        if separations is None:
            lnpr = np.log10((N_comp/0.1) * (1/3600)**2 * 2.2**2)
            return float(0.0 if lnpr > 0.0 else lnpr)
        lnpr = float(lnprior_background(N_comp, np.array([abs(dm)]), separations, contrasts)[0])
        return float(0.0 if lnpr > 0.0 else lnpr)

    def _bg_props_and_ldc(idx: int) -> tuple:
        # radius from mass and logg; LDC from grids at (Z, Teff, logg)
        Mbg = float(masses_comp[idx])
        logg_bg = float(loggs_comp[idx])
        Rbg = float(np.sqrt(G*Mbg*Msun / (10**logg_bg)) / Rsun)
        Teff_bg = int(Teffs_comp[idx])
        Z_bg = float(Zs_comp[idx]) if Zs_comp is not None else 0.0
        this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff_bg))]
        this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg_bg))]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs-Z_bg))] if these_Zs.size > 0 else ldc_Zs[np.argmin(np.abs(ldc_Zs-Z_bg))]
        mask = ((ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg))
        u1b = float(ldc_u1s[mask]) if np.any(mask) else float(ldc_u1s[(ldc_Zs == this_Z)][0])
        u2b = float(ldc_u2s[mask]) if np.any(mask) else float(ldc_u2s[(ldc_Zs == this_Z)][0])
        return Mbg, Rbg, u1b, u2b

    # theta = [uP, uinc, uecc, uargp, urp, uidx]
    def prior_transform(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, urp, uidx = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=True, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        rp = float(sample_rp(np.array([urp]), np.array([M_s]), flatpriors)[0])
        idx = int(np.floor(np.clip(uidx, 1e-12, 1-1e-12) * N_comp))
        idx = min(idx, N_comp-1)
        return np.array([P, inc, ecc, argp_, rp, float(idx)])

    def loglike(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, rp, idxf = theta
        idx = int(idxf)
        Mbg, Rbg, u1b, u2b = _bg_props_and_ldc(idx)
        a = ((G*Mbg*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (rp*Rearth + Rbg*Rsun)/a * e_corr
        if Ptra > 1.0 or (rp*Rearth + Rbg*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        fr_host = float(fluxratios_comp_T[idx])  # background star flux fraction
        lnL = -0.5*ln2pi - lnsigma - lnL_TP(
            time, flux, sigma, rp, P, inc, a, Rbg, u1b, u2b,
            ecc, argp_, companion_fluxratio=fr_host, companion_is_host=True,
            exptime=exptime, nsamples=nsamples
        )
        lnL += lnprior_background_idx(idx)
        return float(lnL)

    lnZ, results = _run_persistent_evidence(
        loglike, prior_transform, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative output
    N_samples = POSTERIOR_NSAMPLES
    try:
        wsamps = _resample_equal(results.samples, results.weights, log_likelihoods=results.log_likelihoods)
        sel = slice(0, min(N_samples, wsamps.shape[0]))
        P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; rp = wsamps[:,4]; idxf = wsamps[:,5]
        idx = np.clip(np.floor(idxf).astype(int), 0, N_comp-1)
        # compute b using background star radius
        Mbg = masses_comp[idx]
        logg_bg = loggs_comp[idx]
        Rbg = np.sqrt(G*Mbg*Msun / (10**logg_bg)) / Rsun
        a_arr = ((G*Mbg*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        rsep = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
        b = rsep*np.cos(inc*pi/180)/(Rbg*Rsun)
        # LDCs per sample
        u1b = np.zeros_like(Rbg); u2b = np.zeros_like(Rbg)
        for i in range(Rbg.shape[0]):
            _, _, u1b[i], u2b[i] = _bg_props_and_ldc(int(idx[i]))
        fr_host = fluxratios_comp_T[idx]
        res = {
            'M_s': Mbg[sel],
            'R_s': Rbg[sel],
            'u1': u1b[sel],
            'u2': u2b[sel],
            'P_orb': P[sel],
            'inc': inc[sel],
            'b': b[sel],
            'R_p': rp[sel],
            'ecc': ecc[sel],
            'argp': argp_[sel],
            'M_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'R_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_comp': fr_host[sel],
            'lnZ': float(lnZ)
        }
    except Exception:
        res = {
            'M_s': np.array([M_s]), 'R_s': np.array([R_s]), 'u1': np.array([0.3]), 'u2': np.array([0.2]),
            'P_orb': np.array([P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
            'R_p': np.array([1.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
            'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]),
            'fluxratio_comp': np.array([0.0]), 'lnZ': float(lnZ)
        }
    res['best_lnL'] = float(np.max(results.log_likelihoods))
    return res


def lnZ_BEB(time: np.ndarray, flux: np.ndarray, sigma: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Tmag: float, Jmag:float, Hmag: float, Kmag: float,
            trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            N_twin: int = None, steps_twin: int = None):
    """
    Calculates the marginal likelihood of the BEB scenario (background eclipsing binary).
    Returns:
        res (dict), res_twin (dict)
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)

    # Background population
    (Tmags_comp, masses_comp, loggs_comp, Teffs_comp, Zs_comp,
        Jmags_comp, Hmags_comp, Kmags_comp) = trilegal_results(trilegal_fname, Tmag)
    delta_T = Tmag - Tmags_comp
    delta_J = Jmag - Jmags_comp
    delta_H = Hmag - Hmags_comp
    delta_K = Kmag - Kmags_comp
    fr_primary_T = 10**(delta_T/2.5) / (1 + 10**(delta_T/2.5))
    N_comp = Tmags_comp.shape[0]

    # Contrast curve (optional)
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = None, None

    # helpers
    def _delta_mag_primary_idx(idx: int) -> float:
        if filt == "J":
            return float(delta_J[idx])
        elif filt == "H":
            return float(delta_H[idx])
        elif filt == "K":
            return float(delta_K[idx])
        else:
            return float(delta_T[idx])

    def _ldc_for_bg(logg_bg: float, Teff_bg: float, Z_bg: float) -> tuple:
        if mission == "TESS":
            ldc_Zs = ldc_T_Zs; ldc_Teffs = ldc_T_Teffs; ldc_loggs = ldc_T_loggs; ldc_u1s = ldc_T_u1s; ldc_u2s = ldc_T_u2s
        else:
            ldc_Zs = ldc_K_Zs; ldc_Teffs = ldc_K_Teffs; ldc_loggs = ldc_K_loggs; ldc_u1s = ldc_K_u1s; ldc_u2s = ldc_K_u2s
        this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff_bg))]
        this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg_bg))]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs-Z_bg))] if these_Zs.size>0 else ldc_Zs[np.argmin(np.abs(ldc_Zs-Z_bg))]
        mask = ((ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg))
        u1b = float(ldc_u1s[mask]) if np.any(mask) else float(ldc_u1s[(ldc_Zs == this_Z)][0])
        u2b = float(ldc_u2s[mask]) if np.any(mask) else float(ldc_u2s[(ldc_Zs == this_Z)][0])
        return u1b, u2b

    def lnprior_background_combined(idx: int, Mbg: float, M2: float) -> float:
        dm_p = _delta_mag_primary_idx(idx)
        if dm_p > 0.0:
            return -np.inf
        if separations is None:
            lnpr = np.log10((N_comp/0.1) * (1/3600)**2 * 2.2**2)
            return float(0.0 if lnpr > 0.0 else lnpr)
        # compute combined delta mag in chosen filter
        if filt == "J":
            fr_p_cc = 10**(delta_J[idx]/2.5)/(1+10**(delta_J[idx]/2.5))
        elif filt == "H":
            fr_p_cc = 10**(delta_H[idx]/2.5)/(1+10**(delta_H[idx]/2.5))
        elif filt == "K":
            fr_p_cc = 10**(delta_K[idx]/2.5)/(1+10**(delta_K[idx]/2.5))
        else:
            fr_p_cc = 10**(delta_T[idx]/2.5)/(1+10**(delta_T[idx]/2.5))
        fr_p_bound_cc = float(flux_relation(np.array([Mbg]), filt)/(flux_relation(np.array([Mbg]), filt)+flux_relation(np.array([M_s]), filt)))
        fr_eb_bound_cc = float(flux_relation(np.array([M2]), filt)/(flux_relation(np.array([M2]), filt)+flux_relation(np.array([M_s]), filt)))
        distance_correction_cc = fr_p_cc / fr_p_bound_cc if fr_p_bound_cc > 0 else 0.0
        fr_eb_cc = fr_eb_bound_cc * distance_correction_cc
        combined = (fr_p_cc/(1-fr_p_cc)) + (fr_eb_cc/(1-fr_eb_cc))
        delta_mag_comb = 2.5*np.log10(combined)
        lnpr = float(lnprior_background(N_comp, np.array([abs(delta_mag_comb)]), separations, contrasts)[0])
        return float(0.0 if lnpr > 0.0 else lnpr)

    # theta = [uP, uinc, uecc, uargp, uq, uidx]
    def prior_transform_single(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, uq, uidx = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=False, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        q = float(sample_q(np.array([uq]), M_s)[0])
        idx = int(np.floor(np.clip(uidx, 1e-12, 1-1e-12) * N_comp))
        idx = min(idx, N_comp-1)
        return np.array([P, inc, ecc, argp_, q, float(idx)])

    def loglike_single(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, idxf = theta
        if q >= 0.95:
            return -np.inf
        idx = int(idxf)
        Mbg = float(masses_comp[idx])
        logg_bg = float(loggs_comp[idx])
        Rbg = float(np.sqrt(G*Mbg*Msun / (10**logg_bg)) / Rsun)
        Teff_bg = float(Teffs_comp[idx])
        Z_bg = float(Zs_comp[idx]) if Zs_comp is not None else 0.0
        u1b, u2b = _ldc_for_bg(logg_bg, Teff_bg, Z_bg)
        M2 = q*Mbg
        R2, _ = stellar_relations(np.array([M2]), np.array([Rbg]), np.array([Teff_bg]))
        R2 = float(R2[0])
        # flux ratios in TESS band, distance-corrected for EB companion
        fr_p_bound = float(flux_relation(np.array([Mbg]))/(flux_relation(np.array([Mbg]))+flux_relation(np.array([M_s]))))
        fr_p_obs = float(fr_primary_T[idx])
        fr_eb_bound = float(flux_relation(np.array([M2]))/(flux_relation(np.array([M2]))+flux_relation(np.array([M_s]))))
        distance_correction = (fr_p_obs / fr_p_bound) if fr_p_bound > 0 else 0.0
        fr_eb_obs = fr_eb_bound * distance_correction
        a = ((G*(Mbg+M2)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R2*Rsun + Rbg*Rsun)/a * e_corr
        if Ptra > 1.0 or (R2*Rsun + Rbg*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB(
            time, flux, sigma, R2, fr_eb_obs, P, inc, a, Rbg, u1b, u2b,
            ecc, argp_, companion_fluxratio=fr_p_obs, companion_is_host=True,
            exptime=exptime, nsamples=nsamples
        )
        lnL += lnprior_background_combined(idx, Mbg, M2)
        return float(lnL)

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, idxf = theta
        if q < 0.95:
            return -np.inf
        idx = int(idxf)
        Mbg = float(masses_comp[idx])
        logg_bg = float(loggs_comp[idx])
        Rbg = float(np.sqrt(G*Mbg*Msun / (10**logg_bg)) / Rsun)
        Teff_bg = float(Teffs_comp[idx])
        Z_bg = float(Zs_comp[idx]) if Zs_comp is not None else 0.0
        u1b, u2b = _ldc_for_bg(logg_bg, Teff_bg, Z_bg)
        M2 = q*Mbg
        R2, _ = stellar_relations(np.array([M2]), np.array([Rbg]), np.array([Teff_bg]))
        R2 = float(R2[0])
        fr_p_bound = float(flux_relation(np.array([Mbg]))/(flux_relation(np.array([Mbg]))+flux_relation(np.array([M_s]))))
        fr_p_obs = float(fr_primary_T[idx])
        fr_eb_bound = float(flux_relation(np.array([M2]))/(flux_relation(np.array([M2]))+flux_relation(np.array([M_s]))))
        distance_correction = (fr_p_obs / fr_p_bound) if fr_p_bound > 0 else 0.0
        fr_eb_obs = fr_eb_bound * distance_correction
        a = ((G*(Mbg+M2)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R2*Rsun + Rbg*Rsun)/a * e_corr
        if Ptra > 1.0 or (2*Rbg*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB_twin(
            time, flux, sigma, R2, fr_eb_obs, P, inc, a, Rbg, u1b, u2b,
            ecc, argp_, companion_fluxratio=fr_p_obs, companion_is_host=True,
            exptime=exptime, nsamples=nsamples
        )
        lnL += lnprior_background_combined(idx, Mbg, M2)
        return float(lnL)

    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )
    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N if N_twin is None else N_twin,
        target_ess=2 * (N if N_twin is None else N_twin),
        mcmc_steps=steps if steps_twin is None else steps_twin
    )

    # representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights, log_likelihoods=res_obj.log_likelihoods)
            sel = slice(0, min(N_samples, wsamps.shape[0]))
            P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; q = wsamps[:,4]; idxf = wsamps[:,5]
            idx = np.clip(np.floor(idxf).astype(int), 0, N_comp-1)
            Mbg = masses_comp[idx]
            logg_bg = loggs_comp[idx]
            Rbg = np.sqrt(G*Mbg*Msun / (10**logg_bg)) / Rsun
            Teff_bg = Teffs_comp[idx]
            Z_bg = Zs_comp[idx] if Zs_comp is not None else np.zeros_like(Mbg)
            # per-sample LDCs
            u1b = np.zeros_like(Mbg); u2b = np.zeros_like(Mbg)
            for i in range(Mbg.shape[0]):
                u1b[i], u2b[i] = _ldc_for_bg(float(logg_bg[i]), float(Teff_bg[i]), float(Z_bg[i]) if np.ndim(Z_bg)>0 else float(Z_bg))
            M2 = q*Mbg
            R2, _ = stellar_relations(M2, Rbg, Teff_bg)
            fr_p_obs = fr_primary_T[idx]
            fr_p_bound = flux_relation(Mbg)/(flux_relation(Mbg)+flux_relation(np.array([M_s])))
            fr_eb_bound = flux_relation(M2)/(flux_relation(M2)+flux_relation(np.array([M_s])))
            distance_correction = np.divide(fr_p_obs, fr_p_bound, out=np.zeros_like(fr_p_obs), where=fr_p_bound>0)
            fr_eb_obs = fr_eb_bound * distance_correction
            # The twin prior transform already changes P_fold to 2P.
            P_eff = P
            a_arr = ((G*(Mbg+M2)*Msun)/(4*pi**2)*(P_eff*86400)**2)**(1/3)
            rsep = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
            b = rsep*np.cos(inc*pi/180)/(Rbg*Rsun)
            out = {
                'M_s': Mbg[sel],
                'R_s': Rbg[sel],
                'u1': u1b[sel],
                'u2': u2b[sel],
                'P_orb': P_eff[sel],
                'inc': inc[sel],
                'b': b[sel],
                'R_p': np.zeros(min(N_samples, wsamps.shape[0])),
                'ecc': ecc[sel],
                'argp': argp_[sel],
                'M_EB': M2[sel],
                'R_EB': R2[sel],
                'fluxratio_EB': fr_eb_obs[sel],
                'fluxratio_comp': fr_p_obs[sel],
            }
        except Exception:
            out = {
                'M_s': np.array([M_s]), 'R_s': np.array([R_s]), 'u1': np.array([0.3]), 'u2': np.array([0.2]),
                'P_orb': np.array([2*P_orb_range[0] if twin else P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
                'R_p': np.array([0.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
                'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]),
                'fluxratio_comp': np.array([0.0]),
            }
        return out

    res = _build_output(res_s, twin=False)
    res['lnZ'] = float(lnZ_single)
    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    return res, res_twin


def lnZ_NTP_unknown(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, Tmag: float, trilegal_fname: str,
                    N: int = 10, steps: int = 20,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NTP scenario for
    a star of unknown properties using dynesty.
    Returns: res (dict)
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)

    # determine properties of possible stars near Tmag
    (Tmags_nearby, masses_nearby, loggs_nearby, Teffs_nearby,
        Zs_nearby, Jmags_nearby, Hmags_nearby, Kmags_nearby) = trilegal_results(trilegal_fname, Tmag)
    mask = (Tmag-1 < Tmags_nearby) & (Tmags_nearby < Tmag+1)
    Tmags_possible = Tmags_nearby[mask]
    masses_possible = masses_nearby[mask]
    loggs_possible = loggs_nearby[mask]
    Teffs_possible = Teffs_nearby[mask]
    Zs_possible = Zs_nearby[mask]
    radii_possible = np.sqrt(G*masses_possible*Msun / 10**loggs_possible) / Rsun
    N_possible = Tmags_possible.shape[0]
    if N_possible == 0:
        return {
            'M_s': 0, 'R_s': 0, 'u1': 0, 'u2': 0, 'P_orb': 0, 'inc': 0, 'R_p': 0,
            'ecc': 0, 'argp': 0, 'M_EB': 0, 'R_EB': 0, 'fluxratio_EB': 0,
            'fluxratio_comp': 0, 'lnZ': -np.inf
        }

    # LDC helper
    def _ldc_for_star(logg_s: float, Teff_s: float, Z_s: float) -> tuple:
        if mission == "TESS":
            ldc_Zs = ldc_T_Zs; ldc_Teffs = ldc_T_Teffs; ldc_loggs = ldc_T_loggs; ldc_u1s = ldc_T_u1s; ldc_u2s = ldc_T_u2s
        else:
            ldc_Zs = ldc_K_Zs; ldc_Teffs = ldc_K_Teffs; ldc_loggs = ldc_K_loggs; ldc_u1s = ldc_K_u1s; ldc_u2s = ldc_K_u2s
        this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff_s))]
        this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg_s))]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs-Z_s))] if these_Zs.size>0 else ldc_Zs[np.argmin(np.abs(ldc_Zs-Z_s))]
        mask = ((ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg))
        u1 = float(ldc_u1s[mask]) if np.any(mask) else float(ldc_u1s[(ldc_Zs == this_Z)][0])
        u2 = float(ldc_u2s[mask]) if np.any(mask) else float(ldc_u2s[(ldc_Zs == this_Z)][0])
        return u1, u2

    # theta = [uP, uinc, uecc, uargp, urp, uidx]
    def prior_transform(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, urp, uidx = u
        # choose star index
        idx = int(np.floor(np.clip(uidx, 1e-12, 1-1e-12) * N_possible))
        idx = min(idx, N_possible-1)
        Mstar = float(masses_possible[idx])
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=True, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        rp = float(sample_rp(np.array([urp]), np.array([Mstar]), flatpriors)[0])
        return np.array([P, inc, ecc, argp_, rp, float(idx)])

    def loglike(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, rp, idxf = theta
        idx = int(idxf)
        Mstar = float(masses_possible[idx])
        Rstar = float(radii_possible[idx])
        logg_s = float(loggs_possible[idx])
        Teff_s = float(Teffs_possible[idx])
        Z_s = float(Zs_possible[idx]) if Zs_possible is not None else 0.0
        if not (logg_s >= 3.5 and Teff_s <= 10000):
            return -np.inf
        u1, u2 = _ldc_for_star(logg_s, Teff_s, Z_s)
        a = ((G*Mstar*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (rp*Rearth + Rstar*Rsun)/a * e_corr
        if Ptra > 1.0 or (rp*Rearth + Rstar*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_TP(
            time, flux, sigma, rp, P, inc, a, Rstar, u1, u2,
            ecc, argp_, exptime=exptime, nsamples=nsamples
        )
        return float(lnL)

    lnZ, res_obj = _run_persistent_evidence(
        loglike, prior_transform, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative outputs
    N_samples = POSTERIOR_NSAMPLES
    try:
        wsamps = _resample_equal(res_obj.samples, res_obj.weights, log_likelihoods=res_obj.log_likelihoods)
        sel = slice(0, min(N_samples, wsamps.shape[0]))
        P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; rp = wsamps[:,4]; idxf = wsamps[:,5]
        idx = np.clip(np.floor(idxf).astype(int), 0, N_possible-1)
        Mstar = masses_possible[idx]
        Rstar = radii_possible[idx]
        logg_s = loggs_possible[idx]
        Teff_s = Teffs_possible[idx]
        Z_s = Zs_possible[idx] if Zs_possible is not None else np.zeros_like(Mstar)
        u1 = np.zeros_like(Mstar); u2 = np.zeros_like(Mstar)
        for i in range(Mstar.shape[0]):
            u1[i], u2[i] = _ldc_for_star(float(logg_s[i]), float(Teff_s[i]), float(Z_s[i]) if np.ndim(Z_s)>0 else float(Z_s))
        a_arr = ((G*Mstar*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        rsep = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
        b = rsep*np.cos(inc*pi/180)/(Rstar*Rsun)
        res = {
            'M_s': Mstar[sel],
            'R_s': Rstar[sel],
            'u1': u1[sel],
            'u2': u2[sel],
            'P_orb': P[sel],
            'inc': inc[sel],
            'b': b[sel],
            'R_p': rp[sel],
            'ecc': ecc[sel],
            'argp': argp_[sel],
            'M_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'R_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_comp': np.zeros(min(N_samples, wsamps.shape[0])),
            'lnZ': float(lnZ)
        }
    except Exception:
        res = {
            'M_s': np.array([0]), 'R_s': np.array([0]), 'u1': np.array([0]), 'u2': np.array([0]),
            'P_orb': np.array([P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
            'R_p': np.array([0.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
            'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]),
            'fluxratio_comp': np.array([0.0]), 'lnZ': float(lnZ)
        }
    return res


def lnZ_NEB_unknown(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, Tmag: float, trilegal_fname: str,
                    N: int = 10, steps: int = 20,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NEB scenario for a star
    of unknown properties.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        Tmag (float): Target star TESS magnitude.
        trilegal_fname (string): File containing trilegal query results.
        N (int): Number of particles.
        steps (int): Number of MCMC steps.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)

    # possible stars near Tmag
    (Tmags_nearby, masses_nearby, loggs_nearby, Teffs_nearby,
        Zs_nearby, Jmags_nearby, Hmags_nearby, Kmags_nearby) = trilegal_results(trilegal_fname, Tmag)
    mask = (Tmag-1 < Tmags_nearby) & (Tmags_nearby < Tmag+1)
    Tmags_possible = Tmags_nearby[mask]
    masses_possible = masses_nearby[mask]
    loggs_possible = loggs_nearby[mask]
    Teffs_possible = Teffs_nearby[mask]
    Zs_possible = Zs_nearby[mask]
    radii_possible = np.sqrt(G*masses_possible*Msun / 10**loggs_possible) / Rsun
    N_possible = Tmags_possible.shape[0]
    if N_possible == 0:
        res = {'M_s': 0,'R_s': 0,'u1': 0,'u2': 0,'P_orb': 0,'inc': 0,'b': 0,'R_p': 0,'ecc': 0,'argp': 0,'M_EB': 0,'R_EB': 0,'fluxratio_EB': 0,'fluxratio_comp': 0,'lnZ': -np.inf}
        return res, res

    # LDC helper
    def _ldc_for_star(logg_s: float, Teff_s: float, Z_s: float) -> tuple:
        if mission == "TESS":
            ldc_Zs = ldc_T_Zs; ldc_Teffs = ldc_T_Teffs; ldc_loggs = ldc_T_loggs; ldc_u1s = ldc_T_u1s; ldc_u2s = ldc_T_u2s
        else:
            ldc_Zs = ldc_K_Zs; ldc_Teffs = ldc_K_Teffs; ldc_loggs = ldc_K_loggs; ldc_u1s = ldc_K_u1s; ldc_u2s = ldc_K_u2s
        this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff_s))]
        this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg_s))]
        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs-Z_s))] if these_Zs.size>0 else ldc_Zs[np.argmin(np.abs(ldc_Zs-Z_s))]
        mask = ((ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg))
        u1 = float(ldc_u1s[mask]) if np.any(mask) else float(ldc_u1s[(ldc_Zs == this_Z)][0])
        u2 = float(ldc_u2s[mask]) if np.any(mask) else float(ldc_u2s[(ldc_Zs == this_Z)][0])
        return u1, u2

    # theta = [uP, uinc, uecc, uargp, uq, uidx]
    def prior_transform_single(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, uq, uidx = u
        idx = int(np.floor(np.clip(uidx, 1e-12, 1-1e-12) * N_possible))
        idx = min(idx, N_possible-1)
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=False, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        q = float(sample_q(np.array([uq]), 1.0)[0])
        return np.array([P, inc, ecc, argp_, q, float(idx)])

    def loglike_single(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, idxf = theta
        if q >= 0.95:
            return -np.inf
        idx = int(idxf)
        M1 = float(masses_possible[idx])
        R1 = float(radii_possible[idx])
        logg_s = float(loggs_possible[idx])
        Teff_s = float(Teffs_possible[idx])
        Z_s = float(Zs_possible[idx]) if Zs_possible is not None else 0.0
        if not (logg_s >= 3.5 and Teff_s <= 10000):
            return -np.inf
        u1, u2 = _ldc_for_star(logg_s, Teff_s, Z_s)
        M2 = q*M1
        R2, _ = stellar_relations(np.array([M2]), np.array([R1]), np.array([Teff_s]))
        R2 = float(R2[0])
        # EB flux ratio in same system (no third light)
        fr_eb = float(flux_relation(np.array([M2]))/(flux_relation(np.array([M2]))+flux_relation(np.array([M1]))))
        a = ((G*(M1+M2)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R2*Rsun + R1*Rsun)/a * e_corr
        if Ptra > 1.0 or (R2*Rsun + R1*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB(
            time, flux, sigma, R2, fr_eb, P, inc, a, R1, u1, u2,
            ecc, argp_, companion_fluxratio=0.0, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        return float(lnL)

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, idxf = theta
        if q < 0.95:
            return -np.inf
        idx = int(idxf)
        M1 = float(masses_possible[idx])
        R1 = float(radii_possible[idx])
        logg_s = float(loggs_possible[idx])
        Teff_s = float(Teffs_possible[idx])
        Z_s = float(Zs_possible[idx]) if Zs_possible is not None else 0.0
        if not (logg_s >= 3.5 and Teff_s <= 10000):
            return -np.inf
        u1, u2 = _ldc_for_star(logg_s, Teff_s, Z_s)
        M2 = q*M1
        R2, _ = stellar_relations(np.array([M2]), np.array([R1]), np.array([Teff_s]))
        R2 = float(R2[0])
        fr_eb = float(flux_relation(np.array([M2]))/(flux_relation(np.array([M2]))+flux_relation(np.array([M1]))))
        a = ((G*(M1+M2)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R2*Rsun + R1*Rsun)/a * e_corr
        if Ptra > 1.0 or (2*R1*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB_twin(
            time, flux, sigma, R2, fr_eb, P, inc, a, R1, u1, u2,
            ecc, argp_, companion_fluxratio=0.0, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        return float(lnL)

    lnZ_s, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )
    lnZ_t, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    def _build_output(res_obj, twin: bool=False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights, log_likelihoods=res_obj.log_likelihoods)
            sel = slice(0, min(N_samples, wsamps.shape[0]))
            P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; q = wsamps[:,4]; idxf = wsamps[:,5]
            idx = np.clip(np.floor(idxf).astype(int), 0, N_possible-1)
            M1 = masses_possible[idx]; R1 = radii_possible[idx]
            logg_s = loggs_possible[idx]; Teff_s = Teffs_possible[idx]
            Z_s = Zs_possible[idx] if Zs_possible is not None else np.zeros_like(M1)
            u1 = np.zeros_like(M1); u2 = np.zeros_like(M1)
            for i in range(M1.shape[0]):
                u1[i], u2[i] = _ldc_for_star(float(logg_s[i]), float(Teff_s[i]), float(Z_s[i]) if np.ndim(Z_s)>0 else float(Z_s))
            M2 = q*M1
            R2, _ = stellar_relations(M2, R1, Teff_s)
            fr_eb = flux_relation(M2)/(flux_relation(M2)+flux_relation(M1))
            # The twin prior transform already changes P_fold to 2P.
            P_eff = P
            a_arr = ((G*(M1+M2)*Msun)/(4*pi**2)*(P_eff*86400)**2)**(1/3)
            rsep = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
            b = rsep*np.cos(inc*pi/180)/(R1*Rsun)
            out = {
                'M_s': M1[sel], 'R_s': R1[sel], 'u1': u1[sel], 'u2': u2[sel],
                'P_orb': P_eff[sel], 'inc': inc[sel], 'b': b[sel],
                'R_p': np.zeros(min(N_samples, wsamps.shape[0])),
                'ecc': ecc[sel], 'argp': argp_[sel],
                'M_EB': M2[sel], 'R_EB': R2[sel],
                'fluxratio_EB': fr_eb[sel], 'fluxratio_comp': np.zeros(min(N_samples, wsamps.shape[0])),
            }
        except Exception:
            out = {
                'M_s': np.array([0]), 'R_s': np.array([0]), 'u1': np.array([0]), 'u2': np.array([0]),
                'P_orb': np.array([2*P_orb_range[0] if twin else P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
                'R_p': np.array([0.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
                'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]), 'fluxratio_comp': np.array([0.0]),
            }
        return out

    res = _build_output(res_s, twin=False); res['lnZ'] = float(lnZ_s)
    res_twin = _build_output(res_t, twin=True); res_twin['lnZ'] = float(lnZ_t)
    return res, res_twin


def lnZ_NTP_evolved(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, R_s: float, Teff: float, Z: float,
                    N: int = 10, steps: int = 20,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NTP scenario for evolved
    (subgiant) hosts using dynesty. Assumes logg~3 to infer M_s from R_s,
    then follows the same prior/likelihood structure as lnZ_TTP.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of particles.
        steps (int): Number of MCMC steps.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)
    logg = 3.0
    M_s = (10**logg)*(R_s*Rsun)**2 / G / Msun
    # target star LDCs
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs; ldc_Teffs = ldc_T_Teffs; ldc_loggs = ldc_T_loggs; ldc_u1s = ldc_T_u1s; ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs; ldc_Teffs = ldc_K_Teffs; ldc_loggs = ldc_K_loggs; ldc_u1s = ldc_K_u1s; ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = ((ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg))
    u1, u2 = float(ldc_u1s[mask]), float(ldc_u2s[mask])

    # theta = [uP, uinc, uecc, uargp, urp]
    def prior_transform(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, urp = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=True, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        rp = float(sample_rp(np.array([urp]), np.array([M_s]), flatpriors)[0])
        return np.array([P, inc, ecc, argp_, rp])

    def loglike(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, rp = theta
        a = ((G*M_s*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (rp*Rearth + R_s*Rsun)/a * e_corr
        if Ptra > 1.0 or (rp*Rearth + R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_TP(
            time, flux, sigma, rp, P, inc, a, R_s, u1, u2,
            ecc, argp_, exptime=exptime, nsamples=nsamples
        )
        return float(lnL)

    lnZ, res_obj = _run_persistent_evidence(
        loglike, prior_transform, ndim=5,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative outputs
    N_samples = POSTERIOR_NSAMPLES
    try:
        wsamps = _resample_equal(res_obj.samples, res_obj.weights, log_likelihoods=res_obj.log_likelihoods)
        sel = slice(0, min(N_samples, wsamps.shape[0]))
        P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; rp = wsamps[:,4]
        a_arr = ((G*M_s*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        rsep = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
        b = rsep*np.cos(inc*pi/180)/(R_s*Rsun)
        res = {
            'M_s': np.full(min(N_samples, wsamps.shape[0]), M_s),
            'R_s': np.full(min(N_samples, wsamps.shape[0]), R_s),
            'u1': np.full(min(N_samples, wsamps.shape[0]), u1),
            'u2': np.full(min(N_samples, wsamps.shape[0]), u2),
            'P_orb': P[sel], 'inc': inc[sel], 'b': b[sel], 'R_p': rp[sel],
            'ecc': ecc[sel], 'argp': argp_[sel],
            'M_EB': np.zeros(min(N_samples, wsamps.shape[0])), 'R_EB': np.zeros(min(N_samples, wsamps.shape[0])),
            'fluxratio_EB': np.zeros(min(N_samples, wsamps.shape[0])), 'fluxratio_comp': np.zeros(min(N_samples, wsamps.shape[0])),
            'lnZ': float(lnZ)
        }
    except Exception:
        res = {
            'M_s': np.array([M_s]), 'R_s': np.array([R_s]), 'u1': np.array([u1]), 'u2': np.array([u2]),
            'P_orb': np.array([P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
            'R_p': np.array([0.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
            'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]),
            'fluxratio_comp': np.array([0.0]), 'lnZ': float(lnZ)
        }
    return res


def lnZ_NEB_evolved(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, R_s: float, Teff: float, Z: float,
                    N: int = 10, steps: int = 20,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NEB scenario
    for subgiant stars.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        P_orb (float): Orbital period [days].
        R_s (float): Target star radius [Solar radii].
        Teff (float): Target star effective temperature [K].
        Z (float): Target star metallicity [dex].
        N (int): Number of particles.
        steps (int): Number of MCMC steps.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(sigma)
    logg = 3.0
    M_s = (10**logg)*(R_s*Rsun)**2 / G / Msun
    # target star LDCs
    if mission == "TESS":
        ldc_Zs = ldc_T_Zs; ldc_Teffs = ldc_T_Teffs; ldc_loggs = ldc_T_loggs; ldc_u1s = ldc_T_u1s; ldc_u2s = ldc_T_u2s
    else:
        ldc_Zs = ldc_K_Zs; ldc_Teffs = ldc_K_Teffs; ldc_loggs = ldc_K_loggs; ldc_u1s = ldc_K_u1s; ldc_u2s = ldc_K_u2s
    this_Z = ldc_Zs[np.argmin(np.abs(ldc_Zs-Z))]
    this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teff))]
    this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-logg))]
    mask = ((ldc_Zs == this_Z) & (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg))
    u1, u2 = float(ldc_u1s[mask]), float(ldc_u2s[mask])

    # theta = [uP, uinc, uecc, uargp, uq]
    def prior_transform_single(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, uq = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=False, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        q = float(sample_q(np.array([uq]), 1.0)[0])
        return np.array([P, inc, ecc, argp_, q])

    def loglike_single(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q = theta
        if q >= 0.95:
            return -np.inf
        M2 = q*M_s
        R2, _ = stellar_relations(np.array([M2]), np.array([R_s]), np.array([Teff]))
        R2 = float(R2[0])
        fr_eb = float(flux_relation(np.array([M2]))/(flux_relation(np.array([M2]))+flux_relation(np.array([M_s]))))
        a = ((G*(M_s+M2)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R2*Rsun + R_s*Rsun)/a * e_corr
        if Ptra > 1.0 or (R2*Rsun + R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB(
            time, flux, sigma, R2, fr_eb, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=0.0, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        return float(lnL)

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q = theta
        if q < 0.95:
            return -np.inf
        M2 = q*M_s
        R2, _ = stellar_relations(np.array([M2]), np.array([R_s]), np.array([Teff]))
        R2 = float(R2[0])
        fr_eb = float(flux_relation(np.array([M2]))/(flux_relation(np.array([M2]))+flux_relation(np.array([M_s]))))
        a = ((G*(M_s+M2)*Msun)/(4*pi**2)*(P*86400)**2)**(1/3)
        e_corr = (1+ecc*np.sin(argp_*pi/180))/(1-ecc**2)
        Ptra = (R2*Rsun + R_s*Rsun)/a * e_corr
        if Ptra > 1.0 or (2*R_s*Rsun) > a*(1-ecc):
            return -np.inf
        inc_min = np.degrees(np.arccos(min(1.0, Ptra)))
        if inc < inc_min:
            return -np.inf
        lnL = -0.5*ln2pi - lnsigma - lnL_EB_twin(
            time, flux, sigma, R2, fr_eb, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=0.0, companion_is_host=False,
            exptime=exptime, nsamples=nsamples
        )
        return float(lnL)

    #lnZ_t, res_t = _run_dynesty_evidence(loglike_twin, prior_transform_twin, ndim=5, nlive=nlive, dlogz=dlogz, dynamic=dynamic)
    lnZ_s, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=5,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )
    lnZ_t, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=5,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    def _build_output(res_obj, twin: bool=False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights, log_likelihoods=res_obj.log_likelihoods)
            sel = slice(0, min(N_samples, wsamps.shape[0]))
            P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; q = wsamps[:,4]
            M2 = q*M_s
            R2, _ = stellar_relations(M2, np.full_like(M2, R_s), np.full_like(M2, Teff))
            fr_eb = flux_relation(M2)/(flux_relation(M2)+flux_relation(np.array([M_s])))
            # The twin prior transform already changes P_fold to 2P.
            P_eff = P
            a_arr = ((G*(M_s+M2)*Msun)/(4*pi**2)*(P_eff*86400)**2)**(1/3)
            rsep = a_arr*(1-ecc**2)/(1+ecc*np.sin(argp_*np.pi/180))
            b = rsep*np.cos(inc*pi/180)/(R_s*Rsun)
            out = {
                'M_s': np.full(min(N_samples, wsamps.shape[0]), M_s), 'R_s': np.full(min(N_samples, wsamps.shape[0]), R_s),
                'u1': np.full(min(N_samples, wsamps.shape[0]), u1), 'u2': np.full(min(N_samples, wsamps.shape[0]), u2),
                'P_orb': P_eff[sel], 'inc': inc[sel], 'b': b[sel],
                'R_p': np.zeros(min(N_samples, wsamps.shape[0])),
                'ecc': ecc[sel], 'argp': argp_[sel],
                'M_EB': M2[sel], 'R_EB': R2[sel],
                'fluxratio_EB': fr_eb[sel], 'fluxratio_comp': np.zeros(min(N_samples, wsamps.shape[0])),
            }
        except Exception:
            out = {
                'M_s': np.array([M_s]), 'R_s': np.array([R_s]), 'u1': np.array([u1]), 'u2': np.array([u2]),
                'P_orb': np.array([2*P_orb_range[0] if twin else P_orb_range[0]]), 'inc': np.array([90.0]), 'b': np.array([0.0]),
                'R_p': np.array([0.0]), 'ecc': np.array([0.0]), 'argp': np.array([90.0]),
                'M_EB': np.array([0.0]), 'R_EB': np.array([0.0]), 'fluxratio_EB': np.array([0.0]), 'fluxratio_comp': np.array([0.0]),
            }
        return out

    res = _build_output(res_s, twin=False); res['lnZ'] = float(lnZ_s)
    res_twin = _build_output(res_t, twin=True); res_twin['lnZ'] = float(lnZ_t)
    return res, res_twin


# Package-only observability; no extra random draws or changes to evidence.
from ..sampling.records import instrument_evidence as _instrument_evidence
_instrument_evidence(globals())
