"""Scenario evidences computed with persistent sampling.

Extracted from the corrected research engine; numerical logic is unchanged.
N is the active particle count and steps is the MCMC effort.
See docs/architecture.md for migration limitations.
"""

from ..models.hosts import (
    Star as _Star,
    KnownHost as _KnownHost,
    BoundCompanionHost as _BoundCompanionHost,
    DilutedBoundHost as _DilutedBoundHost,
)
from ..models.populations import (
    BackgroundHost as _BackgroundHost,
    UnknownHost as _UnknownHost,
)
from ..models.systems import (
    Scenario as _Scenario,
    Planet as _Planet,
    Binary as _Binary,
    OrbitPolicy as _OrbitPolicy,
)
from .scenario import (
    ScenarioPrior as _ScenarioPrior,
    ScenarioLikelihood as _ScenarioLikelihood,
)

import numpy as np
from ..models.systems import (
    period_range as _period_range,
)
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
    return float(np.log(np.mean(values)))


def _sigma_for_observations(sigma, flux, label):
    """Broadcast a scalar error or validate an existing error vector."""
    values = np.atleast_1d(np.asarray(sigma, dtype=float)).reshape(-1)
    size = np.asarray(flux).size
    if values.size == 1:
        return np.full(size, float(values[0]))
    if values.size != size:
        raise ValueError(
            f"{label} uncertainty has length {values.size}; expected {size}"
        )
    return values

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

# The optional ``secondary_loglike`` arguments below are an injection point
# for ``triceratops_new``. Their default is the historical likelihood, so
# direct callers and the intermediate ``triceratops`` snapshot retain their
# previous behaviour.


def _resample_equal(samples, weights, n_samples=None):
    """
    Resample samples with replacement according to weights.
    Similar to dynesty.utils.resample_equal.
    """
    if n_samples is None:
        n_samples = len(samples)
    indices = np.random.choice(len(samples), size=n_samples, p=weights)
    return samples[indices]

def _run_persistent_evidence(
    loglike: Callable[[np.ndarray], float],
    prior_transform: Callable[[np.ndarray], np.ndarray],
    ndim: int,
    n_active: int = 100,
    target_ess: Optional[float] = None,
    mcmc_steps: int = 20,
    loglike_batched: Optional[Callable] = None,
    prior_transform_batched: Optional[Callable] = None,
) -> Tuple[float, object]:
    """
    Run persistent sampling to estimate ln(Z).
    Returns (lnZ, results_object).

    Optional batched callables enable vectorised likelihood eval in the PS
    initialisation and MCMC mutation phases.
    """
    if target_ess is None:
        target_ess = 2 * n_active

    log_Z, weights, positions, log_likelihoods = persistent_sampling(
        log_likelihood_function=loglike,
        prior_transform=prior_transform,
        n_active=n_active,
        ndim=ndim,
        target_ess=target_ess,
        mcmc_steps=mcmc_steps,
        log_likelihood_function_batched=loglike_batched,
        prior_transform_batched=prior_transform_batched,
    )
    
    # Create a mock results object similar to nestle/dynesty
    class MockResults:
        def __init__(self, samples, weights, logz, log_likelihoods):
            self.samples = samples
            self.weights = weights
            self.logz = np.array([logz])  # Make it an array like nestle
            self.log_likelihoods = log_likelihoods

    results = MockResults(positions, weights, log_Z, log_likelihoods)
    return log_Z, results


def _inv_sample_ecc_vec(u_arr, planet: bool, P_orb: float):
    """Vectorised inverse-CDF sampler using scipy.stats.*.ppf on arrays
    (much faster than calling ppf per scalar — amortises Python overhead
    across all N particles in one call)."""
    from scipy.stats import beta as _beta
    from scipy.stats import powerlaw as _powerlaw
    u_arr = np.clip(np.asarray(u_arr, dtype=float), 1e-12, 1.0 - 1e-12)
    if planet:
        return np.asarray(_beta.ppf(u_arr, 0.867, 3.030), dtype=float)
    a_shape = 0.6 if P_orb > 10 else 0.2
    return np.asarray(_powerlaw.ppf(u_arr, a_shape), dtype=float)


def _inv_sample_ecc(u: float, planet: bool, P_orb: float) -> float:
    """Scalar wrapper — same numerical path as _inv_sample_ecc_vec so scalar
    and batched outputs are bit-identical (avoids PS path divergence)."""
    return float(_inv_sample_ecc_vec(np.array([float(u)]), planet, P_orb)[0])


def lnZ_TEB_secondary(time: np.ndarray, flux: np.ndarray, sigma: float,time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            secondary_loglike: Callable = lnL_EB_second):
    """
    Calculates the marginal likelihood of the TEB scenario with persistent sampling.
    
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
        N (int): Number of active particles.
        steps (int): Number of steps for MCMC.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # Shared physical ingredients; this wrapper retains the evidence frame.
    P_orb_range = _period_range(P_orb)
    ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS" else
        (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    host = _KnownHost.prepare(M_s, R_s, Teff, Z, ldc)
    u1, u2 = host.target.u1, host.target.u2
    model = _Scenario(host, _Binary(), P_orb_range)
    norm_const = -0.5*ln2pi - _log_sigma_norm(sigma)
    prior = _ScenarioPrior(model, _inv_sample_ecc)
    loglike = _ScenarioLikelihood(
        model=model, time=time, data=flux, noise=sigma,
        normalization=norm_const, residual_cost=secondary_loglike,
        exptime=exptime, nsamples=nsamples, domain="real",
        secondary=(time_secondary, flux_secondary, sigma_secondary),
    )
    lnZ_single, res_s = _run_persistent_evidence(
        loglike, prior, ndim=model.ndim,
        n_active=N, target_ess=2*N, mcmc_steps=steps,
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
    res['best_lnL'] = float(np.max(res_s.log_likelihoods))
    return res

def lnZ_TEB_evenodd(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the TEB scenario with dynesty.
    Fits even/odd windows at twice the candidate period, without a twin-q cut.
    Eccentricity is drawn at the candidate period before doubling.
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
        N (int): Number of active particles.
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

    lnsigma = _log_sigma_norm(
        _sigma_for_observations(sigma_even, flux_even, "even"),
        _sigma_for_observations(sigma_odd, flux_odd, "odd"),
    )
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

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _KnownHost(_Star(M_s, R_s, Teff, u1, u2))
    _model_loglike_twin = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=None),
        P_orb_range,
        _OrbitPolicy(
            period_factor=2.0,
            doubled_host_collision=True,
            q_branch="all",
            observation_kind="evenodd",
        ),
    )
    prior_transform_twin = _ScenarioPrior(_model_loglike_twin, _inv_sample_ecc)
    loglike_twin = _ScenarioLikelihood(
        model=_model_loglike_twin,
        time=time_even,
        data=flux_even,
        noise=sigma_even,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_EB_evenodd,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
        secondary=(time_odd, flux_odd, sigma_odd),
    )

    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=5,
        n_active=N, target_ess=2 * N, mcmc_steps=steps,
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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

    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    res_twin['best_lnL'] = float(np.max(res_t.log_likelihoods))
    return  res_twin

def lnZ_PEB_secondary(time: np.ndarray, flux: np.ndarray, sigma: float,
                      time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None,
            secondary_loglike: Callable = lnL_EB_second):
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
        N (int): Number of active particles.
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

    # Precompute MOLUSC q_comp pool, if provided
    molusc_qs = None
    if molusc_file is not None:
        molusc_df = read_csv(molusc_file)
        molusc_a = molusc_df["semi-major axis(AU)"].values
        molusc_e = molusc_df["eccentricity"].values
        molusc_df2 = molusc_df[molusc_a*(1-molusc_e) > 10]
        molusc_qs = molusc_df2["mass ratio"].values
        molusc_qs[molusc_qs < 0.1/M_s] = 0.1/M_s


    # Preload contrast curve data if provided
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = np.array([2.2]), np.array([1.0])


    # dynesty prior/likelihood for single and twin branches
    # theta = [uP, uinc, uecc, uargp, uq, uqcomp]


    # Run nested sampling for both branches
    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _DilutedBoundHost(
        _KnownHost(_Star(M_s, R_s, Teff, u1, u2)),
        _BoundCompanionHost.from_prepared(
            _Star(M_s, R_s, Teff, None, None),
            Z,
            _ldc,
            plx,
            separations,
            contrasts,
            None if contrast_curve_file is None else filt,
            molusc_qs,
            ldc_rule="nearest",
        ),
    )
    _model_loglike_single = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=None),
        P_orb_range,
        _OrbitPolicy(
            period_factor=1.0,
            doubled_host_collision=False,
            q_branch="all",
            observation_kind="standard",
        ),
    )
    prior_transform_single = _ScenarioPrior(_model_loglike_single, _inv_sample_ecc)
    loglike_single = _ScenarioLikelihood(
        model=_model_loglike_single,
        time=time,
        data=flux,
        noise=sigma,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=secondary_loglike,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
        secondary=(time_secondary, flux_secondary, sigma_secondary),
    )

    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6, n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
    res['best_lnL'] = float(np.max(res_s.log_likelihoods))
    return res

def lnZ_PEB_evenodd(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,
                      time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None):
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
        N (int): Number of active particles.
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

    lnsigma = _log_sigma_norm(
        _sigma_for_observations(sigma_even, flux_even, "even"),
        _sigma_for_observations(sigma_odd, flux_odd, "odd"),
    )
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


    # Preload contrast curve data if provided
    if contrast_curve_file is not None:
        separations, contrasts = file_to_contrast_curve(contrast_curve_file)
    else:
        separations, contrasts = np.array([2.2]), np.array([1.0])


    # dynesty prior/likelihood for single and twin branches
    # theta = [uP, uinc, uecc, uargp, uq, uqcomp]

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _DilutedBoundHost(
        _KnownHost(_Star(M_s, R_s, Teff, u1, u2)),
        _BoundCompanionHost.from_prepared(
            _Star(M_s, R_s, Teff, None, None),
            Z,
            _ldc,
            plx,
            separations,
            contrasts,
            None if contrast_curve_file is None else filt,
            molusc_qs,
            ldc_rule="nearest",
        ),
    )
    _model_loglike_twin = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=None),
        P_orb_range,
        _OrbitPolicy(
            period_factor=2.0,
            doubled_host_collision=True,
            q_branch="all",
            observation_kind="evenodd",
        ),
    )
    prior_transform_twin = _ScenarioPrior(_model_loglike_twin, _inv_sample_ecc)
    loglike_twin = _ScenarioLikelihood(
        model=_model_loglike_twin,
        time=time_even,
        data=flux_even,
        noise=sigma_even,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_EB_evenodd,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
        secondary=(time_odd, flux_odd, sigma_odd),
    )

    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6, n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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

    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    res_twin['best_lnL'] = float(np.max(res_t.log_likelihoods))
    return  res_twin

def lnZ_SEB_secondary(time: np.ndarray, flux: np.ndarray, sigma: float,
                      time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
        P_orb: float, M_s: float, R_s: float, Teff: float,
        Z: float, plx: float, contrast_curve_file: str = None,
        filt: str = "TESS",
        N: int = 10, steps: int = 20,
        mission: str = "TESS", flatpriors: bool = False,
        exptime: float = 0.00139, nsamples: int = 20,
        molusc_file: str = None,
        secondary_loglike: Callable = lnL_EB_second):
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
        N (int): Number of active particles.
        steps (int): Number of steps for MCMC.
        mission (str): TESS, Kepler, or K2.
        flatpriors (bool): Assume flat Rp and Porb planet priors?
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        res (dict): Best-fit properties and marginal likelihood.
    """
    # Shared physical ingredients; this wrapper retains the evidence frame.
    P_orb_range = _period_range(P_orb)
    ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS" else
        (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    host = _BoundCompanionHost.prepare(
        M_s, R_s, Teff, Z, ldc, plx, contrast_curve_file, filt, molusc_file,
        ldc_rule="nearest",
    )
    model = _Scenario(host, _Binary(), P_orb_range)
    norm_const = -0.5*ln2pi - _log_sigma_norm(sigma)
    prior = _ScenarioPrior(model, _inv_sample_ecc)
    loglike = _ScenarioLikelihood(
        model=model, time=time, data=flux, noise=sigma,
        normalization=norm_const, residual_cost=secondary_loglike,
        exptime=exptime, nsamples=nsamples, domain="real",
        secondary=(time_secondary, flux_secondary, sigma_secondary),
    )
    lnZ_single, res_s = _run_persistent_evidence(
        loglike, prior, ndim=model.ndim,
        n_active=N, target_ess=2*N, mcmc_steps=steps,
    )
    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
            P_eff = P  # prior_transform_twin already sets P=2*P_fold
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
    res['best_lnL'] = float(np.max(res_s.log_likelihoods))
    return res

def lnZ_SEB_evenodd(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,
        time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
        P_orb: float, M_s: float, R_s: float, Teff: float,
        Z: float, plx: float, contrast_curve_file: str = None,
        filt: str = "TESS",
        N: int = 10, steps: int = 20,
        mission: str = "TESS", flatpriors: bool = False,
        exptime: float = 0.00139, nsamples: int = 20,
        molusc_file: str = None):
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
        N (int): Number of active particles.
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

    lnsigma = _log_sigma_norm(
        _sigma_for_observations(sigma_even, flux_even, "even"),
        _sigma_for_observations(sigma_odd, flux_odd, "odd"),
    )

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


    # limb darkening accessor based on companion mass


    # theta = [uP, uinc, uecc, uargp, uq, uqcomp]

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _BoundCompanionHost.from_prepared(
        _Star(M_s, R_s, Teff, None, None),
        Z,
        _ldc,
        plx,
        separations,
        contrasts,
        None if contrast_curve_file is None else filt,
        molusc_qs,
        ldc_rule="nearest",
    )
    _model_loglike_twin = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=None),
        P_orb_range,
        _OrbitPolicy(
            period_factor=2.0,
            doubled_host_collision=True,
            q_branch="all",
            observation_kind="evenodd",
        ),
    )
    prior_transform_twin = _ScenarioPrior(_model_loglike_twin, _inv_sample_ecc)
    loglike_twin = _ScenarioLikelihood(
        model=_model_loglike_twin,
        time=time_even,
        data=flux_even,
        noise=sigma_even,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_EB_evenodd,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
        secondary=(time_odd, flux_odd, sigma_odd),
    )

    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
            P_eff = P  # prior_transform_twin already sets P=2*P_fold
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
    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    res_twin['best_lnL'] = float(np.max(res_t.log_likelihoods))
    return  res_twin


def lnZ_DEB_secondary(time: np.ndarray, flux: np.ndarray, sigma: float,
                      time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, Tmag: float, Jmag: float, Hmag: float,
            Kmag: float, trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            secondary_loglike: Callable = lnL_EB_second):
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
        N (int): Number of active particles.
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


    # theta = [uP, uinc, uecc, uargp, uq, uidx]

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _BackgroundHost.prepare(
        _Star(M_s, R_s, Teff, u1, u2),
        masses_comp,
        loggs_comp,
        Teffs_comp,
        Zs_comp,
        (delta_T, delta_J, delta_H, delta_K),
        _ldc,
        separations,
        contrasts,
        filt,
        companion_is_host=False,
        integer_teff=False,
    )
    _model_loglike_single = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=None),
        P_orb_range,
        _OrbitPolicy(
            period_factor=1.0,
            doubled_host_collision=False,
            q_branch="all",
            observation_kind="standard",
        ),
    )
    prior_transform_single = _ScenarioPrior(_model_loglike_single, _inv_sample_ecc)
    loglike_single = _ScenarioLikelihood(
        model=_model_loglike_single,
        time=time,
        data=flux,
        noise=sigma,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=secondary_loglike,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
        secondary=(time_secondary, flux_secondary, sigma_secondary),
    )

    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )
    
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
            P_eff = P  # prior_transform_twin already sets P=2*P_fold
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
    res['best_lnL'] = float(np.max(res_s.log_likelihoods))
    return res


def lnZ_DEB_evenodd(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,
            time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, Tmag: float, Jmag: float, Hmag: float,
            Kmag: float, trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
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
        N (int): Number of active particles.
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

    lnsigma = _log_sigma_norm(
        _sigma_for_observations(sigma_even, flux_even, "even"),
        _sigma_for_observations(sigma_odd, flux_odd, "odd"),
    )
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


    # theta = [uP, uinc, uecc, uargp, uq, uidx]

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _BackgroundHost.prepare(
        _Star(M_s, R_s, Teff, u1, u2),
        masses_comp,
        loggs_comp,
        Teffs_comp,
        Zs_comp,
        (delta_T, delta_J, delta_H, delta_K),
        _ldc,
        separations,
        contrasts,
        filt,
        companion_is_host=False,
        integer_teff=False,
    )
    _model_loglike_twin = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=None),
        P_orb_range,
        _OrbitPolicy(
            period_factor=2.0,
            doubled_host_collision=True,
            q_branch="all",
            observation_kind="evenodd",
        ),
    )
    prior_transform_twin = _ScenarioPrior(_model_loglike_twin, _inv_sample_ecc)
    loglike_twin = _ScenarioLikelihood(
        model=_model_loglike_twin,
        time=time_even,
        data=flux_even,
        noise=sigma_even,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_EB_evenodd,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
        secondary=(time_odd, flux_odd, sigma_odd),
    )

    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
            P_eff = P  # prior_transform_twin already sets P=2*P_fold
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

    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    res_twin['best_lnL'] = float(np.max(res_t.log_likelihoods))
    return  res_twin


def lnZ_BEB_secondary(time: np.ndarray, flux: np.ndarray, sigma: float,
                      time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Tmag: float, Jmag:float, Hmag: float, Kmag: float,
            trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            secondary_loglike: Callable = lnL_EB_second):
    """
    Calculates the marginal likelihood of the BEB scenario (background eclipsing binary).
    Returns:
        res (dict): Properties and marginal likelihood.
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


    # theta = [uP, uinc, uecc, uargp, uq, uidx]

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _BackgroundHost.prepare(
        _Star(M_s, R_s, Teff, None, None),
        masses_comp,
        loggs_comp,
        Teffs_comp,
        Zs_comp,
        (delta_T, delta_J, delta_H, delta_K),
        _ldc,
        separations,
        contrasts,
        filt,
        companion_is_host=True,
        integer_teff=False,
    )
    _model_loglike_single = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=None),
        P_orb_range,
        _OrbitPolicy(
            period_factor=1.0,
            doubled_host_collision=False,
            q_branch="all",
            observation_kind="standard",
        ),
    )
    prior_transform_single = _ScenarioPrior(_model_loglike_single, _inv_sample_ecc)
    loglike_single = _ScenarioLikelihood(
        model=_model_loglike_single,
        time=time,
        data=flux,
        noise=sigma,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=secondary_loglike,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
        secondary=(time_secondary, flux_secondary, sigma_secondary),
    )

    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
            P_eff = P  # prior_transform_twin already sets P=2*P_fold
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
    res['best_lnL'] = float(np.max(res_s.log_likelihoods))
    res['_samples'] = np.asarray(res_s.samples)
    res['_log_likelihoods'] = np.asarray(res_s.log_likelihoods)
    #res_twin = _build_output(res_t, twin=True)
    #res_twin['lnZ'] = float(lnZ_twin)
    return res#, res_twin


def lnZ_BEB_evenodd(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,
                    time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Tmag: float, Jmag: float, Hmag: float, Kmag: float,
            trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the BEB scenario (background eclipsing binary)
    using even/odd transit data.
    Returns:
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = _log_sigma_norm(
        _sigma_for_observations(sigma_even, flux_even, "even"),
        _sigma_for_observations(sigma_odd, flux_odd, "odd"),
    )

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


    # theta = [uP, uinc, uecc, uargp, uq, uidx]

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _BackgroundHost.prepare(
        _Star(M_s, R_s, Teff, None, None),
        masses_comp,
        loggs_comp,
        Teffs_comp,
        Zs_comp,
        (delta_T, delta_J, delta_H, delta_K),
        _ldc,
        separations,
        contrasts,
        filt,
        companion_is_host=True,
        integer_teff=False,
    )
    _model_loglike_twin = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=None),
        P_orb_range,
        _OrbitPolicy(
            period_factor=2.0,
            doubled_host_collision=False,
            q_branch="all",
            observation_kind="evenodd",
        ),
    )
    prior_transform_twin = _ScenarioPrior(_model_loglike_twin, _inv_sample_ecc)
    loglike_twin = _ScenarioLikelihood(
        model=_model_loglike_twin,
        time=time_even,
        data=flux_even,
        noise=sigma_even,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_EB_evenodd,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
        secondary=(time_odd, flux_odd, sigma_odd),
    )

    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
            P_eff = P  # prior_transform_twin already sets P=2*P_fold
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

    res_twin = _build_output(res_t, twin=True)
    res_twin['lnZ'] = float(lnZ_twin)
    res_twin['best_lnL'] = float(np.max(res_t.log_likelihoods))
    return  res_twin


def lnZ_NTP_unknown(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, Tmag: float, trilegal_fname: str,
                    N: int = 10, steps: int = 20,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NTP scenario for
    a star of unknown properties using persistent sampling.
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

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _UnknownHost.prepare(masses_possible, loggs_possible, Teffs_possible, Zs_possible, _ldc)
    _model_loglike = _Scenario(
        _host,
        _Planet(flatpriors, radius_prior_on_target=False),
        P_orb_range,
        _OrbitPolicy(
            period_factor=1.0,
            doubled_host_collision=False,
            q_branch="all",
            observation_kind="standard",
        ),
    )
    prior_transform = _ScenarioPrior(_model_loglike, _inv_sample_ecc)
    loglike = _ScenarioLikelihood(
        model=_model_loglike,
        time=time,
        data=flux,
        noise=sigma,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_TP,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
    )

    lnZ, res_obj = _run_persistent_evidence(
        loglike, prior_transform, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative outputs
    N_samples = POSTERIOR_NSAMPLES
    try:
        wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
        N (int): Number of active particles.
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

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _UnknownHost.prepare(masses_possible, loggs_possible, Teffs_possible, Zs_possible, _ldc)
    _model_loglike_single = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=1.0),
        P_orb_range,
        _OrbitPolicy(
            period_factor=1.0,
            doubled_host_collision=False,
            q_branch="single",
            observation_kind="legacy",
        ),
    )
    prior_transform_single = _ScenarioPrior(_model_loglike_single, _inv_sample_ecc)
    loglike_single = _ScenarioLikelihood(
        model=_model_loglike_single,
        time=time,
        data=flux,
        noise=sigma,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_EB,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
    )
    _model_loglike_twin = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=1.0),
        P_orb_range,
        _OrbitPolicy(
            period_factor=2.0,
            doubled_host_collision=True,
            q_branch="twin",
            observation_kind="legacy",
        ),
    )
    prior_transform_twin = _ScenarioPrior(_model_loglike_twin, _inv_sample_ecc)
    loglike_twin = _ScenarioLikelihood(
        model=_model_loglike_twin,
        time=time,
        data=flux,
        noise=sigma,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_EB_twin,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
    )

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
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
            P_eff = P  # prior_transform_twin already sets P=2*P_fold
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

    res = _build_output(res_s, twin=False); res['lnZ'] = float(lnZ_s); res['best_lnL'] = float(np.max(res_s.log_likelihoods))
    res_twin = _build_output(res_t, twin=True); res_twin['lnZ'] = float(lnZ_t); res_twin['best_lnL'] = float(np.max(res_t.log_likelihoods))
    return res, res_twin


def lnZ_NTP_evolved(time: np.ndarray, flux: np.ndarray, sigma: float,
                    P_orb: float, R_s: float, Teff: float, Z: float,
                    N: int = 10, steps: int = 20,
                    mission: str = "TESS", flatpriors: bool = False,
                    exptime: float = 0.00139, nsamples: int = 20):
    """
    Calculates the marginal likelihood of the NTP scenario for evolved
    (subgiant) hosts using persistent sampling. Assumes logg~3 to infer M_s from R_s,
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
        N (int): Number of active particles.
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

    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _KnownHost(_Star(M_s, R_s, Teff, u1, u2))
    _model_loglike = _Scenario(
        _host,
        _Planet(flatpriors, radius_prior_on_target=False),
        P_orb_range,
        _OrbitPolicy(
            period_factor=1.0,
            doubled_host_collision=False,
            q_branch="all",
            observation_kind="standard",
        ),
    )
    prior_transform = _ScenarioPrior(_model_loglike, _inv_sample_ecc)
    loglike = _ScenarioLikelihood(
        model=_model_loglike,
        time=time,
        data=flux,
        noise=sigma,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_TP,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
    )

    lnZ, res_obj = _run_persistent_evidence(
        loglike, prior_transform, ndim=5,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative outputs
    N_samples = POSTERIOR_NSAMPLES
    try:
        wsamps = _resample_equal(res_obj.samples, res_obj.weights)
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
        N (int): Number of active particles.
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


    #lnZ_t, res_t = _run_dynesty_evidence(loglike_twin, prior_transform_twin, ndim=5, nlive=nlive, dlogz=dlogz, dynamic=dynamic)
    _ldc = (
        (ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s)
        if mission == "TESS"
        else (ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s)
    )
    _host = _KnownHost(_Star(M_s, R_s, Teff, u1, u2))
    _model_loglike_single = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=1.0),
        P_orb_range,
        _OrbitPolicy(
            period_factor=1.0,
            doubled_host_collision=False,
            q_branch="single",
            observation_kind="legacy",
        ),
    )
    prior_transform_single = _ScenarioPrior(_model_loglike_single, _inv_sample_ecc)
    loglike_single = _ScenarioLikelihood(
        model=_model_loglike_single,
        time=time,
        data=flux,
        noise=sigma,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_EB,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
    )
    _model_loglike_twin = _Scenario(
        _host,
        _Binary(mass_ratio_prior_mass=1.0),
        P_orb_range,
        _OrbitPolicy(
            period_factor=2.0,
            doubled_host_collision=True,
            q_branch="twin",
            observation_kind="legacy",
        ),
    )
    prior_transform_twin = _ScenarioPrior(_model_loglike_twin, _inv_sample_ecc)
    loglike_twin = _ScenarioLikelihood(
        model=_model_loglike_twin,
        time=time,
        data=flux,
        noise=sigma,
        normalization=-0.5 * ln2pi - lnsigma,
        residual_cost=lnL_EB_twin,
        exptime=exptime,
        nsamples=nsamples,
        domain="real",
    )

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
            wsamps = _resample_equal(res_obj.samples, res_obj.weights)
            sel = slice(0, min(N_samples, wsamps.shape[0]))
            P = wsamps[:,0]; inc = wsamps[:,1]; ecc = wsamps[:,2]; argp_ = wsamps[:,3]; q = wsamps[:,4]
            M2 = q*M_s
            R2, _ = stellar_relations(M2, np.full_like(M2, R_s), np.full_like(M2, Teff))
            fr_eb = flux_relation(M2)/(flux_relation(M2)+flux_relation(np.array([M_s])))
            P_eff = P  # prior_transform_twin already sets P=2*P_fold
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

    res = _build_output(res_s, twin=False); res['lnZ'] = float(lnZ_s); res['best_lnL'] = float(np.max(res_s.log_likelihoods))
    res_twin = _build_output(res_t, twin=True); res_twin['lnZ'] = float(lnZ_t); res_twin['best_lnL'] = float(np.max(res_t.log_likelihoods))
    return res, res_twin


from ..sampling.records import instrument_evidence as _instrument_evidence
_instrument_evidence(globals())
