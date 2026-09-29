"""
Fourier-space secondary-eclipse and even-odd marginal likelihoods.

Fourier-domain twins of the lnZ_*_secondary / lnZ_*_evenodd functions in
marginal_likelihoods_eb.py, generated mechanically from that module by swapping
the time-domain Gaussian residual for the DC-dropped rfft residual. See
marginal_likelihoods_fourier and likelihoods.lnL_*_fourier for the convention.

Secondary scenarios: the primary and secondary windows are the same folded flux
at different phase, so they share one `var_fourier` (one PSD). Even-odd
scenarios take separate `var_fourier_even` and `var_fourier_odd`.
"""

import os
import numpy as np
from pandas import read_csv
from astropy import constants

from ..likelihoods.real import *
from ..likelihoods.fourier import *
from ..priors import *
from ..stellar import stellar_relations, flux_relation
from ..evidence.real import (
    ldc_T, ldc_K,
    _run_persistent_evidence, _resample_equal, _inv_sample_ecc,
    ldc_T_Zs, ldc_T_Teffs, ldc_T_loggs, ldc_T_u1s, ldc_T_u2s,
    ldc_K_Zs, ldc_K_Teffs, ldc_K_loggs, ldc_K_u1s, ldc_K_u2s,
)
from ..evidence.fourier import _build_fourier_data

np.seterr(divide='ignore')

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi
ln2pi = np.log(2*pi)

POSTERIOR_NSAMPLES = 100


def lnZ_TEB_secondary_fourier(time: np.ndarray, flux: np.ndarray, sigma: float,
            var_fourier: np.ndarray,time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
            var_fourier_secondary: np.ndarray,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            max_shift: float = None):
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

    lnsigma = np.log(sigma)
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

    flux_ft, var_ft, norm_pri = _build_fourier_data(flux, var_fourier)
    flux_sec_ft, var_sec_ft, norm_sec = _build_fourier_data(flux_secondary, var_fourier_secondary)
    norm_const = norm_pri + norm_sec

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
        #if q >= 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_second_fourier(
            time, flux_ft, var_ft,
            time_secondary, flux_sec_ft, var_sec_ft,
            radii, fluxratio, P, inc, a, R_s, u1, u2,
            ecc, argp_, exptime=exptime, nsamples=nsamples,
            max_shift=max_shift
        )
        return float(lnL)

    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=5,
        n_active=N, target_ess=2 * N, mcmc_steps=steps,
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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


def lnZ_TEB_evenodd_fourier(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,
            var_fourier_even: np.ndarray,time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
            var_fourier_odd: np.ndarray,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            max_shift: float = None):
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

    sigma = 1./np.sqrt(1./sigma_even**2 + 1./sigma_odd**2)

    lnsigma = np.log(sigma_even)
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

    flux_even_ft, var_even_ft, norm_even = _build_fourier_data(flux_even, var_fourier_even)
    flux_odd_ft, var_odd_ft, norm_odd = _build_fourier_data(flux_odd, var_fourier_odd)
    norm_const = norm_even + norm_odd

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

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        # twin branch uses 2*P
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q = theta
        #if q < 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_evenodd_fourier(
            time_even, flux_even_ft, var_even_ft, time_odd, flux_odd_ft, var_odd_ft, radii, fluxratio, P, inc, a, R_s, u1, u2,
            ecc, argp_, exptime=exptime, nsamples=nsamples,
            max_shift=max_shift
        )
        return float(lnL)


    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=5,
        n_active=N, target_ess=2 * N, mcmc_steps=steps,
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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


def lnZ_PEB_secondary_fourier(time: np.ndarray, flux: np.ndarray, sigma: float,
            var_fourier: np.ndarray,
                      time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
            var_fourier_secondary: np.ndarray,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None, max_shift: float = None):
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

    lnsigma = np.log(sigma)
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
    flux_ft, var_ft, norm_pri = _build_fourier_data(flux, var_fourier)
    flux_sec_ft, var_sec_ft, norm_sec = _build_fourier_data(flux_secondary, var_fourier_secondary)
    norm_const = norm_pri + norm_sec

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
        #if q >= 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_second_fourier(
            time, flux_ft, var_ft,
            time_secondary, flux_sec_ft, var_sec_ft, radii, fluxratio, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr_comp, companion_is_host=False,
            exptime=exptime, nsamples=nsamples, max_shift=max_shift
        )
        lnL += companion_lnprior(M_comp)
        return float(lnL)



    # Run nested sampling for both branches
    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6, n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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


def lnZ_PEB_evenodd_fourier(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,
            var_fourier_even: np.ndarray,
                      time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
            var_fourier_odd: np.ndarray,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, plx: float, contrast_curve_file: str = None,
            filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            molusc_file: str = None, max_shift: float = None):
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

    lnsigma = np.log(np.mean([sigma_even, sigma_odd]))
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
    flux_even_ft, var_even_ft, norm_even = _build_fourier_data(flux_even, var_fourier_even)
    flux_odd_ft, var_odd_ft, norm_odd = _build_fourier_data(flux_odd, var_fourier_odd)
    norm_const = norm_even + norm_odd

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

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, qcomp = theta
        #if q < 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_evenodd_fourier(
            time_even, flux_even_ft, var_even_ft, time_odd, flux_odd_ft, var_odd_ft,
            radii, fluxratio, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr_comp, companion_is_host=False,
            exptime=exptime, nsamples=nsamples, max_shift=max_shift
        )
        lnL += companion_lnprior(M_comp)
        return float(lnL)


    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6, n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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


def lnZ_SEB_secondary_fourier(time: np.ndarray, flux: np.ndarray, sigma: float,
            var_fourier: np.ndarray,
                      time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
            var_fourier_secondary: np.ndarray,
        P_orb: float, M_s: float, R_s: float, Teff: float,
        Z: float, plx: float, contrast_curve_file: str = None,
        filt: str = "TESS",
        N: int = 10, steps: int = 20,
        mission: str = "TESS", flatpriors: bool = False,
        exptime: float = 0.00139, nsamples: int = 20,
        molusc_file: str = None, max_shift: float = None):
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

    lnsigma = np.log(sigma)

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
        r_logg = np.clip(np.round(logg_comp/0.5)*0.5, 3.5, 5.0)
        r_Teff = np.clip(np.round(Teff_comp/250)*250, 3500, 13000)
        mask = (Teffs_at_Z == r_Teff) & (loggs_at_Z == r_logg)
        u1 = float(u1s_at_Z[mask][0] if np.any(mask) else u1s_at_Z[0])
        u2 = float(u2s_at_Z[mask][0] if np.any(mask) else u2s_at_Z[0])
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
    flux_ft, var_ft, norm_pri = _build_fourier_data(flux, var_fourier)
    flux_sec_ft, var_sec_ft, norm_sec = _build_fourier_data(flux_secondary, var_fourier_secondary)
    norm_const = norm_pri + norm_sec

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
        #if q >= 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_second_fourier(
            time, flux_ft, var_ft,
            time_secondary, flux_sec_ft, var_sec_ft,  R_eb, fr_eb, P, inc, a, R_comp, u1c, u2c,
            ecc, argp_, companion_fluxratio=fr_host, companion_is_host=True,
            exptime=exptime, nsamples=nsamples, max_shift=max_shift
        )
        lnL += companion_lnprior(M_comp, M_eb)
        return float(lnL)

    # Run nested sampling for both branches
    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )
    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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
            r_logg = np.clip(np.round(logg_comp/0.5)*0.5, 3.5, 5.0)
            r_Teff = np.clip(np.round(Teff_comp/250)*250, 3500, 13000)
            u1c = np.zeros_like(M_comp); u2c = np.zeros_like(M_comp)
            for i in range(M_comp.shape[0]):
                mask = (Teffs_at_Z == r_Teff[i]) & (loggs_at_Z == r_logg[i])
                u1c[i] = float(u1s_at_Z[mask][0] if np.any(mask) else u1s_at_Z[0])
                u2c[i] = float(u2s_at_Z[mask][0] if np.any(mask) else u2s_at_Z[0])
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


def lnZ_SEB_evenodd_fourier(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,
            var_fourier_even: np.ndarray,
        time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
            var_fourier_odd: np.ndarray,
        P_orb: float, M_s: float, R_s: float, Teff: float,
        Z: float, plx: float, contrast_curve_file: str = None,
        filt: str = "TESS",
        N: int = 10, steps: int = 20,
        mission: str = "TESS", flatpriors: bool = False,
        exptime: float = 0.00139, nsamples: int = 20,
        molusc_file: str = None, max_shift: float = None):
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

    lnsigma = np.log(np.mean([sigma_even, sigma_odd]))

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
        r_logg = np.clip(np.round(logg_comp/0.5)*0.5, 3.5, 5.0)
        r_Teff = np.clip(np.round(Teff_comp/250)*250, 3500, 13000)
        mask = (Teffs_at_Z == r_Teff) & (loggs_at_Z == r_logg)
        u1 = float(u1s_at_Z[mask][0] if np.any(mask) else u1s_at_Z[0])
        u2 = float(u2s_at_Z[mask][0] if np.any(mask) else u2s_at_Z[0])
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
    flux_even_ft, var_even_ft, norm_even = _build_fourier_data(flux_even, var_fourier_even)
    flux_odd_ft, var_odd_ft, norm_odd = _build_fourier_data(flux_odd, var_fourier_odd)
    norm_const = norm_even + norm_odd

    def prior_transform_single(u: np.ndarray) -> np.ndarray:
        uP, uinc, uecc, uargp, uq, uqcomp = u
        P = P_orb_range[0] + uP*(P_orb_range[1]-P_orb_range[0]) if P_orb_range[0] != P_orb_range[1] else P_orb_range[0]
        inc = float(sample_inc(np.array([uinc]))[0])
        ecc = _inv_sample_ecc(float(uecc), planet=False, P_orb=P)
        argp_ = float(sample_w(np.array([uargp]))[0])
        q = float(sample_q(np.array([uq]), M_s)[0])
        qcomp = _qcomp_from_u(float(uqcomp))
        return np.array([P, inc, ecc, argp_, q, qcomp])


    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, qcomp = theta
        #if q < 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_evenodd_fourier(
            time_even, flux_even_ft, var_even_ft, time_odd, flux_odd_ft, var_odd_ft,
            R_eb, fr_eb, P, inc, a, R_comp, u1c, u2c,
            ecc, argp_, companion_fluxratio=fr_host, companion_is_host=True,
            exptime=exptime, nsamples=nsamples, max_shift=max_shift
        )
        lnL += companion_lnprior(M_comp, M_eb)
        return float(lnL)


    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # Build representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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
            r_logg = np.clip(np.round(logg_comp/0.5)*0.5, 3.5, 5.0)
            r_Teff = np.clip(np.round(Teff_comp/250)*250, 3500, 13000)
            u1c = np.zeros_like(M_comp); u2c = np.zeros_like(M_comp)
            for i in range(M_comp.shape[0]):
                mask = (Teffs_at_Z == r_Teff[i]) & (loggs_at_Z == r_logg[i])
                u1c[i] = float(u1s_at_Z[mask][0] if np.any(mask) else u1s_at_Z[0])
                u2c[i] = float(u2s_at_Z[mask][0] if np.any(mask) else u2s_at_Z[0])
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


def lnZ_DEB_secondary_fourier(time: np.ndarray, flux: np.ndarray, sigma: float,
            var_fourier: np.ndarray,
                      time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
            var_fourier_secondary: np.ndarray,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, Tmag: float, Jmag: float, Hmag: float,
            Kmag: float, trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            max_shift: float = None):
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

    lnsigma = np.log(sigma)
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
    flux_ft, var_ft, norm_pri = _build_fourier_data(flux, var_fourier)
    flux_sec_ft, var_sec_ft, norm_sec = _build_fourier_data(flux_secondary, var_fourier_secondary)
    norm_const = norm_pri + norm_sec

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
        #if q >= 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_second_fourier(
            time, flux_ft, var_ft,
            time_secondary, flux_sec_ft, var_sec_ft,
              R2, fr_eb, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr_bg, companion_is_host=False,
            exptime=exptime, nsamples=nsamples, max_shift=max_shift
        )
        lnL += lnprior_background_idx(idx)
        return float(lnL)



    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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


def lnZ_DEB_evenodd_fourier(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,
            var_fourier_even: np.ndarray,
            time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
            var_fourier_odd: np.ndarray,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Z: float, Tmag: float, Jmag: float, Hmag: float,
            Kmag: float, trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            max_shift: float = None):
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

    lnsigma = np.log(np.mean([sigma_even, sigma_odd]))
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
    flux_even_ft, var_even_ft, norm_even = _build_fourier_data(flux_even, var_fourier_even)
    flux_odd_ft, var_odd_ft, norm_odd = _build_fourier_data(flux_odd, var_fourier_odd)
    norm_const = norm_even + norm_odd

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

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, idxf = theta
        #if q < 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_evenodd_fourier(
            time_even, flux_even_ft, var_even_ft, time_odd, flux_odd_ft, var_odd_ft,
            R2, fr_eb, P, inc, a, R_s, u1, u2,
            ecc, argp_, companion_fluxratio=fr_bg, companion_is_host=False,
            exptime=exptime, nsamples=nsamples, max_shift=max_shift
        )
        lnL += lnprior_background_idx(idx)
        return float(lnL)


    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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


def lnZ_BEB_secondary_fourier(time: np.ndarray, flux: np.ndarray, sigma: float,
            var_fourier: np.ndarray,
                      time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
            var_fourier_secondary: np.ndarray,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Tmag: float, Jmag:float, Hmag: float, Kmag: float,
            trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            max_shift: float = None):
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

    lnsigma = np.log(sigma)

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
    flux_ft, var_ft, norm_pri = _build_fourier_data(flux, var_fourier)
    flux_sec_ft, var_sec_ft, norm_sec = _build_fourier_data(flux_secondary, var_fourier_secondary)
    norm_const = norm_pri + norm_sec

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
        #if q >= 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_second_fourier(
            time, flux_ft, var_ft,
            time_secondary, flux_sec_ft, var_sec_ft,
            R2, fr_eb_obs, P, inc, a, Rbg, u1b, u2b,
            ecc, argp_, companion_fluxratio=fr_p_obs, companion_is_host=True,
            exptime=exptime, nsamples=nsamples, max_shift=max_shift
        )
        lnL += lnprior_background_combined(idx, Mbg, M2)
        return float(lnL)

    lnZ_single, res_s = _run_persistent_evidence(
        loglike_single, prior_transform_single, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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


def lnZ_BEB_evenodd_fourier(time_even: np.ndarray, flux_even: np.ndarray, sigma_even: float,
            var_fourier_even: np.ndarray,
                    time_odd: np.ndarray, flux_odd: np.ndarray, sigma_odd: float,
            var_fourier_odd: np.ndarray,
            P_orb: float, M_s: float, R_s: float, Teff: float,
            Tmag: float, Jmag: float, Hmag: float, Kmag: float,
            trilegal_fname: str,
            contrast_curve_file: str = None, filt: str = "TESS",
            N: int = 10, steps: int = 20,
            mission: str = "TESS", flatpriors: bool = False,
            exptime: float = 0.00139, nsamples: int = 20,
            max_shift: float = None):
    """
    Calculates the marginal likelihood of the BEB scenario (background eclipsing binary)
    using even/odd transit data.
    Returns:
        res_twin (dict): Best-fit properties and marginal likelihood.
    """
    # Period handling
    if type(P_orb) in [float, int]:
        P_orb_range = (float(P_orb), float(P_orb))
    else:
        P_orb_range = (float(P_orb[0]), float(P_orb[-1]))

    lnsigma = np.log(np.mean([sigma_even, sigma_odd]))

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
    flux_even_ft, var_even_ft, norm_even = _build_fourier_data(flux_even, var_fourier_even)
    flux_odd_ft, var_odd_ft, norm_odd = _build_fourier_data(flux_odd, var_fourier_odd)
    norm_const = norm_even + norm_odd

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

    def prior_transform_twin(u: np.ndarray) -> np.ndarray:
        vals = prior_transform_single(u)
        vals[0] = 2.0*vals[0]
        return vals

    def loglike_twin(theta: np.ndarray) -> float:
        P, inc, ecc, argp_, q, idxf = theta
        #if q >= 0.95:
        #    return -np.inf
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
        lnL = norm_const - lnL_EB_evenodd_fourier(
            time_even, flux_even_ft, var_even_ft, time_odd, flux_odd_ft, var_odd_ft,
            R2, fr_eb_obs, P, inc, a, Rbg, u1b, u2b,
            ecc, argp_, companion_fluxratio=fr_p_obs, companion_is_host=True,
            exptime=exptime, nsamples=nsamples, max_shift=max_shift
        )
        lnL += lnprior_background_combined(idx, Mbg, M2)
        return float(lnL)


    lnZ_twin, res_t = _run_persistent_evidence(
        loglike_twin, prior_transform_twin, ndim=6,
        n_active=N, target_ess=2 * N, mcmc_steps=steps
    )

    # representative outputs
    def _build_output(res_obj, twin: bool = False):
        N_samples = POSTERIOR_NSAMPLES
        try:
            wsamps = _resample_equal(
                res_obj.samples, res_obj.weights,
                log_likelihoods=res_obj.log_likelihoods)
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


from ..sampling.records import instrument_evidence as _instrument_evidence
_instrument_evidence(globals())
