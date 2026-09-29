import numpy as np
from astropy import constants
from pytransit import QuadraticModel

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi

tm = QuadraticModel(interpolate=False)
tm_sec = QuadraticModel(interpolate=False)

# --- set_data cache: avoid re-uploading the same time grid every call ---
# Each model's last (time, exptime, nsamples) signature is tracked; if the
# next call matches, we skip set_data() (~50-100 µs/call saved).
_tm_cache = {"id_time": None, "exptime": None, "nsamples": None, "len": 0}
_tm_sec_cache = {"id_time": None, "exptime": None, "nsamples": None, "len": 0}
# Fixed secondary-depth probe grid (used in simulate_EB_transit). Allocated
# once so the cache key (id) stays stable across calls.
_SEC_DEPTH_GRID = np.linspace(-0.05, 0.05, 25)

def _set_data_cached(model, cache, time, exptime, nsamples):
    """Call model.set_data only when (id(time), exptime, nsamples) differ."""
    key = (id(time), float(exptime), int(nsamples), len(time))
    if (cache["id_time"] != key[0] or cache["exptime"] != key[1]
            or cache["nsamples"] != key[2] or cache["len"] != key[3]):
        model.set_data(time, exptimes=exptime, nsamples=nsamples)
        cache["id_time"] = key[0]; cache["exptime"] = key[1]
        cache["nsamples"] = key[2]; cache["len"] = key[3]

def mean_anomaly_difference(e, psi):  # Equation 2
    s, c = np.sin(psi), np.cos(psi)
    bb = np.sqrt(1 - e ** 2)
    M1 = np.arctan2(bb * s / (1 + e * c), (e + c) / (1 + e * c)) - e * bb * s / (1 + e * c)
    M2 = np.arctan2(-bb * s / (1 - e * c), (e - c) / (1 - e * c)) + e * bb * s / (1 - e * c)
    return np.mod((M2 - M1) / (2 * np.pi), 1.)

def simulate_TP_transit(time: np.ndarray, R_p: float, P_orb: float,
                        inc: float, a: float, R_s: float, u1: float,
                        u2: float, ecc: float, argp: float,
                        companion_fluxratio: float = 0.0,
                        companion_is_host: bool = False,
                        exptime: float = 0.00139,
                        nsamples: int = 20):
    """
    Simulates a transiting planet light curve using PyTransit.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        R_p (float): Planet radius [Earth radii].
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        m.light_curve (numpy array): Normalized flux at eat time given.
    """
    F_target = 1
    F_comp = companion_fluxratio/(1-companion_fluxratio)
    # step 1: simulate light curve assuming only the host star exists
    _set_data_cached(tm, _tm_cache, time, exptime, nsamples)
    flux = tm.evaluate_ps(
        k=R_p*Rearth/(R_s*Rsun),
        ldc=[float(u1), float(u2)],
        t0=0.0,
        p=P_orb,
        a=a/(R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=(90-argp)*(pi/180.)
        )
    # step 2: adjust the light curve to account for flux dilution
    # from non-host star
    if companion_is_host:
        F_dilute = F_target / F_comp
        flux = (flux + F_dilute)/(1 + F_dilute)
    else:
        F_dilute = F_comp / F_target
        flux = (flux + F_dilute)/(1 + F_dilute)
    return flux


def simulate_EB_transit(time: np.ndarray, R_EB: float,
                        EB_fluxratio: float, P_orb: float, inc: float,
                        a: float, R_s: float, u1: float, u2: float,
                        ecc: float, argp: float,
                        companion_fluxratio: float = 0.0,
                        companion_is_host: bool = False,
                        exptime: float = 0.00139,
                        nsamples: int = 20):
    """
    Simulates an eclipsing binary light curve using PyTransit.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): F_comp / (F_comp + F_target).
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if it
                                  is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        m.light_curve (numpy array): Normalized flux at eat time given.
    """
    F_target = 1
    F_comp = companion_fluxratio/(1 - companion_fluxratio)
    F_EB = EB_fluxratio/(1 - EB_fluxratio)
    # step 1: simulate light curve assuming only the host star exists
    # calculate primary eclipse
    _set_data_cached(tm, _tm_cache, time, exptime, nsamples)
    k = R_EB/R_s
    if abs(k - 1.0) < 1e-6:
        k *= 0.999
    flux = tm.evaluate_ps(
        k=k,
        ldc=[float(u1), float(u2)],
        t0=0.0,
        p=P_orb,
        a=a/(R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=(90-argp)*(pi/180.)
        )
    # calculate secondary eclipse depth
    _set_data_cached(tm_sec, _tm_sec_cache, _SEC_DEPTH_GRID, 0.0, 1)
    sec_flux = tm_sec.evaluate_ps(
        k=1/k,
        ldc=[float(u1), float(u2)],
        t0=0.0, p=P_orb,
        a=a/(k*R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=(90-argp+180)*(pi/180.)
        )
    sec_flux = np.min(sec_flux)
    # step 2: adjust the light curve to account for flux dilution
    # from EB and non-host star
    if companion_is_host:
        flux = (flux + F_EB/F_comp)/(1 + F_EB/F_comp)
        sec_flux = (sec_flux + F_comp/F_EB)/(1 + F_comp/F_EB)
        F_dilute = F_target/(F_comp + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute)/(1 + F_dilute)
    else:
        flux = (flux + F_EB/F_target)/(1 + F_EB/F_target)
        sec_flux = (sec_flux + F_target/F_EB)/(1 + F_target/F_EB)
        F_dilute = F_comp/(F_target + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute)/(1 + F_dilute)
    return flux, secdepth



def simulate_EB_transit_secondary(time: np.ndarray,time_secondary: np.ndarray, R_EB: float,
                        EB_fluxratio: float, P_orb: float, inc: float,
                        a: float, R_s: float, u1: float, u2: float,
                        ecc: float, argp: float,
                        companion_fluxratio: float = 0.0,
                        companion_is_host: bool = False,
                        exptime: float = 0.00139,
                        nsamples: int = 20):
    """
    Simulates an eclipsing binary light curve using PyTransit.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): F_comp / (F_comp + F_target).
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if it
                                  is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        m.light_curve (numpy array): Normalized flux at eat time given.
    """
    F_target = 1
    F_comp = companion_fluxratio/(1 - companion_fluxratio)
    F_EB = EB_fluxratio/(1 - EB_fluxratio)
    # step 1: simulate light curve assuming only the host star exists
    # calculate primary eclipse
    _set_data_cached(tm, _tm_cache, time, exptime, nsamples)
    k = R_EB/R_s
    if abs(k - 1.0) < 1e-6:
        k *= 0.999
    flux = tm.evaluate_ps(
        k=k,
        ldc=[float(u1), float(u2)],
        t0=0.0,
        p=P_orb,
        a=a/(R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=(90-argp)*(pi/180.)
        )
    # calculate secondary eclipse depth
    _set_data_cached(tm_sec, _tm_sec_cache, time_secondary, exptime, nsamples)
    t0 = (mean_anomaly_difference(ecc, argp*(pi/180.))-0.5) * P_orb
    sec_flux = tm_sec.evaluate_ps(
        k=1/k,
        ldc=[float(u1), float(u2)],
        t0=t0, p=P_orb,
        a=a/(k*R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=(90-argp+180)*(pi/180.)
        )
    #sec_flux = np.min(sec_flux)
    # step 2: adjust the light curve to account for flux dilution
    # from EB and non-host star
    if companion_is_host:
        flux = (flux + F_EB/F_comp)/(1 + F_EB/F_comp)
        sec_flux = (sec_flux + F_comp/F_EB)/(1 + F_comp/F_EB)
        F_dilute = F_target/(F_comp + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        sec_flux =  (sec_flux + F_dilute)/(1 + F_dilute)
    else:
        flux = (flux + F_EB/F_target)/(1 + F_EB/F_target)
        sec_flux = (sec_flux + F_target/F_EB)/(1 + F_target/F_EB)
        F_dilute = F_comp/(F_target + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        sec_flux =  (sec_flux + F_dilute)/(1 + F_dilute)
    return flux, sec_flux

def simulate_EB_transit_evenodd(time: np.ndarray,time_secondary: np.ndarray, R_EB: float,
                        EB_fluxratio: float, P_orb: float, inc: float,
                        a: float, R_s: float, u1: float, u2: float,
                        ecc: float, argp: float,
                        companion_fluxratio: float = 0.0,
                        companion_is_host: bool = False,
                        exptime: float = 0.00139,
                        nsamples: int = 20):
    """
    Simulates an eclipsing binary light curve using PyTransit.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): F_comp / (F_comp + F_target).
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if it
                                  is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        m.light_curve (numpy array): Normalized flux at eat time given.
    """
    F_target = 1
    F_comp = companion_fluxratio/(1 - companion_fluxratio)
    F_EB = EB_fluxratio/(1 - EB_fluxratio)
    # step 1: simulate light curve assuming only the host star exists
    # calculate primary eclipse
    t0 = (mean_anomaly_difference(ecc, argp*(pi/180.))-0.5) * P_orb
    _set_data_cached(tm, _tm_cache, time, exptime, nsamples)
    k = R_EB/R_s
    if abs(k - 1.0) < 1e-6:
        k *= 0.999
    flux = tm.evaluate_ps(
        k=k,
        ldc=[float(u1), float(u2)],
        t0=t0*-0.5,
        p=P_orb,
        a=a/(R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=(90-argp)*(pi/180.)
        )
    # calculate secondary eclipse depth
    _set_data_cached(tm_sec, _tm_sec_cache, time_secondary, exptime, nsamples)
    
    sec_flux = tm_sec.evaluate_ps(
        k=1/k,
        ldc=[float(u1), float(u2)],
        t0=t0*0.5, p=P_orb,
        a=a/(k*R_s*Rsun),
        i=inc*(pi/180.),
        e=ecc,
        w=(90-argp+180)*(pi/180.)
        )
    #sec_flux = np.min(sec_flux)
    # step 2: adjust the light curve to account for flux dilution
    # from EB and non-host star
    if companion_is_host:
        flux = (flux + F_EB/F_comp)/(1 + F_EB/F_comp)
        sec_flux = (sec_flux + F_comp/F_EB)/(1 + F_comp/F_EB)
        F_dilute = F_target/(F_comp + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        sec_flux =  (sec_flux + F_dilute)/(1 + F_dilute)
    else:
        flux = (flux + F_EB/F_target)/(1 + F_EB/F_target)
        sec_flux = (sec_flux + F_target/F_EB)/(1 + F_target/F_EB)
        F_dilute = F_comp/(F_target + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        sec_flux =  (sec_flux + F_dilute)/(1 + F_dilute)
    return flux, sec_flux


def lnL_TP(time: np.ndarray, flux: np.ndarray, sigma: float, R_p: float,
           P_orb: float, inc: float, a: float, R_s: float,
           u1: float, u2: float, ecc: float, argp: float,
           companion_fluxratio: float = 0.0,
           companion_is_host: bool = False,
           exptime: float = 0.00139,
           nsamples: int = 20):
    """
    Calculates the log likelihood of a transiting planet scenario by
    comparing a simulated light curve and the TESS light curve.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_p (float): Planet radius [Earth radii].
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        Log likelihood (float).
    """
    model = simulate_TP_transit(
        time, R_p, P_orb, inc, a, R_s, u1, u2,
        ecc, argp,
        companion_fluxratio, companion_is_host,
        exptime, nsamples
        )
    return 0.5*(np.sum((flux-model)**2 / sigma**2))


def lnL_EB(time: np.ndarray, flux: np.ndarray, sigma: float,
           R_EB: float, EB_fluxratio: float, P_orb: float, inc: float,
           a: float, R_s: float, u1: float, u2: float,
           ecc: float, argp: float,
           companion_fluxratio: float = 0.0,
           companion_is_host: bool = False,
           exptime: float = 0.00139,
           nsamples: int = 20):
    """
    Calculates the log likelihood of an eclipsing binary scenario with
    q < 0.95 by comparing a simulated light curve and the
    TESS light curve.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        Log likelihood (float).
    """
    model, secdepth = simulate_EB_transit(
        time, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        ecc, argp,
        companion_fluxratio, companion_is_host,
        exptime, nsamples
        )
    # ``sigma`` may be a per-point vector when even and odd folded windows
    # are concatenated.  The non-detection threshold remains a scalar, while
    # the chi-square below retains every point's own uncertainty.
    sigma_threshold = float(np.median(np.asarray(sigma, dtype=float)))
    if secdepth < 1.5*sigma_threshold:
        return 0.5*(np.sum((flux-model)**2 / sigma**2))
    else:
        return np.inf
    
def lnL_EB_second(time: np.ndarray, flux: np.ndarray, sigma: float,
              time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
           R_EB: float, EB_fluxratio: float, P_orb: float, inc: float,
           a: float, R_s: float, u1: float, u2: float,
           ecc: float, argp: float,
           companion_fluxratio: float = 0.0,
           companion_is_host: bool = False,
           exptime: float = 0.00139,
           nsamples: int = 20):
    """
    Calculates the log likelihood of an eclipsing binary scenario with
    q < 0.95 by comparing a simulated light curve and the
    TESS light curve.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        Log likelihood (float).
    """
    model, secmodel = simulate_EB_transit_secondary(
        time,time_secondary, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        ecc, argp,
        companion_fluxratio, companion_is_host,
        exptime, nsamples
        )
    
    return 0.5*(np.sum((flux-model)**2 / sigma**2)) + 0.5*np.sum((flux_secondary-secmodel)**2 / sigma_secondary**2)

def lnL_EB_evenodd(time: np.ndarray, flux: np.ndarray, sigma: float,
              time_secondary: np.ndarray, flux_secondary: np.ndarray, sigma_secondary: float,
           R_EB: float, EB_fluxratio: float, P_orb: float, inc: float,
           a: float, R_s: float, u1: float, u2: float,
           ecc: float, argp: float,
           companion_fluxratio: float = 0.0,
           companion_is_host: bool = False,
           exptime: float = 0.00139,
           nsamples: int = 20):
    """
    Calculates the log likelihood of an eclipsing binary scenario with
    q < 0.95 by comparing a simulated light curve and the
    TESS light curve.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        Log likelihood (float).
    """
    # Reject if the inter-eclipse half-shift t0/2 exceeds the data window.
    # The evenodd simulator shifts primary by -t0/2 and secondary by +t0/2,
    # so if |t0/2| > window_half neither eclipse is visible in either array
    # and the lnL reduces to a degenerate no-eclipse baseline.
    t0 = (mean_anomaly_difference(ecc, argp*(pi/180.)) - 0.5) * P_orb
    win_half = max(float(np.max(np.abs(time))), float(np.max(np.abs(time_secondary))))
    if abs(t0) / 2.0 > win_half:
        return np.inf
    model_even, secmodel_odd = simulate_EB_transit_evenodd(
        time,time_secondary, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        ecc, argp,
        companion_fluxratio, companion_is_host,
        exptime, nsamples
        )
    # Evaluate the reversed primary/secondary assignment on the corresponding
    # observation grids as well.  Swapping the two model arrays directly only
    # works accidentally when even and odd windows have identical lengths.
    model_odd, secmodel_even = simulate_EB_transit_evenodd(
        time_secondary, time,
        R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        ecc, argp,
        companion_fluxratio, companion_is_host,
        exptime, nsamples,
    )
    forward_cost = (
        0.5 * np.sum((flux - model_even)**2 / sigma**2)
        + 0.5 * np.sum(
            (flux_secondary - secmodel_odd)**2 / sigma_secondary**2
        )
    )
    reverse_cost = (
        0.5 * np.sum((flux - secmodel_even)**2 / sigma**2)
        + 0.5 * np.sum(
            (flux_secondary - model_odd)**2 / sigma_secondary**2
        )
    )
    return min(forward_cost, reverse_cost)


def lnL_EB_twin(time: np.ndarray, flux: np.ndarray, sigma: float,
                R_EB: float, EB_fluxratio: float, P_orb: float,
                inc: float, a: float, R_s: float, u1: float, u2: float,
                ecc:float, argp: float,
                companion_fluxratio: float = 0.0,
                companion_is_host: bool = False,
                exptime: float = 0.00139,
                nsamples: int = 20):
    """
    Calculates the log likelihood of an eclipsing binary scenario with
    q >= 0.95 and 2xP_orb by comparing a simulated light curve
    and the TESS light curve.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_EB (float): EB radius [Solar radii].
        EB_fluxratio (float): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (float): Orbital inclination [degrees].
        a (float): Semimajor axis [cm].
        R_s (float): Star radius [Solar radii].
        u1 (float): 1st coefficient in quadratic limb darkening law.
        u2 (float): 2nd coefficient in quadratic limb darkening law.
        ecc (float): Orbital eccentricity.
        argp (float): Argument of periastron [degrees].
        companion_fluxratio (float): Proportion of flux provided by
                                     the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False
                                  if it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        Log likelihood (float).
    """
    model, secdepth = simulate_EB_transit(
        time, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        ecc, argp,
        companion_fluxratio, companion_is_host,
        exptime, nsamples
        )
    return 0.5*(np.sum((flux-model)**2 / sigma**2))


def simulate_TP_transit_p(time: np.ndarray, R_p: np.ndarray,
                          P_orb: float, inc: np.ndarray,
                          a: np.ndarray, R_s: np.ndarray,
                          u1: np.ndarray, u2: np.ndarray,
                          ecc: np.ndarray, argp: np.ndarray,
                          companion_fluxratio: np.ndarray,
                          companion_is_host: bool = False,
                          exptime: float = 0.00139,
                          nsamples: int = 20):
    """
    Simulates a transiting planet light curve using PyTransit.
    Calculates light curves in parallel.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        R_p (numpy array): Planet radius [Earth radii].
        P_orb (float): Orbital period [days].
        inc (numpy array): Orbital inclination [degrees].
        a (numpy array): Semimajor axis [cm].
        R_s (numpy array): Star radius [Solar radii].
        u1 (numpy array): 1st coefficient in quadratic limb darkening law.
        u2 (numpy array): 2nd coefficient in quadratic limb darkening law.
        ecc (numpy array): Orbital eccentricity.
        argp (numpy array): Argument of periastron [degrees].
        companion_fluxratio (numpy array): Proportion of flux provided by
                                           the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        flux (numpy array): Flux for all simulated light curves.
    """
    F_target = 1
    F_comp = companion_fluxratio/(1-companion_fluxratio)
    F_comp = F_comp.reshape(F_comp.shape[0], 1)
    # step 1: simulate light curve assuming only the host star exists
    k = R_p*Rearth/(R_s*Rsun)
    t0 = np.full_like(k, 0.)
    P_orb = np.full_like(k, P_orb)
    a = a/(R_s*Rsun)
    inc *= (pi/180.)
    w = (90-argp)*(pi/180.)
    pvp = np.array([k, t0, P_orb, a, inc, ecc, w]).T
    ldc = np.array([u1, u2]).T
    _set_data_cached(tm, _tm_cache, time, exptime, nsamples)
    flux = tm.evaluate_pv(pvp=pvp, ldc=ldc)
    # step 2: adjust the light curve to account for flux dilution
    # from non-host star
    if companion_is_host:
        F_dilute = F_target / F_comp
        flux = (flux + F_dilute)/(1 + F_dilute)
    else:
        F_dilute = F_comp / F_target
        flux = (flux + F_dilute)/(1 + F_dilute)
    return flux


def simulate_EB_transit_p(time: np.ndarray, R_EB: np.ndarray,
                          EB_fluxratio: np.ndarray,
                          P_orb: float, inc: np.ndarray,
                          a: np.ndarray, R_s: np.ndarray,
                          u1: np.ndarray, u2: np.ndarray,
                          ecc: np.ndarray, argp: np.ndarray,
                          companion_fluxratio: np.ndarray,
                          companion_is_host: bool = False,
                          exptime: float = 0.00139,
                          nsamples: int = 20):
    """
    Simulates an eclipsing binary light curve using PyTransit.
    Calculates light curves in parallel.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        R_EB (numpy array): EB radius [Solar radii].
        EB_fluxratio (numpy array): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (numpy array): Orbital inclination [degrees].
        a (numpy array): Semimajor axis [cm].
        R_s (numpy array): Star radius [Solar radii].
        u1 (numpy array): 1st coefficient in quadratic limb darkening law.
        u2 (numpy array): 2nd coefficient in quadratic limb darkening law.
        ecc (numpy array): Orbital eccentricity.
        argp (numpy array): Argument of periastron [degrees].
        companion_fluxratio (numpy array): F_comp / (F_comp + F_target).
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if it
                                  is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        flux (numpy array): Flux for all simulated light curves.
        sec_depth (numpy array): Max secondary depth for all simulated
                                 light curves.
    """
    F_target = 1
    F_comp = companion_fluxratio/(1 - companion_fluxratio)
    F_comp = F_comp.reshape(F_comp.shape[0], 1)
    F_EB = EB_fluxratio/(1 - EB_fluxratio)
    F_EB = F_EB.reshape(F_EB.shape[0], 1)
    # step 1: simulate light curve assuming only the host star exists
    # calculate primary eclipse
    k = R_EB/R_s
    k[(k - 1.0) < 1e-6] *= 0.999
    t0 = np.full_like(k, 0.)
    P_orb = np.full_like(k, P_orb)
    a = a/(R_s*Rsun)
    inc *= (pi/180.)
    w = (90-argp)*(pi/180.)
    pvp = np.array([k, t0, P_orb, a, inc, ecc, w]).T
    ldc = np.array([u1, u2]).T
    _set_data_cached(tm, _tm_cache, time, exptime, nsamples)
    flux = tm.evaluate_pv(pvp=pvp, ldc=ldc)
    # calculate secondary eclipse depth
    k = R_s/R_EB
    k[(k - 1.0) < 1e-6] *= 0.999
    a_sec = a * k  # normalize by R_EB instead of R_s
    w = (90-argp+180)*(pi/180.)
    pvp = np.array([k, t0, P_orb, a_sec, inc, ecc, w]).T
    _set_data_cached(tm_sec, _tm_sec_cache, _SEC_DEPTH_GRID, 0.0, 1)
    sec_flux = tm_sec.evaluate_pv(pvp=pvp, ldc=ldc)
    sec_flux = np.min(sec_flux, axis=1)
    sec_flux = sec_flux.reshape(sec_flux.shape[0], 1)
    # step 2: adjust the light curve to account for flux dilution
    # from EB and non-host star
    if companion_is_host:
        flux = (flux + F_EB/F_comp)/(1 + F_EB/F_comp)
        sec_flux = (sec_flux + F_comp/F_EB)/(1 + F_comp/F_EB)
        F_dilute = F_target/(F_comp + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute)/(1 + F_dilute)
    else:
        flux = (flux + F_EB/F_target)/(1 + F_EB/F_target)
        sec_flux = (sec_flux + F_target/F_EB)/(1 + F_target/F_EB)
        F_dilute = F_comp/(F_target + F_EB)
        flux = (flux + F_dilute)/(1 + F_dilute)
        secdepth = 1 - (sec_flux + F_dilute)/(1 + F_dilute)
    return flux, secdepth



def lnL_TP_p(time: np.ndarray, flux: np.ndarray, sigma: float,
             R_p: np.ndarray, P_orb: float, inc: np.ndarray,
             a: np.ndarray, R_s: np.ndarray,
             u1: np.ndarray, u2: np.ndarray,
             ecc: np.ndarray, argp: np.ndarray,
             companion_fluxratio: np.ndarray,
             companion_is_host: bool = False,
             exptime: float = 0.00139,
             nsamples: int = 20):
    """
    Calculates the log likelihood of a transiting planet scenario by
    comparing a simulated light curve and the TESS light curve.
    Calculates light curves in parallel.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_p (numpy array): Planet radius [Earth radii].
        P_orb (float): Orbital period [days].
        inc (numpy array): Orbital inclination [degrees].
        a (numpy array): Semimajor axis [cm].
        R_s (numpy array): Star radius [Solar radii].
        u1 (numpy array): 1st coefficient in quadratic limb darkening law.
        u2 (numpy array): 2nd coefficient in quadratic limb darkening law.
        ecc (numpy array): Orbital eccentricity.
        argp (numpy array): Argument of periastron [degrees].
        companion_fluxratio (numpy array): Proportion of flux provided by
                                           the unresolved companion.
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        lnL (numpy array): Log likelihood.
    """
    model = simulate_TP_transit_p(
        time, R_p, P_orb, inc, a, R_s, u1, u2,
        ecc, argp,
        companion_fluxratio, companion_is_host,
        exptime, nsamples
        )
    lnL = 0.5*(np.sum((flux-model)**2 / sigma**2, axis=1))
    return lnL


def lnL_EB_p(time: np.ndarray, flux: np.ndarray, sigma: float,
             R_EB: np.ndarray, EB_fluxratio: np.ndarray,
             P_orb: float, inc: np.ndarray,
             a: np.ndarray, R_s: np.ndarray,
             u1: np.ndarray, u2: np.ndarray,
             ecc: np.ndarray, argp: np.ndarray,
             companion_fluxratio: np.ndarray,
             companion_is_host: bool = False,
             exptime: float = 0.00139,
             nsamples: int = 20):
    """
    Calculates the log likelihood of an eclipsing binary scenario with
    q < 0.95 by comparing a simulated light curve and the
    TESS light curve. Calculates light curves in parallel.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_EB (numpy array): EB radius [Solar radii].
        EB_fluxratio (numpy array): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (numpy array): Orbital inclination [degrees].
        a (numpy array): Semimajor axis [cm].
        R_s (numpy array): Star radius [Solar radii].
        u1 (numpy array): 1st coefficient in quadratic limb darkening law.
        u2 (numpy array): 2nd coefficient in quadratic limb darkening law.
        ecc (numpy array): Orbital eccentricity.
        argp (numpy array): Argument of periastron [degrees].
        companion_fluxratio (numpy array): F_comp / (F_comp + F_target).
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False if
                                  it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        lnL (numpy array): Log likelihood.
    """
    model, secdepth = simulate_EB_transit_p(
        time, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        ecc, argp,
        companion_fluxratio, companion_is_host,
        exptime, nsamples
        )
    lnL = np.zeros(R_EB.shape[0])
    mask = (secdepth < 1.5*sigma)
    mask = mask[:,0]
    lnL[mask] = 0.5*(np.sum((flux-model[mask])**2 / sigma**2, axis=1))
    lnL[~mask] = np.inf
    return lnL


def lnL_EB_twin_p(time: np.ndarray, flux: np.ndarray, sigma: float,
                  R_EB: np.ndarray, EB_fluxratio: np.ndarray,
                  P_orb: float, inc: np.ndarray,
                  a: np.ndarray, R_s: np.ndarray,
                  u1: np.ndarray, u2: np.ndarray,
                  ecc: np.ndarray, argp: np.ndarray,
                  companion_fluxratio: np.ndarray,
                  companion_is_host: bool = False,
                  exptime: float = 0.00139,
                  nsamples: int = 20):
    """
    Calculates the log likelihood of an eclipsing binary scenario with
    q >= 0.95 and 2xP_orb by comparing a simulated light curve
    and the TESS light curve. Calculates light curves in parallel.
    Args:
        time (numpy array): Time of each data point
                            [days from transit midpoint].
        flux (numpy array): Normalized flux of each data point.
        sigma (float): Normalized flux uncertainty.
        R_EB (numpy array): EB radius [Solar radii].
        EB_fluxratio (numpy array): F_EB / (F_EB + F_target).
        P_orb (float): Orbital period [days].
        inc (numpy array): Orbital inclination [degrees].
        a (numpy array): Semimajor axis [cm].
        R_s (numpy array): Star radius [Solar radii].
        u1 (numpy array): 1st coefficient in quadratic limb darkening law.
        u2 (numpy array): 2nd coefficient in quadratic limb darkening law.
        ecc (numpy array): Orbital eccentricity.
        argp (numpy array): Argument of periastron [degrees].
        companion_fluxratio (numpy array): F_comp / (F_comp + F_target).
        companion_is_host (bool): True if the transit is around the
                                  unresolved companion and False
                                  if it is not.
        exptime (float): Exposure time of observations [days].
        nsamples (int): Sampling rate for supersampling.
    Returns:
        Log likelihood (float).
    """
    model, secdepth = simulate_EB_transit_p(
        time, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        ecc, argp,
        companion_fluxratio, companion_is_host,
        exptime, nsamples
        )
    lnL = 0.5*(np.sum((flux-model)**2 / sigma**2, axis=1))
    return lnL
