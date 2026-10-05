"""Full-period Fourier models with both eclipses on every observed grid.

The retained Fourier coefficients, PSD, normalization, physical priors and
parity profiling are unchanged. No local timing prior is inferred from the
full orbital span, and a displaced eclipse is not lost at a half-grid boundary.
"""
from contextlib import contextmanager

import numpy as np

from . import real as lk
from .fourier import _rfft_nodc, _fourier_chi2


def binary_flux(time, parameters, *, alternating=False):
    """Complete aperture-independent EB flux at times relative to the ephemeris.

    For alternating eclipses the simulator uses centers -offset/2 and
    +offset/2 on grids separated by half the binary period. The caller can
    translate the supplied times to evaluate either parity without swapping
    predictions between unequal grids.
    """
    time = np.asarray(time, float)
    if not len(time):
        return np.ones(time.shape)
    # The preserved cache keys use array identity. These coordinate arrays are
    # transient; reset before each independent render to avoid identity reuse.
    lk._tm_cache["id_time"] = None
    lk._tm_sec_cache["id_time"] = None
    simulate = lk.simulate_EB_transit_evenodd if alternating else lk.simulate_EB_transit_secondary
    primary, secondary = simulate(time, time-parameters["P_orb"]/2, **parameters)
    return 1+(primary-1)+(secondary-1)


def _parameters(R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp,
                companion_fluxratio, companion_is_host, exptime, nsamples):
    return dict(R_EB=R_EB, EB_fluxratio=EB_fluxratio, P_orb=P_orb, inc=inc,
                a=a, R_s=R_s, u1=u1, u2=u2, ecc=ecc, argp=argp,
                companion_fluxratio=companion_fluxratio, companion_is_host=companion_is_host,
                exptime=exptime, nsamples=nsamples)


def lnL_EB_second_observed(time, flux_ft, var_ft, time_secondary, flux_sec_ft,
                           var_sec_ft, R_EB, EB_fluxratio, P_orb, inc, a, R_s,
                           u1, u2, ecc, argp, companion_fluxratio=0.,
                           companion_is_host=False, exptime=.00139, nsamples=20,
                           max_shift=None):
    if max_shift is not None:
        raise ValueError("An explicit timing cut requires timing_policy='legacy'")
    parameters = _parameters(R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc,
                              argp, companion_fluxratio, companion_is_host, exptime, nsamples)
    first = binary_flux(time, parameters)
    second = binary_flux(np.asarray(time_secondary)+P_orb/2, parameters)
    return (_fourier_chi2(flux_ft, _rfft_nodc(first), var_ft)
            + _fourier_chi2(flux_sec_ft, _rfft_nodc(second), var_sec_ft))


def lnL_EB_evenodd_observed(time, flux_ft, var_ft, time_secondary, flux_sec_ft,
                            var_sec_ft, R_EB, EB_fluxratio, P_orb, inc, a, R_s,
                            u1, u2, ecc, argp, companion_fluxratio=0.,
                            companion_is_host=False, exptime=.00139, nsamples=20,
                            max_shift=None):
    if max_shift is not None:
        raise ValueError("An explicit timing cut requires timing_policy='legacy'")
    parameters = _parameters(R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc,
                              argp, companion_fluxratio, companion_is_host, exptime, nsamples)
    scores = []
    for reverse in (False, True):
        shift = P_orb/2 if reverse else 0.
        first = binary_flux(np.asarray(time)+shift, parameters, alternating=True)
        second = binary_flux(np.asarray(time_secondary)+P_orb/2+shift, parameters, alternating=True)
        scores.append(_fourier_chi2(flux_ft, _rfft_nodc(first), var_ft)
                      + _fourier_chi2(flux_sec_ft, _rfft_nodc(second), var_sec_ft))
    return min(scores)


@contextmanager
def observed_fourier_engine():
    """Select complete orbital models for the recorded full-period dispatcher.

    Like the existing covariance hooks, this requires process isolation, not
    concurrent threads. The low-level compatibility functions are restored even
    after a failed run and remain available for archived numerical reproduction.
    """
    from ..evidence import fourier_eclipses
    replacements = dict(lnL_EB_second_fourier=lnL_EB_second_observed,
                        lnL_EB_evenodd_fourier=lnL_EB_evenodd_observed)
    saved = {name: getattr(fourier_eclipses, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(fourier_eclipses, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(fourier_eclipses, name, value)
