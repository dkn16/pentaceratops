"""Model-dependent overlap with supplied local windows, without a duration cap."""
import numpy as np

from .sparse_kernels import orbit_setup
from ..likelihoods import real as lk


def contact_overlap(time, center, period, left, right, exptime):
    """Does a periodic eclipse intersect the window's exposure envelope?

    Interior gaps do not become non-detections or exclude a hypothesis. Only
    actual samples contribute to the likelihood after this conservative check.
    Contact bounds are relative to center and may be asymmetric. Uncertain
    contact bounds are passed to the renderer rather than rejected here.
    """
    center, period, left, right = np.broadcast_arrays(center, period, left, right)
    if not len(time):
        return np.zeros(center.shape, bool)
    low = np.min(time)-exptime/2
    high = np.max(time)+exptime/2
    uncertain = ~np.isfinite(left) | ~np.isfinite(right) | (left > right)
    first = np.ceil((low-center-right)/period)
    last = np.floor((high-center-left)/period)
    return uncertain | (first <= last)


def alternating_overlap(even, odd, parameters, exptime):
    """Retain x2P models whose eclipses can overlap either primary window.

    For symmetric windows of full width W and equal eclipse duration D this
    reduces to abs(anomaly offset) <= W+D, not a fixed multiple of D. Unequal
    windows, eclipse durations and contact asymmetry are handled separately.
    """
    p = {key: np.atleast_1d(value) for key, value in parameters.items()}
    period = p["P_orb"]
    offset = (lk.mean_anomaly_difference(p["ecc"], np.deg2rad(p["argp"]))-.5)*period
    k = p["R_EB"]/p["R_s"]
    k = np.where(abs(k-1.) < 1e-6, k*.999, k)
    allowed = np.zeros(len(period), bool)
    for secondary in (False, True):
        center = offset*(.5 if secondary else -.5)
        pars = np.empty((len(period), 10))
        pars[:, 0] = 1/k if secondary else k
        pars[:, 1] = center
        pars[:, 2] = period
        pars[:, 3] = p["a"]/((k if secondary else 1)*p["R_s"]*lk.Rsun)
        pars[:, 4] = np.deg2rad(p["inc"])
        pars[:, 5] = p["ecc"]
        pars[:, 6] = np.deg2rad((270 if secondary else 90)-p["argp"])
        pars[:, 7], pars[:, 8], pars[:, 9] = p["u1"], p["u2"], 1.
        _, bounds = orbit_setup(pars)
        # orbit_setup carries PyTransit's .025-day numerical search padding.
        # Use the physical contact bounds and actual exposure half-width here.
        left, right = bounds[:, 0]+.025, bounds[:, 1]-.025
        for time in (even, odd):
            allowed |= contact_overlap(time, center, period, left, right, exptime)
    return allowed
