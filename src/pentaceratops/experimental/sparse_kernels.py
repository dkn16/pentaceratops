# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact native-exposure averages without full native model arrays.

Uses PyTransit 2.9's quadratic and MeepMeep orbital routines and the same
midpoint exposure-integration rule and evaluation window as its
quadratic_model_s/v routines. The caller supplies the positive integer
subsample count; the adapter default is 20.
Support/finite checks and interval selection deliberately do NOT use fastmath.
"""
import numpy as np
from numba import njit
from pytransit.models.numba.ma_quadratic_nb import (
    eval_quad_z_s, solve2d, sep_c, bounding_box,
)


@njit(cache=True, fastmath=False)
def orbit_setup(parameters):
    # Parameters: k,t0,period,a/Rstar,inclination,eccentricity,w,u1,u2,amplitude.
    n = len(parameters)
    coefficients = np.empty((n, 2, 5))
    windows = np.empty((n, 2))
    for j in range(n):
        k, _, p, a, inc, ecc, w = parameters[j, :7]
        c = solve2d(0., p, a, inc, ecc, w)
        coefficients[j] = c
        left, right = bounding_box(k, c)
        # PyTransit 2.9 uses asymmetric contact bounds, not +/- T14/2.
        windows[j, 0] = left - .025
        windows[j, 1] = right + .025
    return coefficients, windows


@njit(cache=True, fastmath=False)
def support_intervals(phase, t0, period, width):
    # Conservative candidate bounds; the original tc/window check is applied
    # again below. Nonfinite widths conservatively inspect the whole grid.
    ranges = np.zeros((2, 2), dtype=np.int64)
    if not np.isfinite(width) or width >= period/2:
        ranges[0, 1] = len(phase)
        return ranges
    if width < 0:
        return ranges
    center = t0 % period
    padding = 1e-10 * max(1., abs(t0), period)
    lo, hi = center-width-padding, center+width+padding
    if hi-lo >= period:
        ranges[0, 1] = len(phase)
    elif lo < 0:
        ranges[0, 1] = np.searchsorted(phase, hi, side='right')
        ranges[1, 0] = np.searchsorted(phase, lo+period, side='left')
        ranges[1, 1] = len(phase)
    elif hi >= period:
        ranges[0, 1] = np.searchsorted(phase, hi-period, side='right')
        ranges[1, 0] = np.searchsorted(phase, lo, side='left')
        ranges[1, 1] = len(phase)
    else:
        ranges[0, 0] = np.searchsorted(phase, lo, side='left')
        ranges[0, 1] = np.searchsorted(phase, hi, side='right')
    return ranges


def exposure_flux(tc, p, c, exptime, nsamples):
    # Finite-valued arithmetic only: callers enforce validity outside fastmath.
    k = p[0]
    ld = p[7:9]
    value = 0.
    for sample in range(1, nsamples+1):
        offset = exptime*((sample-.5)/nsamples-.5)
        z = sep_c(tc+offset, c)
        value += 1. if z > 1.+k else eval_quad_z_s(z, k, ld)
    return value/nsamples


exposure_strict = njit(cache=True, fastmath=False)(exposure_flux)


@njit(cache=True, fastmath=True)
def exposure_fast(tc, p, c, exptime, nsamples):
    # Deliberately separate Python function: Numba's cache key does not include
    # fastmath flags. Compiling exposure_flux twice can reuse the wrong binary.
    k = p[0]
    ld = p[7:9]
    value = 0.
    for sample in range(1, nsamples+1):
        offset = exptime*((sample-.5)/nsamples-.5)
        z = sep_c(tc+offset, c)
        value += 1. if z > 1.+k else eval_quad_z_s(z, k, ld)
    return value/nsamples


def projector(exposure):
    # Strict driver: NaNs must never pass a fast-math finite check. For the
    # fast variant only the inner finite exposure arithmetic is reassociated.
    @njit(cache=True, fastmath=False)
    def project(parameters, coefficients, windows, times, phase, order,
                bin_index, weights, n_bins, exptime, nsamples):
        signal = np.zeros((len(parameters), n_bins))
        counts = np.zeros(len(parameters), dtype=np.int64)
        for j in range(len(parameters)):
            p = parameters[j]
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(coefficients[j])):
                signal[j, :] = np.nan
                continue
            left, right = windows[j]
            # Symmetric bounds are only a conservative index shortcut. The
            # exact upstream asymmetric condition is applied to every point.
            width = max(abs(left), abs(right))
            ranges = support_intervals(phase, p[1], p[2], width)
            for r in range(2):
                for i in range(ranges[r, 0], ranges[r, 1]):
                    native = order[i]
                    epoch = np.floor((times[native]-p[1]+.5*p[2])/p[2])
                    tc = times[native]-(p[1]+epoch*p[2])
                    if not (left <= tc <= right): continue
                    f = exposure(tc, p, coefficients[j], exptime, nsamples)
                    counts[j] += 1
                    if not np.isfinite(f):
                        signal[j, :] = np.nan
                        break
                    signal[j, bin_index[native]] += weights[native]*(f-1.)*p[9]
        return signal, counts
    return project


project_strict = projector(exposure_strict)
project_fast = projector(exposure_fast)


def quadratic_finite(signals, precision, projected, offsets, precision_offsets):
    answer = np.zeros(len(signals))
    for j in range(len(signals)):
        for block in range(len(offsets)-1):
            lo, hi = offsets[block], offsets[block+1]
            width = hi-lo
            active = np.empty(width, dtype=np.int64)
            n = 0
            for i in range(lo, hi):
                if signals[j, i] != 0.:
                    active[n] = i
                    n += 1
            q = 0.
            linear = 0.
            for i in range(n):
                ix = active[i]
                s = signals[j, ix]
                linear += projected[ix]*s
                row = precision_offsets[block]+(ix-lo)*width
                dot = 0.
                for k in range(n):
                    iy = active[k]
                    dot += precision[row+iy-lo]*signals[j, iy]
                q += s*dot
            answer[j] += linear-.5*q
    return answer


quadratic_strict = njit(cache=True, fastmath=False)(quadratic_finite)


@njit(cache=True, fastmath=True)
def quadratic_fast(signals, precision, projected, offsets, precision_offsets):
    # Keep identical algorithm but a distinct cache identity (see above).
    answer = np.zeros(len(signals))
    for j in range(len(signals)):
        for block in range(len(offsets)-1):
            lo, hi = offsets[block], offsets[block+1]
            width = hi-lo
            active = np.empty(width, dtype=np.int64)
            n = 0
            for i in range(lo, hi):
                if signals[j, i] != 0.:
                    active[n] = i
                    n += 1
            q = 0.
            linear = 0.
            for i in range(n):
                ix = active[i]
                s = signals[j, ix]
                linear += projected[ix]*s
                row = precision_offsets[block]+(ix-lo)*width
                dot = 0.
                for k in range(n):
                    iy = active[k]
                    dot += precision[row+iy-lo]*signals[j, iy]
                q += s*dot
            answer[j] += linear-.5*q
    return answer
