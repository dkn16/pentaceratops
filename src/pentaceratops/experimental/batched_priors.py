"""Vectorized versions of the pinned engine's core prior transforms.

No interpolation of priors; reuse the existing piecewise mass/radius samplers.
The original sampler's unit-cube coordinates must never be mutated.
"""
import inspect
import numpy as np
from scipy.special import betaincinv


def closure(fn):
    return inspect.getclosurevars(fn).nonlocals


def eccentricity(u, planet, period):
    u = np.clip(u, 1e-12, 1-1e-12)
    if planet:
        return betaincinv(.867, 3.030, u)
    return u**(1/np.where(np.asarray(period) <= 10, .2, .6))


def population_index(u, size):
    if size <= 0:
        raise ValueError('An empty population must use the engine empty-support return')
    return np.minimum(np.floor(np.clip(u, 1e-12, 1-1e-12)*size).astype(int), size-1)


def make_prior(scalar):
    """Build a batched transform from a core scenario closure, failing closed.

    All existing prior support, population ordering and pre-2P eccentricity
    conventions are preserved. Source hashes pin the closure implementation.
    """
    env = closure(scalar)
    if scalar.__name__ == 'prior_transform_twin':
        base = make_prior(env['prior_transform_single'])
        def twin(u):
            value = base(u)
            value[:, 0] *= 2
            return value
        return twin
    owner = scalar.__qualname__.split('.')[0]
    planets = {'lnZ_TTP', 'lnZ_PTP', 'lnZ_STP', 'lnZ_DTP', 'lnZ_BTP'}
    binaries = {f'lnZ_{prefix}EB{suffix}' for prefix in ('T', 'P', 'S', 'D', 'B')
                for suffix in ('', '_secondary', '_evenodd')}
    if owner not in planets | binaries:
        raise ValueError(f'Unsupported prior closure: {scalar.__qualname__}')
    planet = owner in planets
    prefix = owner[4]
    mass = env.get('M_s')
    if mass is None and '_q_from_u' in env:
        mass = closure(env['_q_from_u'])['M_s']
    if mass is None:
        raise ValueError('No fixed target mass in prior closure')
    prange = env['P_orb_range']
    flat = env.get('flatpriors', False)
    sample_inc = scalar.__globals__['sample_inc']
    sample_w = scalar.__globals__['sample_w']
    sample_rp = scalar.__globals__['sample_rp']
    sample_q = scalar.__globals__['sample_q']
    sample_q_companion = scalar.__globals__['sample_q_companion']
    companion_population = None
    if prefix in ('P', 'S'):
        qenv = closure(env['_qcomp_from_u'])
        companion_population = qenv['molusc_qs']
    ndim = 5 if prefix == 'T' else 6

    def prior(u):
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != ndim or not np.all(np.isfinite(u)):
            raise ValueError('Invalid unit-cube batch')
        if np.any((u < 0) | (u > 1)):
            raise ValueError('Prior input outside unit cube')
        out = np.empty_like(u)
        out[:, 0] = prange[0] + u[:, 0]*(prange[1]-prange[0])
        out[:, 1] = sample_inc(u[:, 1].copy())
        out[:, 2] = eccentricity(u[:, 2], planet, out[:, 0])
        out[:, 3] = sample_w(u[:, 3].copy())
        if prefix in ('P', 'S'):
            if companion_population is not None and len(companion_population):
                out[:, 5] = companion_population[population_index(u[:, 5], len(companion_population))]
            else:
                out[:, 5] = sample_q_companion(u[:, 5].copy(), mass)
        elif prefix in ('D', 'B'):
            out[:, 5] = population_index(u[:, 5], env['N_comp'])
        if planet:
            host_mass = out[:, 5]*mass if prefix == 'S' else np.full(len(u), mass)
            out[:, 4] = sample_rp(u[:, 4].copy(), host_mass, flat)
        else:
            out[:, 4] = sample_q(u[:, 4].copy(), mass)
        return out
    return prior
