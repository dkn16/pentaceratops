"""Shared callbacks for composed hosts, systems, and explicit orbit policies.

Public wrappers still own normalization, observation preparation, sampling
hooks, and their historical posterior-output policies. No x2P recipe is
implicitly inferred here.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import betaincinv

from ..priors import sample_inc, sample_w


@dataclass(frozen=True, eq=False)
class ScenarioPrior:
    model: object
    inverse_eccentricity: Callable

    def __call__(self, u):
        lo, hi = self.model.period_range
        period = lo if lo == hi else lo + u[0] * (hi - lo)
        inc = float(sample_inc(np.array([u[1]]))[0])
        planet = self.model.system.kind == "planet"
        ecc = self.inverse_eccentricity(float(u[2]), planet=planet, P_orb=period)
        argp = float(sample_w(np.array([u[3]]))[0])
        host = self.model.host
        q = host.sample(float(u[5])) if host.has_coordinate else None
        size = float(
            self.model.system.size_prior(np.array([u[4]]), host.mass(q), host.target.mass)[0]
        )
        values = [period * self.model.orbit.period_factor, inc, ecc, argp, size]
        return np.array(values + [q] if host.has_coordinate else values)

    def batch(self, u):
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != self.model.ndim or not np.isfinite(u).all():
            raise ValueError("Invalid unit-cube batch")
        if np.any((u < 0) | (u > 1)):
            raise ValueError("Prior input outside unit cube")
        lo, hi = self.model.period_range
        out = np.empty_like(u)
        out[:, 0] = lo + u[:, 0] * (hi - lo)
        out[:, 1] = sample_inc(u[:, 1].copy())
        ue = np.clip(u[:, 2], 1e-12, 1 - 1e-12)
        out[:, 2] = (
            betaincinv(0.867, 3.030, ue)
            if self.model.system.kind == "planet"
            else ue ** (1 / np.where(out[:, 0] <= 10, 0.2, 0.6))
        )
        out[:, 3] = sample_w(u[:, 3].copy())
        host = self.model.host
        q = host.sample(u[:, 5]) if host.has_coordinate else None
        if host.has_coordinate:
            out[:, 5] = q
        out[:, 4] = self.model.system.size_prior(u[:, 4], host.mass(q), host.target.mass)
        out[:, 0] *= self.model.orbit.period_factor
        return out


@dataclass(frozen=True, eq=False)
class ScenarioLikelihood:
    model: object
    time: np.ndarray
    data: np.ndarray
    noise: object
    normalization: float
    residual_cost: Callable
    exptime: float
    nsamples: int
    domain: str
    secondary: tuple = ()
    cost_options: dict = field(default_factory=dict)

    @property
    def kind(self):
        return self.model.kind

    def __call__(self, theta):
        if not self.model.host.valid(theta[5] if self.model.host.has_coordinate else None):
            return -np.inf
        star, m2, allowed, columns = self.model.physical(theta)
        if not allowed:
            return -np.inf
        columns = columns.copy()
        if self.kind == "planet":
            del columns["R_EB"], columns["EB_fluxratio"]
        else:
            del columns["R_p"]
        value = self.normalization - self.residual_cost(
            self.time,
            self.data,
            self.noise,
            *self.secondary,
            **columns,
            exptime=self.exptime,
            nsamples=self.nsamples,
            **self.cost_options,
        )
        if self.model.host.has_coordinate:
            value += self.model.log_weight(theta, star, m2)
        return float(value)

    def physical_batch(self, theta):
        if self.domain != "real":
            raise ValueError("The fast adapter only supports real-space observations")
        t = np.asarray(theta, float)
        if t.ndim != 2 or t.shape[1] != self.model.ndim:
            raise ValueError("Unexpected physical parameter shape")
        base = np.full(len(t), -np.inf)
        valid = np.isfinite(t).all(axis=1)
        valid &= (t[:, 0] > 0) & (t[:, 2] >= 0) & (t[:, 2] < 1) & (t[:, 4] > 0)
        if self.model.host.has_coordinate:
            valid &= self.model.host.valid(t[:, 5])
        ids = np.flatnonzero(valid)
        if not len(ids):
            return base, ids, {}
        star, m2, allowed, columns = self.model.physical(t[ids])
        prior = self.model.log_weight(t[ids], star, m2)
        allowed &= np.isfinite(prior)
        columns = {
            name: np.broadcast_to(value, (len(ids),))[allowed] for name, value in columns.items()
        }
        base[ids[allowed]] = self.normalization + np.broadcast_to(prior, (len(ids),))[allowed]
        return base, ids[allowed], columns
