"""Planet/binary ingredients and shared orbital geometry, in explicit units."""

from dataclasses import dataclass

import numpy as np
from astropy import constants

from ..priors import sample_rp, sample_q, lnprior_bound_TP, lnprior_bound_EB
from ..stellar import stellar_relations

G = constants.G.cgs.value
MSUN = constants.M_sun.cgs.value
RSUN = constants.R_sun.cgs.value
REARTH = constants.R_earth.cgs.value


def period_range(period):
    if type(period) in (float, int):
        return (float(period), float(period))
    return (float(period[0]), float(period[-1]))


def semimajor_axis(period, total_mass):
    return ((G * total_mass * MSUN) / (4 * np.pi**2) * (period * 86400) ** 2) ** (1 / 3)


def transit_geometry(period, inc, ecc, argp, total_mass, radius_sum_cm, collision_cm=None):
    a = semimajor_axis(period, total_mass)
    correction = (1 + ecc * np.sin(argp * np.pi / 180)) / (1 - ecc**2)
    probability = radius_sum_cm / a * correction
    minimum_inc = np.degrees(np.arccos(np.minimum(1.0, probability)))
    collision = radius_sum_cm if collision_cm is None else collision_cm
    allowed = ~(probability > 1.0) & ~(collision > a * (1 - ecc)) & ~(inc < minimum_inc)
    return a, allowed


@dataclass(frozen=True)
class Planet:
    flatpriors: bool = False
    radius_prior_on_target: bool = False
    kind = "planet"

    def size_prior(self, u, host_mass, target_mass):
        return sample_rp(
            np.array(u, dtype=float, copy=True),
            np.broadcast_to(target_mass if self.radius_prior_on_target else host_mass, np.shape(u)),
            self.flatpriors,
        )

    def properties(self, size, star, host, coordinate):
        return size, np.zeros_like(size), np.zeros_like(size), np.zeros_like(size)

    def log_weight(self, host, coordinate, star, secondary_mass):
        return host.weight(coordinate, star, None, lnprior_bound_TP)


@dataclass(frozen=True)
class Binary:
    """Binary physics shared by primary-only, secondary-window, and x2P fits.

    The mass-ratio prior defaults to target mass; specialized N recipes can
    explicitly request 1 Msun. The host supplies the historical temperature
    cap and flux frame. Branch and collision cuts belong to OrbitPolicy.
    """

    kind = "binary"
    mass_ratio_prior_mass: object = None

    def size_prior(self, u, host_mass, target_mass):
        mass = target_mass if self.mass_ratio_prior_mass is None else self.mass_ratio_prior_mass
        return sample_q(np.array(u, dtype=float, copy=True), mass)

    def properties(self, size, star, host, coordinate):
        mass = size * star.mass
        m = np.atleast_1d(mass)
        radius, _ = stellar_relations(
            m,
            np.broadcast_to(star.radius, m.shape),
            np.broadcast_to(host.secondary_teff(star), m.shape),
        )
        if np.ndim(mass) == 0:
            radius = radius.item()
        return np.zeros_like(size), mass, radius, host.secondary_flux(coordinate, star, mass)

    def log_weight(self, host, coordinate, star, secondary_mass):
        return host.weight(coordinate, star, secondary_mass, lnprior_bound_EB)


@dataclass(frozen=True)
class OrbitPolicy:
    """Explicit period/collision policy, separate from host and transit physics.

    Eccentricity is drawn at the candidate period before ``period_factor`` is
    applied. Legacy primary-only binaries split at q=.95; windowed x2P does not.
    """

    period_factor: float = 1.0
    doubled_host_collision: bool = False
    q_branch: str = "all"
    observation_kind: str = "standard"

    def __post_init__(self):
        if self.period_factor not in (1.0, 2.0):
            raise ValueError("Orbit period factor must be one or two")
        if self.q_branch not in {"all", "single", "twin"}:
            raise ValueError("Unknown binary mass-ratio branch")
        if self.observation_kind not in {"standard", "legacy", "evenodd"}:
            raise ValueError("Unknown observation kind")

    def accepts(self, size):
        if self.q_branch == "single":
            return size < 0.95
        if self.q_branch == "twin":
            return size >= 0.95
        return np.ones_like(size, dtype=bool)


@dataclass(frozen=True, eq=False)
class Scenario:
    """Host and occulting system are composed, not encoded in a scenario name."""

    host: object
    system: object
    period_range: tuple
    orbit: OrbitPolicy = OrbitPolicy()

    @property
    def ndim(self):
        return 6 if self.host.has_coordinate else 5

    @property
    def kind(self):
        return "x2p" if self.orbit.observation_kind == "evenodd" else self.system.kind

    def log_weight(self, theta, star, secondary_mass):
        coordinate = np.asarray(theta)[..., 5] if self.host.has_coordinate else None
        return self.system.log_weight(self.host, coordinate, star, secondary_mass)

    def physical(self, theta):
        t = np.asarray(theta)
        # Preserve NumPy scalars for a single draw. Zero-dimensional arrays
        # incorrectly select PyTransit's vector evaluator.
        coordinates = t if t.ndim == 1 else t.T
        p, inc, ecc, argp, size = coordinates[:5]
        q = coordinates[5] if self.host.has_coordinate else None
        star = self.host.resolve(q)
        rp, m2, r2, f2 = self.system.properties(size, star, self.host, q)
        planet = self.system.kind == "planet"
        mass = star.mass if planet else star.mass + m2
        rsum = (rp * REARTH if planet else r2 * RSUN) + star.radius * RSUN
        collision = 2 * star.radius * RSUN if self.orbit.doubled_host_collision else rsum
        a, allowed = transit_geometry(p, inc, ecc, argp, mass, rsum, collision)
        allowed &= self.orbit.accepts(size)
        fc = self.host.dilution(q, star)
        return (
            star,
            m2,
            allowed,
            dict(
                P_orb=p,
                inc=inc,
                ecc=ecc,
                argp=argp,
                a=a,
                R_s=star.radius,
                u1=star.u1,
                u2=star.u2,
                R_p=rp,
                R_EB=r2,
                EB_fluxratio=f2,
                companion_fluxratio=fc,
                companion_is_host=self.host.companion_is_host,
            ),
        )
