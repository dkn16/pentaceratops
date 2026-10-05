"""Known and bound-companion hosts shared by planet and binary scenarios.

All flux fractions retain the reference engine's target-relative convention.
Catalogue loading is preparation work, never part of likelihood evaluation.
"""

from dataclasses import dataclass

import numpy as np
from astropy import constants
from pandas import read_csv

from ..priors import sample_q_companion
from ..stellar import (
    file_to_contrast_curve,
    flux_relation,
    nearest_ldc_coefficients,
    stellar_relations,
)

G = constants.G.cgs.value
MSUN = constants.M_sun.cgs.value
RSUN = constants.R_sun.cgs.value


@dataclass(frozen=True)
class Star:
    mass: object
    radius: object
    teff: object
    u1: object
    u2: object


@dataclass(frozen=True)
class KnownHost:
    target: Star
    has_coordinate = False
    companion_is_host = False

    @classmethod
    def prepare(cls, mass, radius, teff, metallicity, ldc):
        zs, temperatures, gravities, u1s, u2s = ldc
        logg = np.log10(G * (mass * MSUN) / (radius * RSUN) ** 2)
        z = zs[np.argmin(np.abs(zs - metallicity))]
        t = temperatures[np.argmin(np.abs(temperatures - teff))]
        g = gravities[np.argmin(np.abs(gravities - logg))]
        mask = (zs == z) & (temperatures == t) & (gravities == g)
        return cls(Star(mass, radius, teff, u1s[mask], u2s[mask]))

    def mass(self, q=None):
        return self.target.mass

    def resolve(self, q=None):
        return self.target

    def flux_fraction(self, mass, filt=None):
        # Bound-component flux relative to the observed target, NOT host.
        m = np.atleast_1d(mass)
        t = np.array([self.target.mass])
        f = flux_relation(m) if filt is None else flux_relation(m, filt)
        ft = flux_relation(t) if filt is None else flux_relation(t, filt)
        result = f / (f + ft)
        return result if np.ndim(mass) else result.item()

    def log_weight(self, primary_mass, secondary_mass, prior):
        return np.zeros_like(primary_mass, dtype=float)

    def valid(self, coordinate):
        return True

    def dilution(self, coordinate, star):
        return 0.0

    def secondary_teff(self, star):
        return self.target.teff

    def secondary_flux(self, coordinate, star, mass):
        return self.flux_fraction(mass)

    def weight(self, coordinate, star, mass, prior):
        return np.zeros_like(star.mass, dtype=float)


@dataclass(frozen=True)
class BoundCompanionHost:
    """A bound companion is eclipsed; the observed target supplies dilution.

    ``ldc_rule`` records inherited differences explicitly: real-space uses
    nearest available coefficients; Fourier STP/SEB use rounded grids with
    different temperature ceilings. This refactor does not change those rules.
    """

    target: Star
    ldc: tuple
    parallax: float
    separations: np.ndarray
    contrasts: np.ndarray
    contrast_filter: object
    q_pool: object = None
    ldc_rule: str = "nearest"
    has_coordinate = True
    companion_is_host = True

    @classmethod
    def from_prepared(
        cls,
        target,
        metallicity,
        ldc,
        parallax,
        separations,
        contrasts,
        contrast_filter,
        q_pool,
        ldc_rule="nearest",
    ):
        """Compose already loaded inputs; do not read files a second time."""
        zs, ts, gs, u1s, u2s = ldc
        mask = zs == zs[np.abs(zs - metallicity).argmin()]
        return cls(
            target,
            (ts[mask], gs[mask], u1s[mask], u2s[mask]),
            parallax,
            separations,
            contrasts,
            contrast_filter,
            q_pool,
            ldc_rule,
        )

    @classmethod
    def prepare(
        cls,
        mass,
        radius,
        teff,
        metallicity,
        ldc,
        parallax,
        contrast_curve_file=None,
        filt="TESS",
        molusc_file=None,
        ldc_rule="nearest",
    ):
        if ldc_rule not in {"nearest", "rounded_10000", "rounded_13000"}:
            raise ValueError(f"Unknown bound-host limb-darkening rule: {ldc_rule}")
        zs, temperatures, gravities, u1s, u2s = ldc
        mask = zs == zs[np.abs(zs - metallicity).argmin()]
        grid = (temperatures[mask], gravities[mask], u1s[mask], u2s[mask])
        if contrast_curve_file is None:
            separations, contrasts = np.array([2.2]), np.array([1.0])
        else:
            separations, contrasts = file_to_contrast_curve(contrast_curve_file)
        pool = None
        if molusc_file is not None:
            frame = read_csv(molusc_file)
            keep = frame["semi-major axis(AU)"].values * (1 - frame["eccentricity"].values) > 10
            pool = frame[keep]["mass ratio"].values.copy()
            pool[pool < 0.1 / mass] = 0.1 / mass
        return cls(
            Star(mass, radius, teff, None, None),
            grid,
            parallax,
            separations,
            contrasts,
            None if contrast_curve_file is None else filt,
            pool,
            ldc_rule,
        )

    def sample(self, u):
        scalar = np.ndim(u) == 0
        u = np.atleast_1d(u).astype(float, copy=True)
        if self.q_pool is not None and len(self.q_pool):
            index = np.floor(np.clip(u, 1e-12, 1 - 1e-12) * len(self.q_pool)).astype(int)
            result = self.q_pool[np.minimum(index, len(self.q_pool) - 1)]
        else:
            result = sample_q_companion(u, self.target.mass)
        return result.item() if scalar else result

    def mass(self, q):
        return q * self.target.mass

    def valid(self, coordinate):
        return np.asarray(coordinate) > 0

    def dilution(self, coordinate, star):
        return self.flux_fraction(star.mass)

    def secondary_teff(self, star):
        return self.target.teff

    def secondary_flux(self, coordinate, star, mass):
        return self.flux_fraction(mass)

    def weight(self, coordinate, star, mass, prior):
        return self.log_weight(star.mass, mass, prior)

    def resolve(self, q):
        mass = self.mass(q)
        scalar = np.ndim(mass) == 0
        m = np.atleast_1d(mass)
        radius, teff = stellar_relations(
            m, np.full_like(m, self.target.radius), np.full_like(m, self.target.teff)
        )
        if scalar:
            radius, teff = radius.item(), teff.item()
        logg = np.log10(G * (mass * MSUN) / (radius * RSUN) ** 2)
        ts, gs, u1s, u2s = self.ldc
        if self.ldc_rule == "nearest":
            u1, u2 = nearest_ldc_coefficients(teff, logg, ts, gs, u1s, u2s)
        else:
            ceiling = 10000 if self.ldc_rule == "rounded_10000" else 13000
            rt = np.atleast_1d(np.clip(np.round(teff / 250) * 250, 3500, ceiling))
            rg = np.atleast_1d(np.clip(np.round(logg / 0.5) * 0.5, 3.5, 5.0))
            pairs = []
            for t, g in zip(rt, rg):
                mask = (ts == t) & (gs == g)
                pairs.append((u1s[mask][0], u2s[mask][0]) if np.any(mask) else (u1s[0], u2s[0]))
            u1, u2 = np.asarray(pairs).T
            if scalar:
                u1, u2 = u1.item(), u2.item()
        return Star(mass, radius, teff, u1, u2)

    def flux_fraction(self, mass, filt=None):
        return KnownHost(self.target).flux_fraction(mass, filt)

    def log_weight(self, primary_mass, secondary_mass, prior):
        f = self.flux_fraction(primary_mass, self.contrast_filter)
        combined = f / (1 - f)
        if secondary_mass is not None:
            f2 = self.flux_fraction(secondary_mass, self.contrast_filter)
            combined += f2 / (1 - f2)
        dm = 2.5 * np.log10(combined)
        result = prior(
            self.target.mass,
            self.parallax,
            np.atleast_1d(abs(dm)),
            self.separations,
            self.contrasts,
        )
        result = np.where((result > 0) | (dm > 0), -np.inf, result)
        return result.item() if np.ndim(primary_mass) == 0 else result


@dataclass(frozen=True)
class DilutedBoundHost:
    """The known target is eclipsed, with light from a bound companion (P)."""

    primary: KnownHost
    companion: BoundCompanionHost
    has_coordinate = True
    companion_is_host = False

    @property
    def target(self):
        return self.primary.target

    def sample(self, u):
        return self.companion.sample(u)

    def mass(self, coordinate):
        return self.target.mass

    def resolve(self, coordinate):
        return self.target

    def valid(self, coordinate):
        return self.companion.valid(coordinate)

    def dilution(self, coordinate, star):
        return self.companion.flux_fraction(self.companion.mass(coordinate))

    def secondary_teff(self, star):
        return self.target.teff

    def secondary_flux(self, coordinate, star, mass):
        return self.primary.flux_fraction(mass)

    def weight(self, coordinate, star, mass, prior):
        # The eclipsing secondary belongs to the TARGET, not the companion.
        return self.companion.log_weight(self.companion.mass(coordinate), None, prior)
