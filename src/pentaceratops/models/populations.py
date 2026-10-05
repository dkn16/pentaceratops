"""Discrete stellar populations for background and unknown-neighbor hosts.

Population ordering, base-10 background weights, and the historical
distance correction are preserved. These are not new population priors.
"""

from dataclasses import dataclass

import numpy as np

from ..priors import lnprior_background
from ..stellar import flux_relation
from .hosts import G, MSUN, RSUN, KnownHost, Star


def population_index(u, size):
    index = np.floor(np.clip(u, 1e-12, 1 - 1e-12) * size).astype(int)
    result = np.minimum(index, size - 1).astype(float)
    return result.item() if np.ndim(u) == 0 else result


def catalog_ldc(logg, teff, metallicity, ldc):
    """Nearest T/logg, then nearest available metallicity at that grid point."""
    zs, ts, gs, u1s, u2s = ldc
    t = ts[np.argmin(np.abs(ts - teff))]
    g = gs[np.argmin(np.abs(gs - logg))]
    available = zs[(ts == t) & (gs == g)]
    z = (
        available[np.argmin(np.abs(available - metallicity))]
        if len(available)
        else zs[np.argmin(np.abs(zs - metallicity))]
    )
    mask = (zs == z) & (ts == t) & (gs == g)
    return (
        (float(u1s[mask][0]), float(u2s[mask][0]))
        if np.any(mask)
        else (float(u1s[zs == z][0]), float(u2s[zs == z][0]))
    )


def catalog_stars(masses, loggs, temperatures, metallicities, ldc, integer_teff=False):
    z = np.zeros(len(masses)) if metallicities is None else metallicities
    lookup_t = np.asarray(temperatures, int) if integer_teff else temperatures
    coefficients = np.array(
        [catalog_ldc(float(g), float(t), float(v), ldc) for g, t, v in zip(loggs, lookup_t, z)]
    )
    radii = np.sqrt(G * masses * MSUN / (10**loggs)) / RSUN
    return Star(masses, radii, temperatures, coefficients[:, 0], coefficients[:, 1])


def select_star(stars, coordinate):
    index = np.asarray(coordinate).astype(int)
    # NumPy scalar values, not 0-D arrays, preserve PyTransit scalar dispatch.
    if index.ndim == 0:
        index = int(index)
    return Star(
        *(
            np.asarray(getattr(stars, name))[index]
            for name in ("mass", "radius", "teff", "u1", "u2")
        )
    )


@dataclass(frozen=True)
class BackgroundHost:
    """D: target eclipsed with a background contaminant; B: background eclipsed."""

    target: Star
    stars: Star
    fractions: np.ndarray
    delta_mag: np.ndarray
    separations: object
    contrasts: object
    filt: str
    companion_is_host: bool
    has_coordinate = True

    @classmethod
    def prepare(
        cls,
        target,
        masses,
        loggs,
        temperatures,
        metallicities,
        delta_mags,
        ldc,
        separations,
        contrasts,
        filt,
        companion_is_host,
        integer_teff=False,
    ):
        dt, dj, dh, dk = delta_mags
        fractions = 10 ** (dt / 2.5) / (1 + 10 ** (dt / 2.5))
        dm = {"J": dj, "H": dh, "K": dk}.get(filt, dt)
        # D needs only the contaminant's magnitudes; never require unused LDCs.
        stars = (
            catalog_stars(masses, loggs, temperatures, metallicities, ldc, integer_teff)
            if companion_is_host
            else Star(masses, None, None, None, None)
        )
        return cls(target, stars, fractions, dm, separations, contrasts, filt, companion_is_host)

    def sample(self, u):
        return population_index(u, len(self.fractions))

    def valid(self, coordinate):
        c = np.asarray(coordinate)
        return (c >= 0) & (c < len(self.fractions))

    def mass(self, coordinate):
        return self.resolve(coordinate).mass

    def resolve(self, coordinate):
        return select_star(self.stars, coordinate) if self.companion_is_host else self.target

    def dilution(self, coordinate, star):
        return self.fractions[np.asarray(coordinate).astype(int)]

    def secondary_teff(self, star):
        return star.teff

    def secondary_flux(self, coordinate, star, mass):
        known = KnownHost(self.target)
        fraction = known.flux_fraction(mass)
        if self.companion_is_host:
            bound = known.flux_fraction(star.mass)
            correction = np.divide(
                self.dilution(coordinate, star),
                bound,
                out=np.zeros_like(bound, dtype=float),
                where=bound > 0,
            )
            fraction = fraction * correction
        return fraction

    def weight(self, coordinate, star, secondary_mass, prior):
        dm = self.delta_mag[np.asarray(coordinate).astype(int)]
        n = len(self.fractions)
        if self.separations is None:
            result = np.full_like(dm, min(0.0, np.log10((n / 0.1) * (1 / 3600) ** 2 * 2.2**2)))
        else:
            measured = dm
            if self.companion_is_host and secondary_mass is not None:
                fp = 10 ** (dm / 2.5) / (1 + 10 ** (dm / 2.5))
                known = KnownHost(self.target)
                bound = known.flux_fraction(star.mass, self.filt)
                correction = np.divide(
                    fp, bound, out=np.zeros_like(bound, dtype=float), where=bound > 0
                )
                feb = known.flux_fraction(secondary_mass, self.filt) * correction
                measured = 2.5 * np.log10(fp / (1 - fp) + feb / (1 - feb))
            result = np.minimum(
                0.0,
                lnprior_background(
                    n, np.atleast_1d(abs(measured)), self.separations, self.contrasts
                ),
            )
        result = np.where(dm > 0, -np.inf, result)
        return result.item() if np.ndim(coordinate) == 0 else result


@dataclass(frozen=True)
class UnknownHost:
    """Nearby star drawn from an already magnitude-selected stellar population."""

    stars: Star
    allowed: np.ndarray
    # Reference NEB's q prior is explicitly solar-mass, not the sampled mass.
    target: Star = Star(1.0, None, None, None, None)
    has_coordinate = True
    companion_is_host = False

    @classmethod
    def prepare(cls, masses, loggs, temperatures, metallicities, ldc):
        stars = catalog_stars(masses, loggs, temperatures, metallicities, ldc)
        return cls(stars, (loggs >= 3.5) & (temperatures <= 10000))

    def sample(self, u):
        return population_index(u, len(self.allowed))

    def valid(self, coordinate):
        c = np.asarray(coordinate)
        in_range = (c >= 0) & (c < len(self.allowed))
        safe = np.where(np.isfinite(c), c, 0)
        index = np.clip(safe, 0, len(self.allowed) - 1).astype(int)
        return in_range & self.allowed[index]

    def resolve(self, coordinate):
        return select_star(self.stars, coordinate)

    def mass(self, coordinate):
        return self.resolve(coordinate).mass

    def dilution(self, coordinate, star):
        return 0.0

    def secondary_teff(self, star):
        return star.teff

    def secondary_flux(self, coordinate, star, mass):
        f, fhost = flux_relation(np.atleast_1d(mass)), flux_relation(np.atleast_1d(star.mass))
        result = f / (f + fhost)
        return result.item() if np.ndim(mass) == 0 else result

    def weight(self, coordinate, star, mass, prior):
        return np.zeros_like(star.mass, dtype=float)
