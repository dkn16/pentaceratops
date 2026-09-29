"""Restore SPOC's normalized aperture frame before validation preprocessing.

PDCSAP has already removed catalogue crowding. Both validation engines model
that dilution themselves, so they need 1 + CROWDSAP * (PDCSAP/median - 1).
FLFRCSAP cancels upon normalization. Apply the same factor to errors BEFORE
estimating a PSD, detrending, or folding; covariance then scales quadratically.
"""
from __future__ import annotations

import numpy as np

APERTURE_FRAME = "normalized_aperture_restored_v1"


def lightcurve_flux_frame(lc):
    """Require an explicit flux origin; never guess a missing correction."""
    meta = lc.meta
    origin = str(meta.get("FLUX_ORIGIN", "")).lower()
    if origin not in ("pdcsap_flux", "sap_flux"):
        raise ValueError(f"Unknown flux origin {origin!r}; cannot establish dilution frame")
    crowd = meta.get("CROWDSAP")
    if origin == "pdcsap_flux":
        try:
            factor = float(crowd)
        except (ValueError, TypeError):
            raise ValueError("PDCSAP requires a finite CROWDSAP in (0, 1]") from None
        if not np.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("PDCSAP requires a finite CROWDSAP in (0, 1]")
    else:
        factor = 1.0  # SAP has NOT had catalogue dilution removed.
    return dict(flux_frame=APERTURE_FRAME, flux_origin=origin,
                crowding_restore_factor=factor,
                crowding_metric=None if crowd is None else float(crowd),
                aperture_flux_fraction=meta.get("FLFRCSAP"),
                source_file=str(meta.get("FILENAME", "")))


def restore_aperture_flux(flux, error, factor):
    """Affine inverse of normalized crowding correction; no input mutation."""
    factor = float(factor)
    if not np.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("Crowding restore factor must be in (0, 1]")
    return 1.0 + factor * (np.asarray(flux, float) - 1.0), factor * np.asarray(error, float)


def require_aperture_frame(data):
    """Reject old caches; a new code version cannot repair an old folded flux."""
    if "flux_frame" not in data or str(data["flux_frame"]) != APERTURE_FRAME:
        raise ValueError(
            "Prepared photometry is not in the restored aperture frame. "
            "Regenerate preprocessing in a NEW cache/output directory; legacy "
            "PDCSAP caches must not be passed to a diluted model."
        )


def aperture_catalog_depth(depth, time, error, sector, products, epoch, period, duration):
    """Host-eligibility depth proxy, not a fit or change to the catalogue.

TOI depth is treated as the crowding-corrected target depth. Weight each
selected sector's CROWDSAP by the inverse-variance weight of its in-transit
observations. This scalar is only the calc_depths eligibility input; actual
flux/error scaling is per sector, never by this average. Unobserved transits
raise rather than silently guessing. For SAP the correction still needs the
sector CROWDSAP because the catalogue depth remains target-normalized.
"""
    time, error, sector = map(np.asarray, (time, error, sector))
    if not (time.shape == error.shape == sector.shape):
        raise ValueError("Native time/error/sector shapes differ")
    phase = (time - epoch + period / 2) % period - period / 2
    use = np.isfinite(time) & np.isfinite(error) & (error > 0) & (abs(phase) <= duration / 2)
    if not np.any(use):
        raise ValueError("No in-transit cadences for aperture depth conversion")
    scales = {}
    for row in products:
        c = float(row["crowding_metric"])
        if not np.isfinite(c) or not 0 < c <= 1:
            raise ValueError("Catalogue depth conversion requires valid CROWDSAP")
        if int(row["sector"]) in scales:
            raise ValueError("Duplicate sector in flux-frame provenance")
        scales[int(row["sector"])] = c
    factors = np.array([scales[int(s)] for s in sector[use]])
    weights = (np.min(error[use]) / error[use]) ** 2
    effective = float(np.average(factors, weights=weights))
    return float(depth) * effective, effective
