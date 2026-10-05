"""Aperture geometry tied to the same FITS product as the science photometry.

Celestial coordinates and aperture pixels must refer to the same camera/CCD.
Independent TessCut products can select a different detector at CCD boundaries.
These helpers do not query catalogs or manufacture an aperture on failure.
"""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.special import ndtr


def geometry_from_fits(filename, sky_coordinates, *, tic, sector):
    """Return detector coordinates and bit-2 pipeline pixels from a TESS LC FITS.

    sky_coordinates has shape (stars, 2), RA/Dec degrees, target first. FITS WCS
    handles both celestial and physical transformations with zero-based image
    coordinates. Catalog coordinates are used as supplied (no implicit epoch
    or proper-motion changes relative to the existing target catalog).
    """
    sky = np.asarray(sky_coordinates, float)
    if sky.ndim != 2 or sky.shape[1] != 2 or not len(sky) or not np.isfinite(sky).all():
        raise ValueError("Require finite RA/Dec pairs, target first")
    with fits.open(filename, memmap=True) as hdus:
        if int(hdus[0].header['TICID']) != int(tic) or int(hdus[0].header['SECTOR']) != int(sector):
            raise ValueError("Science FITS target/sector identity mismatch")
        hdu = hdus['APERTURE']
        header, image = hdu.header, np.asarray(hdu.data)
        if image.ndim != 2 or image.dtype.kind not in 'iu':
            raise ValueError("Require a two-dimensional integer aperture mask")
        required = ('CTYPE1P', 'CTYPE2P', 'CRPIX1P', 'CRPIX2P',
                    'CRVAL1P', 'CRVAL2P', 'CDELT1P', 'CDELT2P')
        if any(k not in header for k in required):
            raise ValueError("Science aperture lacks explicit detector WCS")
        if (header['CTYPE1P'], header['CTYPE2P']) != ('RAWX', 'RAWY'):
            raise ValueError("Unexpected detector coordinate axes")
        world = WCS(header).celestial
        physical = WCS(header, key='P')
        local = world.all_world2pix(sky, 0)
        if not np.isfinite(local).all():
            raise ValueError("Nonfinite science-product WCS coordinates")
        ny, nx = image.shape
        if not (-.5 <= local[0, 0] < nx-.5 and -.5 <= local[0, 1] < ny-.5):
            raise ValueError("Target catalog position falls outside science aperture image")
        y, x = np.nonzero(image & 2)
        if not len(x):
            raise ValueError("Science product has no selected pipeline aperture pixels")
        selected = np.column_stack((x, y))
        pixels = physical.all_pix2world(selected, 0)
        detector = physical.all_pix2world(local, 0)
        if not np.allclose(pixels, np.rint(pixels), atol=1e-8, rtol=0):
            raise ValueError("Detector aperture pixels do not have integer centers")
        # Our existing circular PSF has sigma in native detector pixels.
        steps = physical.all_pix2world([[0, 0], [1, 0], [0, 1]], 0)
        if not np.allclose(steps[1:]-steps[0], np.eye(2), atol=1e-10, rtol=0):
            raise ValueError("Expected unit, axis-aligned native detector pixels")
        return dict(sector=int(sector), camera=int(hdus[0].header['CAMERA']),
                    ccd=int(hdus[0].header['CCD']), source_file=str(Path(filename).resolve()),
                    pix_coords=np.asarray(detector), aperture=np.rint(pixels).astype(int),
                    target_stamp_xy=local[0].copy())


def _normal_interval(lower, upper):
    # Positive-tail CDF subtraction would round both terms to one.
    return np.where(lower > 0, ndtr(-lower)-ndtr(-upper), ndtr(upper)-ndtr(lower))


def gaussian_aperture_flux(magnitudes, positions, aperture, *, sigma=.75):
    """Exact separable Gaussian pixel integrals, matching the established PSF.

    This evaluates the same circular sigma=0.75-pixel model as the numerical
    double integrals in calc_depths, with stable tail subtraction.
    """
    mag, xy, pixels = map(lambda a: np.asarray(a, float), (magnitudes, positions, aperture))
    if (mag.ndim != 1 or not len(mag) or xy.shape != (len(mag), 2)
            or pixels.ndim != 2 or pixels.shape[1] != 2 or not len(pixels)
            or not np.isfinite(mag).all() or not np.isfinite(xy).all()
            or not np.isfinite(pixels).all() or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("Invalid magnitudes, positions or aperture pixels")
    if len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError("Duplicate aperture pixels")
    lo = (pixels[None, :, :]-.5-xy[:, None, :])/sigma
    hi = lo+1/sigma
    integral = np.prod(_normal_interval(lo, hi), axis=2).sum(axis=1)
    relative = 10**((mag.min()-mag)/2.5)*integral
    if not np.isfinite(relative).all() or relative.sum() <= 0:
        raise ValueError("No stellar flux reaches aperture; check camera/CCD and WCS agreement")
    return relative


def dilution_depths(magnitudes, geometries, transit_depth):
    """Return mean aperture fractions and required host depths (fractional units).

    Each sector receives equal weight, preserving the established calc_depths
    convention. A required host depth above one marks that star ineligible (0),
    rather than invalidating other stars. Invalid sector geometry fails visibly.
    """
    if not np.isfinite(transit_depth) or not 0 < transit_depth < 1:
        raise ValueError("transit_depth must be a fractional depth strictly between 0 and 1")
    ratios = []
    for g in geometries:
        flux = gaussian_aperture_flux(magnitudes, g['pix_coords'], g['aperture'])
        ratios.append(flux/flux.sum())
    if not ratios:
        raise ValueError("No aperture geometries")
    per_sector = np.asarray(ratios)
    mean = per_sector.mean(axis=0)
    depth = np.zeros_like(mean)
    eligible = mean >= transit_depth
    depth[eligible] = transit_depth/mean[eligible]
    if not np.any(depth > 0):
        raise ValueError("Every star would require an eclipse deeper than 100% under this aperture model")
    return mean, depth, per_sector
