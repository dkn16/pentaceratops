# Aperture coordinates from science FITS

When TESS observes a source near a detector boundary, an independently requested
TessCut can select a different CCD from the light curve used for inference.
Numerically finite pixel coordinates alone do not establish agreement: both the
star positions and aperture pixels must refer to the same detector.

`pentaceratops.preprocessing.aperture.geometry_from_fits` uses the light-curve
FITS APERTURE extension, celestial WCS and physical RAWX/RAWY WCS. It verifies the
TIC ID and sector, requires an explicit pipeline aperture (bit 2), and checks
that the target lies inside the science stamp. Coordinates are supplied in
RA/Dec degrees with the target first; no implicit proper-motion correction is
applied. Retain and document the catalog epoch convention used by the analysis.

```python
from pentaceratops.preprocessing.aperture import geometry_from_fits, dilution_depths

geometry = geometry_from_fits(
    lightcurve_fits, stars[["ra", "dec"]].to_numpy(), tic=tic, sector=sector,
)
# Repeat for every selected sector, using its actual light-curve FITS product.
fraction, host_depth, per_sector = dilution_depths(
    stars.Tmag.to_numpy(), geometries, transit_depth=aperture_depth_fraction,
)
stars["fluxratio"] = fraction
stars["tdepth"] = host_depth
```

The circular 0.75-pixel Gaussian PSF and equal sector weighting match the
established calc_depths convention. Pixel integrals use the analytic normal-CDF
formula, with stable positive-tail subtraction. A zero total aperture flux raises
an explicit coordinate/WCS error. Required intrinsic depth is observed aperture
depth divided by the star's flux fraction. Stars requiring more than a 100%
eclipse receive zero eligibility; they do not invalidate other valid hosts.
Depth arguments are dimensionless fractions, not ppm. This is the established
approximate Gaussian aperture model, not a replacement with a calibrated TESS
pixel-response function. No fallback 5x5 aperture or catalog value is invented.

Synthetic WCS and independent numerical quadrature tests are in
`tests/test_aperture.py`. The HZ recovery campaign records real FITS/source hashes,
per-sector fractions, and the explicit catalog and segment-selection provenance.
