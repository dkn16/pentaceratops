import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from scipy.integrate import dblquad

from pentaceratops.preprocessing.aperture import (
    geometry_from_fits, gaussian_aperture_flux, dilution_depths,
)


def product(tmp_path):
    w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN'];w.wcs.crpix=[4.,3.]
    w.wcs.crval=[241.,85.];w.wcs.cdelt=[-.005,.005]
    h=w.to_header();h.update(CTYPE1P='RAWX',CTYPE2P='RAWY',CRPIX1P=1.,CRPIX2P=1.,
        CRVAL1P=561.,CRVAL2P=11.,CDELT1P=1.,CDELT2P=1.)
    mask=np.zeros((6,8),np.int16);mask[2:4,3:5]=3
    primary=fits.PrimaryHDU();primary.header.update(TICID=17,SECTOR=40,CAMERA=4,CCD=2)
    p=tmp_path/'lc.fits';fits.HDUList([primary,fits.ImageHDU(mask,header=h,name='APERTURE')]).writeto(p)
    return p,w


def test_geometry_uses_same_science_detector_and_zero_origin(tmp_path):
    p,w=product(tmp_path);sky=w.all_pix2world([[3.,2.],[4.1,2.7]],0)
    g=geometry_from_fits(p,sky,tic=17,sector=40)
    np.testing.assert_allclose(g['pix_coords'],[[564.,13.],[565.1,13.7]],atol=2e-11)
    np.testing.assert_array_equal(g['aperture'],[[564,13],[565,13],[564,14],[565,14]])
    assert (g['camera'],g['ccd'])==(4,2)
    with pytest.raises(ValueError,match='identity'):geometry_from_fits(p,sky,tic=18,sector=40)
    with pytest.raises(ValueError,match='outside'):geometry_from_fits(p,w.all_pix2world([[300,400]],0),tic=17,sector=40)


def test_analytic_pixel_integrals_match_independent_quadrature():
    mag=np.array([10.,11.2,15.]);xy=np.array([[.2,-.3],[1.1,.8],[8.,-.2]])
    pixels=np.array([[0,0],[0,1],[1,0]]);sigma=.75
    expected=[]
    for m,(x,y) in zip(mag,xy):
        amp=10**((mag.min()-m)/2.5)
        expected.append(sum(dblquad(lambda yy,xx: amp/(2*np.pi*sigma**2)*
            np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma**2)), px-.5,px+.5,py-.5,py+.5,
            epsabs=1e-28,epsrel=1e-10)[0] for px,py in pixels))
    np.testing.assert_allclose(gaussian_aperture_flux(mag,xy,pixels),expected,atol=1e-28,rtol=1e-10)


def test_zero_total_flux_is_coordinate_error_not_physical_rejection():
    with pytest.raises(ValueError,match='WCS'):
        gaussian_aperture_flux([10.],[[1559.,-.8]],[[565,16],[566,16]])


def test_invalid_host_does_not_remove_valid_target():
    g=dict(pix_coords=np.array([[0,0],[0,0]]),aperture=np.array([[0,0]]))
    fraction,depth,per_sector=dilution_depths([10.,15.],[g,g],.14)
    np.testing.assert_allclose(fraction,[1/1.01,.01/1.01])
    np.testing.assert_allclose(depth,[.14*1.01,0])
    np.testing.assert_allclose(per_sector.sum(axis=1),1)
    with pytest.raises(ValueError,match='100%'):dilution_depths([10.,10.],[g],.8)
    with pytest.raises(ValueError,match='fractional'):dilution_depths([10.,15.],[g],145640.)
