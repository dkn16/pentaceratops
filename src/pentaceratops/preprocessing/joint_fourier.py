"""Prepare joint Fourier blocks from cached, uniformly binned observations.

Pure array preprocessing, with optional explicit serialization. PSD estimation
uses the package FGP routine; the likelihood retains RAW aperture-frame flux,
not the interpolated, Gaussianized, or detrended stream used by that estimator.
Sector PSDs and normalization are fixed empirical inputs during inference.
"""
import json
from pathlib import Path

import numpy as np

from .fgp import detrend_regular_segment
from ..likelihoods.joint_fourier import GaussianBlock, total_fft_power


def prepare_sector(time, raw_flux, error, *, period, epoch, duration, cadence,
                   exptime=None, padding_fraction=.1, protect_durations=5.,
                   iterations=3, cutoff_per_day=3.):
    """Use every supplied sample once; place explicit gaps on the PSD grid.

    The finite DC prior uses the lowest-frequency total PSD power. White noise
    is counted once through the estimated PSD floor, not added as formal errors.
    Formal errors enter the existing PSD estimator and are reported separately.
    """
    time, raw_flux, error = map(lambda a: np.asarray(a,float), (time, raw_flux, error))
    if (time.ndim != 1 or len(time)<2 or raw_flux.shape!=time.shape or error.shape!=time.shape
            or not np.isfinite(time).all() or not np.isfinite(raw_flux).all()
            or not np.all(np.isfinite(error)&(error>0)) or not np.all(np.diff(time)>0)):
        raise ValueError("Require finite, ordered retained observations and positive errors")
    if (not np.all(np.isfinite([period,epoch,duration,cadence,padding_fraction,protect_durations,cutoff_per_day]))
            or not 0<duration<period/4 or cadence<=0 or padding_fraction<0
            or protect_durations<=0 or cutoff_per_day<=0 or iterations<1):
        raise ValueError("Invalid PSD preparation settings")
    exposure=cadence if exptime is None else float(exptime)
    ids=np.rint((time-time[0])/cadence).astype(int)
    if len(np.unique(ids))!=len(ids) or not np.allclose(time,time[0]+ids*cadence,atol=1e-8,rtol=0):
        raise ValueError("Expected cached uniform-grid samples with gaps; rebin explicitly first")
    base_length=int(ids[-1]+1);length=base_length+int(np.ceil(padding_fraction*base_length))
    grid=time[0]+np.arange(length)*cadence
    flux=np.full(length,np.nan);errors=flux.copy()
    flux[ids]=raw_flux;errors[ids]=error
    relative=(grid-epoch+period/2)%period-period/2
    secondary=(grid-epoch)%period-period/2
    protected=(abs(relative)<=protect_durations*duration)|(abs(secondary)<=protect_durations*duration)
    fit=detrend_regular_segment(grid,flux,errors,protected,cadence,
                                iterations=iterations,cutoff_per_day=cutoff_per_day)
    center=float(fit["center"]);scale=float(fit["scale"])/center
    non_dc=(fit["stellar_psd"]+length*fit["white_variance"])*scale**2
    power=total_fft_power(fit["stellar_psd"],float(fit["white_variance"]),length,scale,
                          dc_power=float(non_dc[0]))
    observed=np.zeros(length,bool);observed[ids]=True
    normalized=flux/center
    block=GaussianBlock.from_spectrum(grid,normalized,power,exposure,observed=observed)
    np.testing.assert_array_equal(block.time,grid[ids])
    np.testing.assert_allclose(block.time,time,atol=1e-8,rtol=0)
    np.testing.assert_allclose(block.flux,raw_flux/center,atol=0,rtol=0)
    diff=np.diff(normalized)[fit["fit_mask"][1:]&fit["fit_mask"][:-1]]
    white_sigma=float(np.sqrt(fit["white_variance"])*scale)
    robust=float(np.median(abs(diff-np.median(diff)))/.6744897501960817/np.sqrt(2)) if len(diff) else None
    ratio=None if robust is None else robust/white_sigma
    diagnostics=dict(n_observed=len(ids),n_grid=length,n_padding=length-base_length,
        observed_fraction=len(ids)/base_length,n_psd_training=int(fit["fit_mask"].sum()),
        n_protected=int(fit["protected"].sum()),center=center,relative_scale=scale,
        white_sigma=white_sigma,raw_to_psd_white_sigma=ratio,
        white_ratio_flag=ratio is None or not .5<=ratio<=2.,
        psd_fixed=True,dc_policy="lowest_frequency_total_power",white_noise_count=1,
        covariance="observed submatrix of full-band stationary periodic PSD covariance",
        likelihood_flux="raw aperture flux / fixed sector center",
        interpolated_likelihood_samples=0,secondary_window_required=False)
    spectrum=dict(time=grid,observed=observed,total_fft_power=power,
        stellar_psd=fit["stellar_psd"],white_variance=fit["white_variance"],
        center=center,relative_scale=scale,psd_training_mask=fit["fit_mask"],protected=fit["protected"])
    return block,diagnostics,spectrum


def save_block(block,directory):
    """Save covariance precision separately for read-only memory mapping by workers."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    for key in ("time","flux","precision","projected_residual","sigma"):
        np.save(directory/(key+".npy"),getattr(block,key),allow_pickle=False)
    (directory/"block.json").write_text(json.dumps(dict(logdet=block.logdet,
        null_loglike=block.null_loglike,exptime=block.exptime,n_observed=len(block.time)),indent=2)+"\n")


def load_block(directory,*,mmap_mode="r"):
    """Load an externally hash-verified block; use mode r for shared worker pages."""
    directory=Path(directory);meta=json.loads((directory/"block.json").read_text())
    arrays={k:np.load(directory/(k+".npy"),allow_pickle=False,mmap_mode=mmap_mode)
            for k in ("time","flux","precision","projected_residual","sigma")}
    n=meta.pop("n_observed")
    if arrays["precision"].shape!=(n,n) or any(arrays[k].shape!=(n,) for k in arrays if k!="precision"):
        raise ValueError("Saved block shapes changed")
    return GaussianBlock(**arrays,**meta)


def save_spectrum_block(grid_time, flux, power, exptime, directory, *, observed=None,
                        chunk_rows=256, progress=None):
    """Save the exact observed Gaussian using one in-place dense LAPACK buffer.

    This is the same covariance as GaussianBlock.from_spectrum. All observed
    orbital phases remain in the fit. Chunking controls temporary memory only;
    it does not divide the noise process into independent blocks. The inverse
    is stored in C order for the existing sparse quadratic likelihood kernel.
    """
    from scipy.linalg.lapack import dpotrf, dpotri
    from ..likelihoods.joint_fourier import observed_covariance

    grid_time, flux = np.asarray(grid_time, float), np.asarray(flux, float)
    if (grid_time.ndim != 1 or len(grid_time) < 2 or flux.shape != grid_time.shape
            or not np.isfinite(grid_time).all()):
        raise ValueError("Require matching finite regular-grid times and flux")
    spacing = np.diff(grid_time)
    if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-10):
        raise ValueError("Require regular-grid times with explicit missing samples")
    if observed is None:
        observed = np.isfinite(flux)
    else:
        observed = np.asarray(observed)
        if observed.dtype != bool or observed.shape != flux.shape:
            raise ValueError("observed must be a matching boolean mask")
    ids = np.flatnonzero(observed)
    if not len(ids) or not np.isfinite(flux[ids]).all() or not np.isfinite(exptime) or exptime <= 0:
        raise ValueError("Require finite retained observations and positive exposure")
    if isinstance(chunk_rows, bool) or int(chunk_rows) != chunk_rows or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    chunk_rows = int(chunk_rows)
    power = np.asarray(power, float)
    observed_covariance(power, len(grid_time), ids[:1])  # same PSD validation
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    n, length = len(ids), len(grid_time)
    kernel = np.fft.irfft(power / length, n=length)
    matrix = np.empty((n, n), dtype=float, order='F')
    report = progress if progress is not None else lambda stage: None
    report('construct_covariance')
    for lo in range(0, n, chunk_rows):
        hi = min(n, lo+chunk_rows)
        matrix[:, lo:hi] = kernel[(ids[:, None]-ids[None, lo:hi]) % length]
    report('cholesky')
    factor, info = dpotrf(matrix, lower=1, overwrite_a=1, clean=0)
    if info != 0:
        raise np.linalg.LinAlgError(f'Cholesky factorization failed: info={info}')
    logdet = float(2*np.log(np.diag(factor)).sum())
    report('invert_cholesky')
    inverse, info = dpotri(factor, lower=1, overwrite_c=1)
    if info != 0:
        raise np.linalg.LinAlgError(f'Cholesky inverse failed: info={info}')
    report('save_precision')
    precision = np.lib.format.open_memmap(directory/'precision.npy', mode='w+',
        dtype=float, shape=(n, n), fortran_order=False)
    projected = np.empty(n)
    residual = flux[ids]-1.
    columns = np.arange(n)[None, :]
    for lo in range(0, n, chunk_rows):
        hi = min(n, lo+chunk_rows)
        tile = np.where(columns <= np.arange(lo, hi)[:, None],
                        inverse[lo:hi, :], inverse[:, lo:hi].T)
        precision[lo:hi] = tile
        projected[lo:hi] = tile @ residual
    precision.flush()
    sigma = np.full(n, np.sqrt(kernel[0]))
    for name, value in dict(time=grid_time[ids], flux=flux[ids],
                            projected_residual=projected, sigma=sigma).items():
        np.save(directory/(name+'.npy'), value, allow_pickle=False)
    null = float(-.5*(residual@projected + logdet + n*np.log(2*np.pi)))
    (directory/'block.json').write_text(json.dumps(dict(logdet=logdet,
        null_loglike=null, exptime=float(exptime), n_observed=n), indent=2)+'\n')
    report('complete')
    return load_block(directory)
