"""Build evidence inputs from saved example preparations without refitting a PSD."""
import numpy as np
from scipy.sparse import coo_matrix, diags

from .candidate import PANELS, assignments
from .catalog import Candidate
from .flux import require_aperture_frame
from .folded import FoldedFourierData
from .folded_covariance import propagate_fourier_covariance
from .posterior import conditional_posterior
from ..experimental.folded_joint import full_orbit_fold
from ..likelihoods.joint_fourier import GaussianBlock, total_fft_power


def candidate_state(saved):
    """Check saved normalization, ephemerides, grids and held-out observations."""
    if saved.metadata.get("artifact") != "candidate_preparation":
        raise ValueError("Require a bundle from the candidate preprocessing scripts")
    settings = saved.metadata["settings"]
    ephemeris = Candidate(**settings["candidate"])
    windows = saved.output["windows"]
    require_aperture_frame(windows)
    dt = float(windows["bin_days"])
    if not np.isfinite(dt) or dt <= 0 or float(windows["exptime_days"]) != dt:
        raise ValueError("Invalid saved bin/exposure duration")
    for key, value in (("period", ephemeris.period_days), ("epoch_bjd", ephemeris.epoch_bjd),
                       ("time_zero_bjd", ephemeris.time_zero)):
        if float(windows[key]) != value:
            raise ValueError(f"Saved {key} differs from the candidate ephemeris")
    centers = np.asarray(windows["time_even"], float)
    if centers.ndim != 1 or len(centers) < 2 or not np.isfinite(centers).all():
        raise ValueError("Invalid saved window grid")
    for panel in PANELS:
        if not np.array_equal(windows["time_"+panel], centers):
            raise ValueError("Candidate preparation requires its original common window grid")
    if not np.allclose(np.diff(centers), dt, atol=1e-10, rtol=1e-8):
        raise ValueError("Window grid does not match the saved cadence")
    segments = saved.output["segments"]
    if not segments or len({int(s["segment"]) for s in segments}) != len(segments):
        raise ValueError("Require unique saved sectors/quarters")
    native = []
    for sector in segments:
        t = np.asarray(sector["time"], float)
        obs = np.asarray(sector["observed"])
        if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all()
                or not np.allclose(np.diff(t), dt, atol=1e-8, rtol=1e-8)
                or obs.dtype != bool or obs.shape != t.shape or not obs.any()):
            raise ValueError("Invalid saved regular grid or observation mask")
        raw, error = (np.asarray(sector[k], float) for k in ("raw_flux", "input_error"))
        if raw.shape != t.shape or error.shape != t.shape:
            raise ValueError("Saved flux/error shapes differ from the grid")
        if not np.array_equal(obs, np.isfinite(raw) & np.isfinite(error) & (error > 0)):
            raise ValueError("Saved observation mask does not match the raw flux/errors")
        _, protected = assignments(t, obs, ephemeris, centers, dt)
        if (not np.array_equal(protected, sector["protected"])
                or not np.array_equal(obs & ~protected, sector["fit_mask"])):
            raise ValueError("Saved FGP protection/training masks differ from the likelihood windows")
        center, scale = float(sector["center"]), float(sector["scale"])
        if not np.isfinite([center, scale]).all() or min(center, scale) <= 0:
            raise ValueError("Invalid saved FGP normalization")
        if not np.allclose(np.asarray(sector["relative_error"])[obs], error[obs]/center,
                           rtol=1e-12, atol=0):
            raise ValueError("Saved errors are not in the normalized aperture frame")
        native.append(t[obs])
    joined = np.concatenate(native)
    if len(np.unique(joined)) != len(joined):
        raise ValueError("Duplicate observed times across saved sectors/quarters")
    return ephemeris, windows, segments, settings["options"], dt


def candidate_folded_real(saved):
    """Propagate the fixed-PSD conditional posterior through the original folds.

    Shared coefficient columns retain all cross-panel covariance. Measurement
    errors use the established HZ median within each panel, separately from
    the trend factor. No MAP optimization or PSD estimation is repeated.
    """
    ephem, windows, segments, options, dt = candidate_state(saved)
    centers = np.asarray(windows["time_even"])
    size, operators = len(centers), []
    sums = np.zeros(3*size)
    for sector in segments:
        assigned, _ = assignments(sector["time"], sector["observed"], ephem, centers, dt)
        rows, columns, weights = [], [], []
        for i, panel in enumerate(PANELS):
            ids, use = assigned[panel]
            rows.extend(i*size+ids[use])
            columns.extend(np.flatnonzero(use))
            weights.extend(1/np.asarray(sector["relative_error"])[use]**2)
        operator = coo_matrix((weights, (rows, columns)),
                              shape=(3*size, len(sector["time"]))).tocsr()
        sums += np.asarray(operator.sum(axis=1)).ravel()
        operators.append(operator)
    if not np.all(np.isfinite(sums) & (sums > 0)):
        raise ValueError("Missing observations in a saved folded window")
    normalize = diags(1/sums)
    mean, map_flux = np.zeros(3*size), np.zeros(3*size)
    factors = []
    for sector, operator in zip(segments, operators):
        operator = normalize @ operator
        if operator[:, np.asarray(sector["fit_mask"], bool)].nnz:
            raise ValueError("FGP training reused a likelihood observation")
        posterior = conditional_posterior(sector["fit_values"], sector["fit_mask"],
            sector["stellar_psd"], sector["white_variance"], dt, options["fgp_cutoff_per_day"])
        basis = (operator @ posterior.design) * (sector["scale"]/sector["center"])
        obs = np.asarray(sector["observed"], bool)
        mean += operator @ np.where(obs, sector["cleaned_flux"], 0.)
        mean -= basis @ posterior.coefficient_mean
        map_flux += operator @ np.where(obs, sector["residual_flux"], 0.)
        factors.append(basis @ posterior.coefficient_factor)
    data = dict(period=ephem.period_days, exptime=dt, flux=mean, map_flux=map_flux,
                factor=np.concatenate(factors, axis=1))
    for i, panel in enumerate(PANELS):
        part = slice(i*size, (i+1)*size)
        error = 1/np.sqrt(sums[part])
        if (not np.allclose(map_flux[part], windows["flux_"+panel], atol=2e-12, rtol=0)
                or not np.allclose(error, windows["err_"+panel], atol=0, rtol=1e-10)):
            raise ValueError("Saved folded data do not match the native observation operator")
        data.update({panel+"_time": centers.copy(), panel+"_mean_flux": mean[part],
                     panel+"_map_flux": map_flux[part], panel+"_input_error": error,
                     panel+"_sigma": np.full(size, np.median(error))})
    data["sigma"] = np.concatenate([data[p+"_sigma"] for p in PANELS])
    return data


def candidate_folded_fourier(saved):
    """Fold raw observed aperture flux at 2P and propagate the saved full PSD.

    Uses FFT covariance products; no native dense covariance or per-sector
    inverse is built. The DC prior equals the lowest-frequency total power,
    as in prepare_sector. Measurement white noise enters exactly once.
    """
    ephem, _, segments, _, dt = candidate_state(saved)
    times = np.concatenate([s["time"][s["observed"]] for s in segments])
    flux = np.concatenate([s["raw_flux"][s["observed"]]/s["center"] for s in segments])
    operator, centers, bins, weights = full_orbit_fold(times, ephem.period_days,
                                                      ephem.epoch_relative, dt)
    covariance = np.zeros((len(centers), len(centers)))
    start = 0
    for sector in segments:
        indices = np.flatnonzero(sector["observed"])
        length, stop = len(sector["time"]), start+len(indices)
        scale = float(sector["scale"])/float(sector["center"])
        dc = (sector["stellar_psd"][0]+length*sector["white_variance"])*scale**2
        power = total_fft_power(sector["stellar_psd"], sector["white_variance"],
                                length, scale, dc_power=dc)
        sector_operator = coo_matrix((weights[start:stop], (bins[start:stop], indices)),
                                     shape=(len(centers), length)).tocsr()
        covariance += propagate_fourier_covariance(power, length, sector_operator)
        start = stop
    folded_flux = 1+np.asarray(operator @ (flux-1)).ravel()
    block = GaussianBlock.from_covariance(centers, folded_flux, covariance, dt)
    return FoldedFourierData(block, times, bins, weights, ephem.period_days, ephem.epoch_relative)
