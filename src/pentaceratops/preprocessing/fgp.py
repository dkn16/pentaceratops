"""Fourier-GP detrending adapted from ``exoplanet/validate_tess``.

This keeps the legacy iterative structure (initial gap model, smoothed
periodogram, low-frequency Fourier MAP fit, and robust gaussianization) while
making the cadence explicit and keeping protected eclipse samples in the
returned residual.  The original code used one mask for both gaps and
transits; doing that would replace a protected transit by the GP itself.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import interpolate, optimize, stats


OUTLIER_DIST = (0.00827352, 1.55134374, 0.86912289, 1.98435008, 0.71418219)


def robust_location_scale(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("cannot normalize fewer than two finite flux samples")
    center = float(np.median(values))
    scale = float(np.std(values))
    correction = 0.9865783925581086
    for _ in range(20):
        if not np.isfinite(scale) or scale <= 0:
            break
        keep = np.abs(values - center) < 3.0 * scale
        if keep.sum() < 2:
            break
        center = float(np.median(values[keep]))
        scale = float(np.std(values[keep]) / correction)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid robust flux scale")
    return center, scale


def gaussianize_correlated_outliers(values):
    """Legacy Gaussian-mixture CDF transform with neighbour protection."""
    values = np.asarray(values, float)
    amplitude, dof, noncentral, location, scale = OUTLIER_DIST
    cumulative = (
        (1.0 - amplitude) * stats.norm.cdf(values)
        + amplitude * stats.nct.cdf(
            values, dof, noncentral, loc=location, scale=scale
        )
    )
    transformed = stats.norm.ppf(np.clip(cumulative, 1e-12, 1.0 - 1e-12))
    normal_pdf = stats.norm.pdf(values)
    outlier_pdf = stats.nct.pdf(
        values, dof, noncentral, loc=location, scale=scale
    )
    mixture = amplitude * outlier_pdf + (1.0 - amplitude) * normal_pdf
    probability_normal = 1.0 - (
        amplitude * outlier_pdf
        / np.maximum(mixture, np.finfo(float).tiny)
    )
    result = np.empty_like(values)
    if len(values) == 1:
        result[0] = transformed[0]
        return result
    neighbour_normal = probability_normal[:-2] * probability_normal[2:]
    result[1:-1] = (
        neighbour_normal * transformed[1:-1]
        + (1.0 - neighbour_normal) * values[1:-1]
    )
    result[0] = (
        probability_normal[1] * transformed[0]
        + (1.0 - probability_normal[1]) * values[0]
    )
    result[-1] = (
        probability_normal[-2] * transformed[-1]
        + (1.0 - probability_normal[-2]) * values[-1]
    )
    return result


def _moving_median(values, half_width):
    output = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - half_width)
        hi = min(len(values), i + half_width + 1)
        output[i] = np.median(values[lo:hi])
    return output


def automatic_bandwidth(power):
    """Port of the adaptive periodogram smoothing bandwidth estimator."""
    power = np.asarray(power, float)
    if len(power) < 70:
        raise ValueError("Fourier-GP requires at least 70 rFFT power bins")
    locations = np.unique(
        np.logspace(1, np.log10(len(power)), min(1000, len(power)), dtype=int)
    )
    locations = np.unique(np.concatenate(([0], locations, [len(power) - 1])))
    maximum_width = max(20, len(power) // 3)
    widths = np.unique(
        np.logspace(np.log10(maximum_width), np.log10(20), 30).astype(int)
    )[::-1]
    optimal = np.full(len(locations), 20, dtype=float)
    for j, location in enumerate(locations):
        for width in widths:
            width = int(min(width, (len(power) - 1) // 2))
            if width < 1:
                continue
            lo = int(np.clip(location - width, 0, len(power) - 2 * width - 1))
            hi = lo + 2 * width + 1
            band = power[lo:hi]
            average = float(np.mean(band))
            if not np.isfinite(average) or average <= 0:
                continue
            coordinate = np.arange(-width, width + 1)
            delta_chi2 = (
                np.sum(coordinate * (band / average - 1.0)) ** 2
                / np.sum(coordinate ** 2)
            )
            if delta_chi2 < 1.0:
                optimal[j] = width
                break
    optimal = _moving_median(optimal, 3)
    if len(locations) >= 4:
        spline = interpolate.UnivariateSpline(
            locations, np.log(optimal), s=len(locations) / 10.0
        )
        result = np.exp(spline(np.arange(len(power))))
    else:
        result = np.interp(np.arange(len(power)), locations, optimal)
    return np.clip(result.astype(int), 1, max(1, (len(power) - 1) // 2))


def smooth_periodogram(power, bandwidth):
    power = np.asarray(power, float)
    bandwidth = np.asarray(bandwidth, int)
    smoothed = np.empty_like(power)
    for i, width in enumerate(bandwidth):
        width = min(int(width), (len(power) - 1) // 2)
        lo = int(np.clip(i - width, 0, len(power) - 2 * width - 1))
        hi = lo + 2 * width + 1
        smoothed[i] = np.mean(power[lo:hi])
    return smoothed


def estimate_psd(values, bandwidth=None):
    power = np.abs(np.fft.rfft(values)) ** 2
    if bandwidth is None:
        bandwidth = automatic_bandwidth(power)
    # Drop DC. The retained array includes Nyquist for an even-length series.
    return smooth_periodogram(power, bandwidth)[1:], bandwidth


def modes_to_flux(parameters, length):
    count = len(parameters) // 2
    coefficient = parameters[:count] + 1j * parameters[count:]
    # The DC coefficient is fixed to zero; parameter zero is Fourier k=1.
    return np.fft.irfft(np.concatenate(([0.0j], coefficient)), n=length)


def fit_fourier_modes(values, fit_mask, psd, white_variance, dt, cutoff):
    """MAP low-frequency Fourier GP with the corrected DC indexing."""
    values = np.asarray(values, float)
    fit_mask = np.asarray(fit_mask, bool)
    maximum = min(len(psd), len(np.fft.rfft(values)) - 1)
    count = min(maximum, max(1, int(cutoff * dt * len(values))))
    coefficient = np.fft.rfft(values)[1:count + 1]
    initial = np.concatenate((coefficient.real, coefficient.imag))
    prior_power = np.maximum(np.asarray(psd[:count], float), 1e-12)
    prior_double = np.concatenate((prior_power, prior_power))
    inverse_mask = fit_mask.astype(float)
    white_variance = max(float(white_variance), 1e-12)

    def objective(parameters):
        difference = (modes_to_flux(parameters, len(values)) - values) * inverse_mask
        inverse_fourier = np.fft.ifft(difference)[1:count + 1]
        gradient = (
            4.0 * np.concatenate(
                (inverse_fourier.real, -inverse_fourier.imag)
            ) / white_variance
            + 4.0 * parameters / prior_double
        )
        value = (
            np.sum(difference ** 2) / white_variance
            + 2.0 * np.sum(parameters ** 2 / prior_double)
        )
        return float(value), gradient

    parameters, _, info = optimize.fmin_l_bfgs_b(objective, x0=initial)
    if info.get("warnflag", 0):
        warnings.warn(f"Fourier-GP optimizer warning: {info.get('task')}")
    return modes_to_flux(parameters, len(values)), count


def initial_gap_model(time, values, fit_mask):
    """Use the legacy celerite initializer, with smoothed interpolation fallback."""
    try:
        import celerite

        class RotationTerm(celerite.terms.Term):
            parameter_names = ("log_amp", "log_timescale", "log_period")

            def get_real_coefficients(self, params):
                _, log_timescale, _ = params
                return 0.0, np.exp(-log_timescale)

            def get_complex_coefficients(self, params):
                log_amp, log_timescale, log_period = params
                return (
                    np.exp(log_amp), 0.0, np.exp(-log_timescale),
                    2.0 * np.pi * np.exp(-log_period),
                )

        initial = (
            (0.25688231, 5.83340667, 15.19218574),
            (0.15927505, 5.19432664, 7.21339376),
            (0.07445868, 7.34722188, 4.65003445),
        )
        kernel = celerite.terms.JitterTerm(log_sigma=0.0, bounds=[(-5, 2)])
        for amplitude, timescale, period in initial:
            log_amp, log_timescale, log_period = np.log(
                (amplitude, timescale, period)
            )
            bounds = {
                "log_amp": (log_amp - 0.1, log_amp + 0.1),
                "log_timescale": (log_timescale - 0.1, log_timescale + 0.1),
                "log_period": (log_period - 0.1, log_period + 0.1),
            }
            kernel += RotationTerm(
                log_amp, log_timescale, log_period, bounds=bounds
            )
        gp = celerite.GP(kernel, mean=0.0, fit_mean=False)
        gp.compute(time[fit_mask])

        def loss(parameters):
            gp.set_parameter_vector(parameters)
            return -gp.log_likelihood(values[fit_mask])

        result = optimize.minimize(
            loss, gp.get_parameter_vector(), method="L-BFGS-B",
            bounds=gp.get_parameter_bounds(),
        )
        gp.set_parameter_vector(result.x)
        return gp.predict(values[fit_mask], time, return_cov=False)
    except Exception as error:
        warnings.warn(
            f"celerite Fourier-GP initializer failed ({error}); using smoothed interpolation"
        )
        from scipy.ndimage import gaussian_filter1d
        index = np.arange(len(values))
        interpolated = np.interp(index, index[fit_mask], values[fit_mask])
        # Pure interpolation passes through every training observation, leaving
        # exactly zero residual scatter for the first gaussianization step.
        # Smooth only this training-data initializer; the subsequent PSD/MAP
        # fit is unchanged, and protected observations never enter this step.
        return gaussian_filter1d(interpolated, sigma=3., mode="nearest")


def gaussianize_residual(values, model, observed, protected):
    residual = values - model
    usable = observed & ~protected & np.isfinite(residual)
    center, scale = robust_location_scale(residual[usable])
    standardized = (residual - center) / scale
    cleaned = gaussianize_correlated_outliers(standardized) * scale + center
    output = model + cleaned
    # Candidate eclipse samples are science data, not outliers to gaussianize.
    output[protected & observed] = values[protected & observed]
    return output


def detrend_regular_segment(
    time, flux, error, protected, dt, iterations=3, cutoff_per_day=3.0,
):
    """Return cleaned flux, FGP model, residual flux, and diagnostics."""
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    error = np.asarray(error, float)
    observed = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    protected = np.asarray(protected, bool) & observed
    fit_mask = observed & ~protected
    if fit_mask.sum() < 140:
        raise ValueError("too few out-of-eclipse samples for Fourier-GP detrending")
    center, scale = robust_location_scale(flux[fit_mask])
    normalized = np.zeros_like(flux)
    normalized[observed] = (flux[observed] - center) / scale
    normalized_error = error / scale

    model = initial_gap_model(time, normalized, fit_mask)
    working = normalized.copy()
    working[~observed] = model[~observed]
    cleaned = gaussianize_residual(working, model, observed, protected)
    for _ in range(int(iterations)):
        psd, bandwidth = estimate_psd(cleaned)
        white_variance = float(np.nanmedian(psd[-max(5, len(psd) // 20):]))
        stellar_psd = np.maximum(psd - white_variance, 1e-8)
        fit_values = cleaned.copy()
        model, mode_count = fit_fourier_modes(
            cleaned, fit_mask, stellar_psd, white_variance / len(cleaned),
            dt, cutoff_per_day,
        )
        cleaned = gaussianize_residual(working, model, observed, protected)
        working[~observed] = model[~observed]

    # ``flux`` entered this routine after division by the sector median, but
    # rebinning and the out-of-eclipse selection give a second robust baseline
    # ``center`` that need not be exactly one.  Undo both internal transforms
    # exactly: in native units this is
    #   1 + (Gaussianized variation - FGP variation) / flux_baseline.
    # Here ``scale*(...)`` is in the once-normalized flux units and ``center``
    # is that stream's baseline, hence the additional division by center.
    if not np.isfinite(center) or center <= 0:
        raise ValueError("invalid Fourier-GP flux baseline")
    cleaned_flux = 1.0 + (scale / center) * cleaned
    model_flux = (scale / center) * model
    residual_flux = 1.0 + (scale / center) * (cleaned - model)
    relative_error = error / center
    return {
        "cleaned_flux": cleaned_flux,
        "model_flux": model_flux,
        "residual_flux": residual_flux,
        "relative_error": relative_error,
        "normalized_error": normalized_error,
        "fit_mask": fit_mask,
        "observed": observed,
        "protected": protected,
        "scale": scale,
        "center": center,
        "mode_count": mode_count,
        "bandwidth_median": float(np.median(bandwidth)),
        # Preserve the exact fixed-PSD objective for later conditional-GP
        # uncertainty propagation. Final cleaned values differ from fit_values
        # because gaussianization runs again after the last MAP optimization.
        "fit_values": fit_values,
        "stellar_psd": stellar_psd,
        "white_variance": white_variance / len(cleaned),
        "psd_total": psd,
    }
