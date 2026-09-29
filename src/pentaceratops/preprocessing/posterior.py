"""Exact conditional posterior of the low-frequency FGP coefficients.

PSD, white variance, masks, normalization, and Gaussianized data are fixed.
Matches tess_test.fgp_detrend.fit_fourier_modes, whose objective is minus
twice the log posterior, not minus the log posterior.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular


def fourier_design(length, count):
    """Map [Re(rFFT[k]), Im(rFFT[k])] to flux, k starting at 1."""
    if count < 1 or 2 * count >= length:
        raise ValueError("Require positive complex modes strictly below Nyquist.")
    phase = 2 * np.pi * np.outer(np.arange(length), np.arange(1, count + 1)) / length
    return (2.0 / length) * np.concatenate((np.cos(phase), -np.sin(phase)), axis=1)


@dataclass
class FourierPosterior:
    design: np.ndarray
    prior_sigma: np.ndarray
    whitened_mean: np.ndarray
    precision_cholesky: np.ndarray
    white_variance: float

    @property
    def coefficient_mean(self):
        return self.prior_sigma * self.whitened_mean

    @property
    def coefficient_factor(self):
        # C = R R^T, R = diag(prior_sigma) L^{-T}.
        inverse_transpose = solve_triangular(
            self.precision_cholesky.T, np.eye(len(self.prior_sigma)), lower=False
        )
        return self.prior_sigma[:, None] * inverse_transpose

    @property
    def flux_mean(self):
        return self.design @ self.coefficient_mean

    @property
    def flux_factor(self):
        return self.design @ self.coefficient_factor

    def sample(self, size, rng):
        """Independent draws and normalized log p(a | data)."""
        if size < 1:
            raise ValueError("size must be positive")
        noise = rng.standard_normal((len(self.prior_sigma), size))
        perturbation = solve_triangular(self.precision_cholesky.T, noise, lower=False)
        coefficients = self.prior_sigma[:, None] * (self.whitened_mean[:, None] + perturbation)
        log_normalizer = (np.log(np.diag(self.precision_cholesky)).sum()
                          - np.log(self.prior_sigma).sum()
                          - 0.5 * len(self.prior_sigma) * np.log(2 * np.pi))
        return coefficients, log_normalizer - 0.5 * np.sum(noise**2, axis=0)


def conditional_posterior(values, fit_mask, psd, white_variance, dt, cutoff):
    values = np.asarray(values, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    psd = np.asarray(psd, dtype=float)
    if values.ndim != 1 or fit_mask.shape != values.shape:
        raise ValueError("values and mask must be matching one-dimensional arrays")
    if not fit_mask.any() or not np.isfinite(values[fit_mask]).all():
        raise ValueError("Need finite training observations")
    if dt <= 0 or cutoff <= 0 or not np.isfinite(white_variance) or white_variance <= 0:
        raise ValueError("Invalid cadence, cutoff, or white variance")
    maximum = min(len(psd), len(np.fft.rfft(values)) - 1)
    count = min(maximum, max(1, int(cutoff * dt * len(values))))
    power = np.maximum(psd[:count], 1e-12)
    if not np.isfinite(power).all():
        raise ValueError("Nonfinite prior power")
    white_variance = max(float(white_variance), 1e-12)
    design = fourier_design(len(values), count)
    # E|a_k|^2 = P_k, so Var(Re a_k) = Var(Im a_k) = P_k / 2.
    prior_sigma = np.sqrt(np.tile(power, 2) / 2)
    training = design[fit_mask] * prior_sigma
    training /= np.sqrt(white_variance)
    precision = np.eye(2 * count) + training.T @ training
    lower = cholesky(precision, lower=True)
    rhs = training.T @ (values[fit_mask] / np.sqrt(white_variance))
    mean = cho_solve((lower, True), rhs)
    return FourierPosterior(design, prior_sigma, mean, lower, white_variance)
