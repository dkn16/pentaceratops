"""Joint Gaussian likelihood induced by a Fourier PSD on observed samples.

The latent noise lives on a regular, optionally padded grid. Missing samples
are marginalized by selecting the observed covariance, never by interpolating
flux or setting residuals in gaps to zero. All rFFT modes (DC and Nyquist
included) have their exact real-data normalization. P[k] = E|rFFT(noise)[k]|^2.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def observed_covariance(power, length, indices):
    """Return S F^-1 diag(P/N) F S.T for an observation selection S."""
    if isinstance(length, bool) or not isinstance(length, Integral) or length < 2:
        raise ValueError("length must be an integer >= 2")
    power = np.asarray(power, float)
    indices = np.asarray(indices)
    if (power.shape != (length // 2 + 1,)
            or not np.all(np.isfinite(power) & (power > 0))):
        raise ValueError("Require positive finite rFFT power including DC and Nyquist")
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
            or np.any(indices < 0) or np.any(indices >= length)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("Require unique observed grid indices in range")
    kernel = np.fft.irfft(power / length, n=length)
    covariance = kernel[(indices[:, None] - indices[None, :]) % length]
    return .5 * (covariance + covariance.T)


def total_fft_power(stellar_power, white_variance, length, relative_scale, *, dc_power):
    """Combine stellar power and one white-noise floor in relative-flux units.

    Stellar power excludes DC. dc_power is explicitly supplied in the SAME
    final relative-flux FFT-power units; this function invents no offset prior.
    Formal measurement errors must not be added again when using this floor.
    """
    stellar_power = np.asarray(stellar_power, float)
    if (stellar_power.shape != (length // 2,)
            or not np.all(np.isfinite(stellar_power) & (stellar_power >= 0))):
        raise ValueError("Invalid stellar rFFT power")
    if not np.all(np.isfinite([white_variance, relative_scale, dc_power])) or min(
            white_variance, relative_scale, dc_power) <= 0:
        raise ValueError("Invalid white variance, scale, or explicit DC power")
    return np.r_[dc_power, (stellar_power + length * white_variance) * relative_scale**2]


@dataclass
class GaussianBlock:
    """One independent noise block, preserving its actual observation times."""
    time: np.ndarray
    flux: np.ndarray
    precision: np.ndarray
    projected_residual: np.ndarray
    sigma: np.ndarray
    logdet: float
    null_loglike: float
    exptime: float

    @classmethod
    def from_covariance(cls, time, flux, covariance, exptime):
        time, flux, covariance = map(lambda a: np.asarray(a, float), (time, flux, covariance))
        n = time.size if time.ndim == 1 else 0
        if (time.ndim != 1 or not n or flux.shape != (n,) or covariance.shape != (n, n)
                or not np.isfinite(time).all() or not np.isfinite(flux).all()
                or not np.isfinite(covariance).all() or not np.isfinite(exptime) or exptime <= 0):
            raise ValueError("Invalid observed-data block")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0):
            raise ValueError("Covariance must be symmetric")
        lower = cho_factor(covariance, lower=True, check_finite=False)
        precision = cho_solve(lower, np.eye(n), check_finite=False)
        precision = .5 * (precision + precision.T)
        residual = flux - 1.
        projected = precision @ residual
        logdet = float(2 * np.log(np.diag(lower[0])).sum())
        null = float(-.5 * (residual @ projected + logdet + n * np.log(2*np.pi)))
        return cls(time.copy(), flux.copy(), precision, projected,
                   np.sqrt(np.diag(covariance)), logdet, null, float(exptime))

    @classmethod
    def from_spectrum(cls, grid_time, flux, power, exptime, *, observed=None):
        """Accept a full regular grid with explicit gaps and arbitrary padding.

        With an explicit mask, flux outside it is ignored, even if finite.
        Unmasked nonfinite values fail; no valid observation is silently dropped.
        """
        grid_time, flux = np.asarray(grid_time, float), np.asarray(flux, float)
        if (grid_time.ndim != 1 or len(grid_time) < 2 or flux.shape != grid_time.shape
                or not np.isfinite(grid_time).all()):
            raise ValueError("Require matching regular-grid time and flux arrays")
        spacing = np.diff(grid_time)
        if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-10):
            raise ValueError("PSD grid must be uniformly spaced; represent gaps with observed")
        if observed is None:
            observed = np.isfinite(flux)
        else:
            observed = np.asarray(observed)
            if observed.dtype != bool or observed.shape != flux.shape:
                raise ValueError("observed must be a matching boolean mask")
        ids = np.flatnonzero(observed)
        return cls.from_covariance(grid_time[ids], flux[ids],
                                   observed_covariance(power, len(grid_time), ids), exptime)


class JointFourierMetric:
    """Normalized Gaussian likelihood with full covariance within each block.

    Independent blocks can represent sectors. There is no primary/secondary
    split inside a block. gain is relative to the same-data unit-flux null.
    """
    def __init__(self, blocks):
        self.blocks = tuple(blocks)
        if not self.blocks or any(not isinstance(b, GaussianBlock) for b in self.blocks):
            raise ValueError("Require one or more GaussianBlock objects")
        self.offsets = np.r_[0, np.cumsum([len(b.flux) for b in self.blocks])]
        self.flux = np.concatenate([b.flux for b in self.blocks])
        self.sigma = np.concatenate([b.sigma for b in self.blocks])
        self.time = np.concatenate([b.time for b in self.blocks])
        self.logdet = sum(b.logdet for b in self.blocks)
        self.null_loglike = sum(b.null_loglike for b in self.blocks)

    def gain(self, model):
        signal = np.asarray(model, float) - 1.
        if signal.shape != self.flux.shape:
            raise ValueError("Model/data ordering or shape differs")
        if not np.isfinite(signal).all():
            return -np.inf
        answer = 0.
        for b, lo, hi in zip(self.blocks, self.offsets[:-1], self.offsets[1:]):
            s = signal[lo:hi]
            ids = np.flatnonzero(s)
            if len(ids):
                v = s[ids]
                answer += b.projected_residual[ids] @ v - .5 * v @ b.precision[np.ix_(ids, ids)] @ v
        return float(answer)

    def loglike(self, model):
        return self.null_loglike + self.gain(model)

    def gains(self, models):
        """Exact strict-kernel batching; sparse eclipse support is only a shortcut."""
        from ..experimental.sparse_kernels import quadratic_strict
        signals = np.ascontiguousarray(np.asarray(models, float) - 1.)
        if signals.ndim != 2 or signals.shape[1] != len(self.flux):
            raise ValueError("Expected models shaped (batch, observed samples)")
        valid = np.isfinite(signals).all(axis=1)
        answer = np.full(len(signals), -np.inf)
        answer[valid] = 0.
        for b, lo, hi in zip(self.blocks, self.offsets[:-1], self.offsets[1:]):
            n = int(hi-lo)
            answer[valid] += quadratic_strict(np.ascontiguousarray(signals[valid, lo:hi]),
                b.precision.ravel(), b.projected_residual, np.array([0,n]), np.array([0,n*n]))
        return answer
