"""One diagonal Fourier noise model shared by all folded scenario families.

Full-period parity folds are the authoritative observations. Repeating models
can use their inverse-variance combined sufficient statistic; alternating
models retain both folds. DC and the real Nyquist mode retain the original
omission convention. Gapped/nonstationary inputs require the dense folded API.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from .joint_fourier import JointFourierMetric


def complex_bins(values):
    values = np.asarray(values, float)
    coefficients = np.fft.rfft(values, axis=-1)[..., 1:]
    return coefficients[..., :-1] if values.shape[-1] % 2 == 0 else coefficients


@njit(cache=True, fastmath=False)
def circulant_gains(signals, kernels, projected, offsets):
    """Exact quadratic forms using sparse eclipse support and circulant rows."""
    answer = np.zeros(len(signals))
    for j in range(len(signals)):
        for block in range(len(offsets)-1):
            lo, hi = offsets[block], offsets[block+1]
            width = hi-lo
            active = np.empty(width, dtype=np.int64)
            count = 0
            for i in range(width):
                if signals[j, lo+i] != 0.:
                    active[count] = i
                    count += 1
            linear, quadratic = 0., 0.
            for i in range(count):
                ix = active[i]
                value = signals[j, lo+ix]
                linear += projected[lo+ix]*value
                dot = 0.
                for k in range(count):
                    iy = active[k]
                    dot += kernels[lo+(ix-iy)%width]*signals[j, lo+iy]
                quadratic += value*dot
            answer[j] += linear-.5*quadratic
    return answer


@dataclass
class FourierFold:
    time: np.ndarray
    flux: np.ndarray
    variance: np.ndarray
    exptime: float

    def __post_init__(self):
        self.time, self.flux, variance = [np.asarray(x, float).copy()
                                         for x in (self.time, self.flux, self.variance)]
        n = self.time.size if self.time.ndim == 1 else 0
        if (self.time.ndim != 1 or n < 3 or self.flux.shape != (n,)
                or variance.shape != (n//2,) or not np.isfinite(self.time).all()
                or not np.isfinite(self.flux).all()
                or not np.all(np.isfinite(variance) & (variance > 0))
                or not np.isfinite(self.exptime) or self.exptime <= 0
                or np.any(np.diff(self.time) <= 0)
                or not np.allclose(np.diff(self.time), self.exptime, rtol=1e-7, atol=1e-10)):
            raise ValueError("Require finite full-period flux and positive matching Fourier variances")
        self.variance = variance[:-1] if n % 2 == 0 else variance
        self.coefficients = complex_bins(self.flux-1.)
        spectrum = np.zeros(n//2+1)
        spectrum[1:1+len(self.variance)] = 1/self.variance
        self.kernel = n*np.fft.irfft(spectrum, n=n)
        self.projected = n*np.fft.irfft(spectrum*np.fft.rfft(self.flux-1.), n=n)
        self.null_loglike = float(-np.sum(np.log(np.pi*self.variance))
                                 - np.sum(abs(self.coefficients)**2/self.variance))
        self.logdet = float(np.log(self.variance).sum())
        self.sigma = np.full(n, np.sqrt(2*self.variance.sum())/n)


class FoldedFourierMetric(JointFourierMetric):
    """Full-period FFT metric with no independent primary/secondary split.

    The circulant quadratic form is exactly the retained-mode Fourier
    likelihood ratio. All host dilution is applied to models, not this metric.
    The two-fold implementation assumes independent noise between parities;
    use FoldedFourierData for a supplied covariance with cross-parity terms.
    """
    def __init__(self, folds):
        self.blocks = tuple(folds)
        if not self.blocks or any(not isinstance(b, FourierFold) for b in self.blocks):
            raise ValueError("Require one or two full-period FourierFold objects")
        sizes = [len(b.flux) for b in self.blocks]
        if len(sizes) > 2 or len(set(sizes)) != 1:
            raise ValueError("Parity folds must have identical grid lengths")
        self.length = sizes[0]
        self.offsets = np.r_[0, np.cumsum(sizes)]
        for name in ('time', 'flux', 'sigma', 'kernel', 'projected'):
            setattr(self, name, np.ascontiguousarray(np.concatenate([getattr(b, name) for b in self.blocks])))
        self.null_loglike = sum(b.null_loglike for b in self.blocks)
        self.logdet = sum(b.logdet for b in self.blocks)
        self.common_kernel = np.ascontiguousarray(sum(b.kernel for b in self.blocks))
        self.common_projected = np.ascontiguousarray(sum(b.projected for b in self.blocks))
        self.combined_variance = 1/sum(1/b.variance for b in self.blocks)
        self.combined_coefficients = self.combined_variance*sum(
            b.coefficients/b.variance for b in self.blocks)

    def gains(self, models):
        signals = np.ascontiguousarray(np.asarray(models, float)-1.)
        if signals.ndim != 2 or signals.shape[1] != len(self.flux):
            raise ValueError("Model and Fourier fold shapes differ")
        valid = np.isfinite(signals).all(axis=1)
        values = np.full(len(signals), -np.inf)
        values[valid] = circulant_gains(signals[valid], self.kernel, self.projected, self.offsets)
        return values

    def gain(self, model):
        return float(self.gains(np.asarray(model)[None, :])[0])

    def common_gains(self, models):
        """Score one repeated P model using the sufficient statistic."""
        signals = np.ascontiguousarray(np.asarray(models, float)-1.)
        if signals.ndim != 2 or signals.shape[1] != self.length:
            raise ValueError("Expected a common full-period model")
        valid = np.isfinite(signals).all(axis=1)
        values = np.full(len(signals), -np.inf)
        values[valid] = circulant_gains(signals[valid], self.common_kernel,
            self.common_projected, np.array([0, self.length]))
        return values

    def fft_gain(self, models):
        """Independent reference, including every retained Fourier mode."""
        values = np.asarray(models, float)
        if values.shape != self.flux.shape:
            raise ValueError("Model and Fourier fold shapes differ")
        total = 0.
        for block, lo, hi in zip(self.blocks, self.offsets[:-1], self.offsets[1:]):
            model = complex_bins(values[lo:hi]-1.)
            total += np.sum((2*np.real(np.conj(block.coefficients)*model)-abs(model)**2)/block.variance)
        return float(total)


def prepare_folded_metric(flux, psd, phase, dt, period, ntransits, *,
                          parity_flux=None, n_even=None, n_odd=None,
                          psd_even=None, psd_odd=None):
    """Prepare one shared aperture-frame metric before any host/scenario loop."""
    flux, psd = np.asarray(flux, float), np.asarray(psd, float)
    if (flux.ndim != 1 or len(flux) < 3 or not np.isfinite(flux).all()
            or not np.isfinite([phase, dt, period, ntransits]).all()
            or min(dt, period, ntransits) <= 0
            or not np.isclose(len(flux)*dt, period, rtol=1e-7, atol=1e-10)):
        raise ValueError("Require a finite full-period grid with length*dt=period")
    time = np.arange(len(flux))*dt
    if parity_flux is None:
        if any(value is not None for value in (n_even, n_odd, psd_even, psd_odd)):
            raise ValueError("Parity counts/PSDs require the corresponding parity fluxes")
        return FoldedFourierMetric([FourierFold(time, flux, psd/ntransits, dt)])
    parity_flux = np.asarray(parity_flux, float)
    if parity_flux.shape != (2*len(flux),):
        raise ValueError("Require two matching full-period parity folds")
    if (n_even is None or n_odd is None or not np.isfinite([n_even, n_odd]).all()
            or min(n_even, n_odd) <= 0):
        raise ValueError("Both parity folds require positive contributing-transit counts")
    powers = [psd if psd_even is None else np.asarray(psd_even, float),
              psd if psd_odd is None else np.asarray(psd_odd, float)]
    return FoldedFourierMetric([
        FourierFold(time, parity_flux[:len(flux)], powers[0]/n_even, dt),
        FourierFold(time+period, parity_flux[len(flux):], powers[1]/n_odd, dt),
    ])
