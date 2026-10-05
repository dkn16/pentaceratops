# Joint Fourier likelihood with gaps

The opt-in `pentaceratops.fourier.calc_probs_joint_fourier` API fits all supplied
observations together. It does not fold at P, split primary/secondary windows,
interpolate science flux, impose an 80% phase-coverage threshold, or infer a
secondary non-detection from missing data. It evaluates P planet/EB models and
2P alternating-EB models at the actual observation times. Both parity
orientations use the same data; `parity="profile"` preserves the existing
maximum-over-parity convention, while `parity="marginalize"` averages their
likelihoods with equal prior weights.

## Observation operator and covariance

For fixed linear observation/transform operator A, apply precisely the same
operator to observations and physical predictions:

    z = A y; mu(theta) = A m(theta); K = A C A.T

Here the transform is real; complex coefficients require a conjugate transpose
and correct treatment of conjugate symmetry. A mask changes both the transformed
mean and the covariance. Using the masked model does not restore independent
Fourier bins. A redundant full FFT of zero-padded missing samples can yield a
singular covariance; it does not create additional information.

This implementation evaluates the equivalent Gaussian in observed time samples.
The noise covariance is still specified by its full Fourier power spectrum.
Let N be the latent regular-grid length and P[k] = E|rFFT(noise)[k]|^2. Then

    c = irfft(P / N, n=N)
    C_observed[i,j] = c[(observed_index[i] - observed_index[j]) % N]

Selecting the observed covariance marginalizes missing values. Selecting a
submatrix of the full-grid *precision* instead would condition on missing
values and is incorrect. The stationary covariance/spectrum relation is
explained in [GPML chapter 4](https://gaussianprocess.org/gpml/chapters/RW4.pdf).
The formulas above are exact for the specified finite periodic Gaussian model.
Monte Carlo draws followed by the same observation operator provide an
independent check; they are not needed to estimate this known covariance.

DC and Nyquist (when present) are included with their real-valued normalization.
`GaussianBlock.from_spectrum` requires explicit positive power for every mode,
including DC. `prepare_sector` uses the lowest nonzero-frequency total power
for DC; this is a finite sector-offset prior, not a claim that the offset is
known or an improper unconstrained mean fit.

## Preparation and API

```python
from pentaceratops.preprocessing.joint_fourier import prepare_sector
from pentaceratops.fourier import calc_probs_joint_fourier

# Repeat for each independent sector. Times and epoch must use the same origin.
block, diagnostics, spectrum = prepare_sector(
    time, raw_aperture_flux, errors,
    period=period_days, epoch=epoch_days, duration=duration_days,
    cadence=cadence_days, exptime=exposure_days,
)
results = calc_probs_joint_fourier(
    target, period_days, blocks=[block], epoch=epoch_days,
    trilegal_fname=shared_population, N=500, steps=50,
    nsamples=7, posterior_samples=2000, parity="profile",
)
```

The existing package FGP routine estimates a fixed sector PSD. Its Gaussianized
and gap-filled internal stream is used for PSD estimation only. The likelihood
receives the supplied raw aperture flux divided by its fixed sector center.
Protection of candidate events applies to PSD training, not removal from the
likelihood. The empirical white-noise floor is included once; formal errors are
not added again. Saved diagnostics compare it with the observed difference
scatter and record the assumptions. Inputs must already occupy a uniform
cadence grid with explicit gaps; rebinning is an explicit upstream operation.
The exposure approximation evaluates each retained bin as a boxcar of the
supplied duration, not an exact replay of individual contributing exposures.

The dispatcher takes the photometric mission from target.mission (TESS,
Kepler, or K2); direct JointFourierAdapter callers should set mission explicitly
for Kepler/K2. This selects the correct limb-darkening table. The optional filt
argument specifies the contrast-curve band and defaults to the mission band.

Alternatively construct blocks directly with
`GaussianBlock.from_spectrum(grid_time, flux, power, exptime, observed=mask)`
or `GaussianBlock.from_covariance(time, flux, covariance, exptime)` to supply a
validated noise model. A list of blocks assumes independence between those
blocks, normally sectors; correlations within each block are retained.
This initial adapter requires a common exposure duration and a fixed period.
The sector PSD and center are fixed empirical inputs, with no marginalization
over their estimation uncertainty. Gaussian noise, within-sector stationarity,
and finite-grid periodic boundaries remain model assumptions.

## Evidence, optimization and outputs

The sampler reuses the package's 15 target-star population/geometry recipes and
3 recipes per eligible neighbor. Legacy window parameters are only compatibility
placeholders: every photometric callback scores the joint observations once.
There is no legacy hard secondary-depth veto. An eclipse is constrained wherever
its modeled flux differs at an actual observation, even away from P/2.

For residual r = y - 1, signal s = m - 1 and Q = C_observed^-1,

    log L(m) - log L(1) = (Q r).T s - 0.5 s.T Q s

The exact sparse evaluation uses only the support of s after precomputing Q r;
the full observed data still enter that vector. Strict optimized exposure
kernels and batched priors accelerate sampling. Results retain `lnBF`, full
normalized `lnZ = lnBF + null_loglike`, weighted sample pools, posterior
resamples, and an independent dense PyTransit replay of the best retained
point. `lnBF` here is evidence relative to the unit-flux null under the same
noise model; it includes the package's scenario occurrence factors. FPP uses
all included scenario evidences, with the existing TP/PTP/DTP planet grouping.
Normalized evidence from different data selections is not a direct method
comparison.

Dense covariance preparation costs O(n^3) time and O(n^2) memory per sector.
It is performed once, not at each physical-model sample. `save_block` and
`load_block(mmap_mode="r")` let isolated workers share read-only precision
arrays. The process-local sampler hooks are not safe for threaded concurrent
calls; use worker processes. `sample_joint_scenario` is available for campaigns
that distribute individual star/scenario tasks across 128 workers.

## Validation and deployment

Focused tests independently cover Fourier-basis covariance, Monte Carlo
covariance, marginalization across gaps, normalized multivariate Gaussians,
DC/Nyquist, complete-grid agreement with the existing Fourier cost for matching
modes, P/2P models and both parity treatments, all 15 physical recipes, optimized
versus scalar calculations, real sampler pool/replay integration, and sector
preparation/serialization. These are numerical correctness tests, not a claim
of posterior convergence or empirical PSD calibration on a particular target.

The existing `calc_probs_fourier` API and frozen HZ comparison remain unchanged.
New HZ results must use a separately frozen campaign and a distinct method name.

## Full-orbit folding

The native observation likelihood can also be compressed with the same folding
operator for data and model, and covariance `A C A.T`. This is implemented by
`prepare_folded_fourier` and `calc_probs_folded_fourier`; see
[folded HZ runs](hz_runs.md) for the adopted conventions, including gaps,
alternating eclipses, exposure integration, and the separate historical recipe.
