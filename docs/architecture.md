# Architecture and migration boundary

## What was extracted

Numerical source was copied from the current research working tree into
separate modules. Source-to-destination paths, source hashes, initial imported
hashes, and the dirty source Git state are recorded in `source_manifest.json`.
That manifest describes the initial extraction, not hashes of later package
edits. It is not an instruction to overwrite the preserved source checkout.

| Research source | Package module |
| --- | --- |
| `triceratops_new.py` | `_target.py` (compatibility implementation) |
| `likelihoods.py`, `likelihoods_fourier.py` | `likelihoods.real`, `likelihoods.fourier` |
| Real marginal-likelihood modules | `evidence.real`, `evidence.eclipses` |
| Fourier marginal-likelihood modules | `evidence.fourier`, `evidence.fourier_eclipses` |
| `persistent.py`, `scenario_sampling.py` | `sampling.persistent`, `sampling.policy` |
| `funcs.py`, `priors.py` | `stellar`, `priors` |
| Vendored `stecomp` | `companions` |
| TESS flux-frame and FGP helpers | `preprocessing.flux`, `preprocessing.fgp` |
| Conditional FGP posterior | `preprocessing.posterior` |
| Covariance and batched CPU adapters | `experimental` |

Original Triceratops and the intermediate persistent-sampling snapshot stay
in the research repository as independent references, not runtime dependencies.

## Package-only changes

- Relative imports and bundled-table paths follow the new module layout.
- Catalog imports are deferred until a catalog function is actually used.
- Context-local observers record evidence outputs and full sampler pools,
  without resampling or consuming random numbers.
- A one-dimensional sampler shape error is fixed: `np.cov` returns a scalar
  for one parameter, so the proposal covariance uses `np.atleast_2d`.
  Multi-parameter proposal values are unchanged.
- Result/CLI interfaces validate explicit FPP weighting and save portable
  bundles. The real-space wrapper restores posterior-count globals after calls.
- Experimental matched-window adapters accept a positive integer exposure
  subsample count instead of hard-coding 20 in the fast path. The default is
  unchanged, and the fast adapter still requires fixed P/2P. See
  [exposure integration](exposure.md).
- Download-first examples resolve NASA TOI/KOI ephemerides (or explicitly
  selected Bayesian TESS ephemerides), preserve catalogue snapshots and flux
  provenance, and use identical FGP-protection and folded-window assignments.
  These are preparation-only interfaces, not benchmark-runner replacements.
- FGP outputs retain the final MAP objective's input, PSD, and white variance
  for later replay. If optional `celerite` initialization is unavailable or
  fails, smoothed interpolation replaces the old unsmoothed fallback: passing
  exactly through every training observation left zero residual variance and
  prevented the first Gaussianization step. The successful `celerite` path
  and subsequent Fourier MAP objective are unchanged.

Reference tests compare numerical function bodies (allowing the documented
shape fix), physical templates, a seeded sampler, and small real/Fourier TP
evidence calculations. These are migration tests, not a claim that every
scientific assumption or every scenario has been validated.

## Scientific conventions

### Flux frame and preprocessing

The validation model applies aperture dilution itself. Normalized PDCSAP must
first be restored to the aperture frame, per sector:

```text
f_aperture = 1 + CROWDSAP * (f_PDCSAP - 1)
sigma_aperture = CROWDSAP * sigma_PDCSAP
```

Do this before PSD estimation, detrending, or folding. Do not apply it again to
an already-restored cache. PSD/covariance scales quadratically. Normalized SAP
uses restoration factor one. Missing flux provenance should not be guessed.

The [candidate examples](../examples/README.md) implement sector selection,
other-planet masking, matched protection/input windows, binning, and explicit
coverage gates. They are not a complete TIC-to-FPP runner or a reproduction
of every historical research preprocessor. They retain MAP state, but do not
yet propagate FGP covariance to the returned windows. Those choices cannot
be inferred from a single folded light curve.

### Real-space compatibility versus corrected benchmarks

`Target` preserves the extracted `triceratops_new` dispatcher's behavior,
including the historical one-factor noise normalization in `evidence.real`,
rather than replacing it with a full Gaussian determinant. Per-host and
per-data-set evidence-frame comparisons require consistent outer bookkeeping.

**A bare `Target.calc_probs` call is not a drop-in replacement for a corrected
fixed-trend or covariance benchmark runner.** The package warns at this entry
point. Do not silently recompute publication FPPs from its raw `lnZ` column
across incompatible host frames.

`experimental.covariance.GaussianMetric` implements a same-data, flat-model-null
log-likelihood ratio with covariance `diag(sigma**2) + U @ U.T`, preserving
cross-window covariance. End-to-end use requires the matching model adapter,
window ordering, and corrected preprocessing. PSD/noise hyperparameters are
fixed in the conditional FGP posterior; their uncertainty is not integrated out.

### Fourier likelihood

Retained coefficients are the complex positive-frequency coefficients of
NumPy's unnormalized real FFT. DC and the purely real Nyquist coefficient
(for even lengths) are omitted. The residual cost is

```text
sum(abs(data_ft - model_ft)**2 / var_ft)
```

There is no extra factor one-half for complex coefficients. The normalization
is `-sum(log(pi * var_ft))`. Supply powers in this FFT convention, not an
arbitrary density per unit frequency. The dispatcher retains effective-transit
scaling and separate full/even/odd PSDs; no PSD calibration is added here.

The final table's `lnBF` is null-referenced. The null includes the actual data
residual term, not just a determinant. Half-split/even-odd frame alignment is
also retained. Use the comparable final column, not a raw per-sampler
`log_evidence`, for scenario probabilities.

### Even/odd and secondary hypotheses

Ordinary scenarios retain concatenated even/odd observations and corresponding
per-point errors; x2P scenarios receive separate grids. The established
trimming and parity-choice conventions remain: this extraction does not turn
the best of two parity assignments into a parity-marginalized likelihood.

The optional secondary-outside-window flat-data treatment and Fourier
anomaly-shift argument are preserved. Preprocessing must protect the data
appropriate to the chosen workflow. The package does not infer a new window
or timing prior from a candidate identifier.

## Execution and thread safety

Use one process per target. Legacy model caches, module-level settings, and
NumPy's global RNG mean the engine is **not thread-safe**. Context-local
recording does not remove this constraint. Experimental adapters also patch
engine globals temporarily and must not run concurrently in threads. CPU
batching remains opt-in under `experimental`.

An explicit `run_evidence(..., seed=...)` temporarily sets and restores the
NumPy RNG state. Without a seed, the ordinary global RNG behavior remains.
Keep BLAS/Numba threads at one when packing target workers. Long sampling and
broad regressions belong in Slurm; small unit tests are local.

## Remaining release gates

1. Extract one corrected benchmark runner with frame/null bookkeeping,
   sector-aware preprocessing, and input validation.
2. Reproduce representative complete TESS/Kepler scenario tables, beyond the
   small TP and kernel migration tests.
3. Make model/data contracts and independent RNGs explicit before claiming
   thread-safe use or enabling batching by default.
4. Review companion-table licensing/attribution and supported dependency
   versions before public distribution.

The package is useful for development and controlled numerical tests; the
preserved research pipelines remain the production reference.
