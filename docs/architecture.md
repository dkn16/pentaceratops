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
- PyTransit 2.9.2 replaces the old orbital approximation with MeepMeep. The
  sparse backend now shares its coefficient solver, separation function, and
  asymmetric contact bounds. Scalar calls use the supported `evaluate` API.
  Result records include PyTransit/MeepMeep/Numba versions. This dependency
  change is not bitwise identical to historical runs; see [upgrade checks](pytransit.md).
- All real/Fourier evidence wrappers compose shared host, planet/binary, and
  orbit components, including x2P and legacy twin branches. Fast paths read explicit
  model fields rather than inferring the scenario from closure variables and
  function names. Scientific conventions and public signatures are preserved.
  TP additionally shares its sampler call and posterior packing; the other
  migrated wrappers retain their original result-packing policies.

Reference tests compare unmigrated numerical function bodies (allowing the
documented shape fix and `evaluate_ps` to `evaluate` API rename), physical
templates, a seeded sampler, and tiny evidence calculations for every wrapper.
For the migrated wrappers, differential prior/likelihood/result tests replace
body-identity checks; signature checks remain. These are migration tests, not
a claim that every scientific assumption or every scenario has been validated.

## Shared scenario components

Scenarios share code along independent host, system, and orbit axes, without
a scenario-class inheritance tree:

| Host component | Planet | Binary with secondary window | Even/odd at 2P |
| --- | --- | --- | --- |
| `KnownHost` | TP | EB | EBx2P |
| `DilutedBoundHost` | PTP | PEB | PEBx2P |
| `BoundCompanionHost` | STP | SEB | SEBx2P |
| `BackgroundHost`, target eclipsed | DTP | DEB | DEBx2P |
| `BackgroundHost`, background eclipsed | BTP | BEB | BEBx2P |
| `KnownHost`, nearby-star inputs | NTP | NEB | NEBx2P |

The specialized unknown-property N wrappers use `UnknownHost` with a discrete
magnitude-selected population. Evolved N wrappers use a `KnownHost` with the
inherited logg=3 mass inference. Their legacy primary-only EB/twin branches
are also composed; no new secondary/even-odd API is invented for these fallbacks.

- `models.hosts` prepares stellar properties, limb-darkening lookup, optional
  companion populations, target-relative dilution, and contrast-curve inputs.
  Catalogue/table preparation happens once, outside likelihood evaluation.
- `models.populations` supplies discrete background/neighbor selection,
  catalogue LDC lookup, distance-corrected secondary light, and background
  weights. Population order and the historical base-10 prior are retained.
- `models.systems` supplies `Planet`, `Binary`, shared orbital geometry, and
  their composition with a host as `Scenario`. The system selects its size
  prior and the appropriate companion-weight function; SEB contrast weighting
  includes both eclipsing stars, whereas STP uses only its companion host.
- `OrbitPolicy` records candidate-to-orbit period scaling, collision rules,
  legacy mass-ratio cuts, and whether observations are even/odd windows.
  Eccentricity is drawn at the candidate period before doubling. Windowed
  x2P recipes have no q=.95 split; legacy single/twin recipes do. Windowed
  B-family x2P uses the sum of component radii for collision rejection, while
  T/P/S/D use twice the host radius. Legacy twins all retain twice-host-radius
  rejection. None of these differences is silently harmonized.
- `evidence.scenario.ScenarioPrior` maps the same five or six unit-cube
  coordinates to physical parameters. Inclinations remain isotropic before
  geometry cuts; fixed period still occupies a coordinate. No prior
  conditioning or demographic reweighting is introduced.
- `ScenarioLikelihood` supplies explicit observation/noise, normalization,
  residual-cost callback, and optional secondary-window inputs. The same
  physical derivation serves scalar and batched evaluations. Real-space
  standard deviations and Fourier complex variances remain distinct.
- `evidence.target_planet` is now a small construction/packing facade over
  these ingredients, not a new public runner API. Other result-packing blocks
  stay in their wrappers: their resampling, best-fit placement,
  posterior counts, and fallback policies are intentionally not unified yet.
  Full weighted pools are still recorded externally.

All 52 public evidence functions in the four evidence modules are migrated,
including the duplicated specialized-N compatibility entry points. Signatures, Fourier
preparation, and domain-specific normalizations remain in the wrappers.
Experimental covariance hooks still supply the residual callback and null
normalization; the shared model does not choose an evidence frame. The original
fast real-space adapter remains fixed-P. The prepared-input HZ interfaces also
integrate the full-orbit folded Fourier adapter; see [folded HZ runs](hz_runs.md).

Compatibility is explicit where recipes differ. Fourier STP and SEB retain
their rounded limb-darkening grids and respective 10,000/13,000 K ceilings;
real-space companion hosts retain nearest-available coefficients. STP's radius
prior uses the companion mass, while SEB's mass-ratio prior and secondary
temperature cap retain the target-star inputs used by the reference engine.
This is preservation, not an endorsement or correction of those differences.

BTP still uses the target mass for its radius prior, unlike STP's companion
mass. Unknown/evolved NEB keeps its 1-Msun mass-ratio-prior input. A rejected
unknown host stays in the population prior: the logg/temperature cut belongs
to the likelihood, not a renormalized catalogue. Empty nearby populations
retain the original impossible-evidence outputs.

Fast dispatch uses explicit metadata for all standard T/P/S/D/B combinations
and known nearby stars, and is tested for the specialized N planet recipes.
The closure adapters remain only for compatibility with older callbacks and
reference comparisons. Legacy primary-only binary recipes and Fourier
callbacks are explicitly rejected by the fast real-space adapter.

Remaining structural work is to consolidate recipe-specific input preparation
and result packing without erasing their historical output differences. Do not
infer a new prior or collision rule merely from similar transit templates.

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
window ordering, and corrected preprocessing. `load_folded_real` checks the
paired caches and `calc_probs_folded_real` supplies the optimized adapter,
aperture-frame comparison, null reference, and probability reporting.
PSD/noise hyperparameters are
fixed in the conditional FGP posterior; their uncertainty is not integrated out.

### Fourier coefficient conventions

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

The full-orbit folded interface instead retains the native PSD's Gaussian
covariance through the observed folding operator, with the same operator on
exposure-integrated models. Empty bins are absent and no additional modes are
removed. The separate historical folded-covariance interface preserves the
original model grids and explicit nuisance-mode projection. Both conventions
and their replay checks are described in [folded HZ runs](hz_runs.md).

### Even/odd and secondary hypotheses

Ordinary scenarios retain concatenated even/odd observations and corresponding
per-point errors; x2P scenarios receive separate grids. Archived
trimming and parity-choice conventions remain available: this extraction does not turn
the best of two parity assignments into a parity-marginalized likelihood.
The inherited Fourier parity-swap kernel requires matching even/odd grids,
as prepared by its dispatcher; direct calls with unequal grids are unsupported.
Legacy Fourier primary-only EB recipes also retain a scalar time-domain
`sigma_veto` for the secondary-depth veto, separate from their Fourier powers.

The ordinary EB secondary-outside-window flat-data treatment is preserved
under both timing policies. The Fourier anomaly-shift argument remains
available for explicit reproduction. Standard recorded folded calls use
[observed timing](timing.md), including partial x2P eclipse overlap and
complete full-period Fourier EB models. Preprocessing must protect the data
appropriate to the chosen workflow. The package does not infer a new window
or timing prior from a candidate identifier.

## Execution and thread safety

Use one process per target. Legacy model caches, module-level settings, and
NumPy's global RNG mean the engine is **not thread-safe**. Context-local
recording does not remove this constraint. Experimental adapters also patch
engine globals temporarily and must not run concurrently in threads. CPU
batching remains opt-in under `experimental` for low-level compatibility use;
the prepared-input HZ APIs select their optimized adapters explicitly.

An explicit `run_evidence(..., seed=...)` temporarily sets and restores the
NumPy RNG state. Without a seed, the ordinary global RNG behavior remains.
Keep BLAS/Numba threads at one when packing target workers. Long sampling and
broad regressions belong in Slurm; small unit tests are local.

## Remaining release gates

1. Integrate catalogue-to-input orchestration with the prepared-input folded
   inference interfaces; preserve explicit selection and noise-model provenance.
2. Extend saved-production replay to fresh seeded full-effort scenario tables
   across representative TESS/Kepler targets and assess evidence convergence.
   Saved real fits for all seven retained TESS targets and both adopted Fourier
   folding updates are covered by the current replay tool.
3. Make model/data contracts and independent RNGs explicit before claiming
   thread-safe use or enabling batching by default.
4. Review companion-table licensing/attribution and supported dependency
   versions before public distribution.

The package is useful for development and controlled numerical tests; the
preserved research pipelines remain the production reference.
