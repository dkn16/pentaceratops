# Initial software milestone: 2026-09-28

Version: `0.1.0.dev0`. This is a development milestone, not a public release or
an end-to-end scientific validation of all production scenarios.

## Checks completed

- **71 tests passed**, zero failures, against the unpacked standalone wheel.
  The final test execution took 23.63 seconds with single-threaded numerical
  libraries. No production sampling or catalog queries were run.
- Small synthetic real-space and Fourier-space TP calculations reproduce the
  preserved engine's evidence and returned physical posterior arrays exactly
  at the same seed. Their full weighted pools and log-target values survive
  save/load round trips.
- Numerical function-body comparisons cover the real/Fourier likelihoods,
  scenario evidence modules, prior module, sampler, Fourier dispatcher, and
  sampling policy. The documented one-dimensional covariance shape fix is
  explicitly accounted for in the reference test.
- Dilution restoration, even/odd grid/error handling, Fourier Parseval
  normalization, conditional FGP covariance, null-referenced Gaussian costs,
  explicit demographic weighting, and result persistence have dedicated tests.
- The wheel imports independently of the research checkout and contains its
  required small tables. Importing it does not import Lightkurve, Astroquery,
  or the old Triceratops package.
- **All 32 original imported source files** still match their extraction
  SHA-256 hashes. The research code was not modified.

The local wheel is retained in `dist/` (ignored by Git):

```text
pentaceratops-0.1.0.dev0-py3-none-any.whl
SHA-256: 801f5167b7bbd9360fc03c095aabfe1415c59ce2550a4101fdda4e1ee1c0031e
```

`wheel-tests.xml` is the corresponding local pytest report. Re-run using
`tools/verify_distribution.py --tests tests --reference-root ...`; the
developer guide gives the full command and cache settings.

## Tested environment

Python 3.13; NumPy 2.2.6; SciPy 1.16.0; pandas 2.3.1; Astropy 7.1.0;
PyTransit 2.6.14; Matplotlib 3.10.3; MechanicalSoup 1.4.0; BeautifulSoup 4.13.4;
setuptools 80.9.0. Optional catalog packages were installed in this environment
but were not needed by the tests: Astroquery 0.4.10 and Lightkurve 2.5.1.

The suite emitted 249 inherited deprecation warnings: one PyTransit
`pkg_resources` warning and 124 array-to-scalar conversion warnings from each
of the old and extracted transit-model implementations. These are documented
compatibility work, not failed numerical comparisons.

## Not yet established

- Dependency resolution across fresh environments or a multi-version CI matrix.
- Complete corrected benchmark-runner equivalence for all scenarios/targets.
- Thread-safe execution, automatic failed-run checkpoints, or default batching.
- Companion-table licensing/attribution clearance for public redistribution.

The next integration milestone is to move the corrected benchmark runner's
explicit data/frame/null bookkeeping into the package and compare complete
scenario tables before changing any production imports.

## Exposure extension: `0.1.0.dev1` (2026-09-28)

The experimental matched-window adapters now accept configurable positive
integer exposure subsample counts, while keeping 20 as the default and
retaining the fast path's fixed-P/2P constraint. See
[exposure integration](exposure.md) for usage and limitations.

- **139 tests passed** against the unpacked updated wheel in 21.82 seconds,
  including 68 new exposure tests and all 71 existing tests.
- The new tests cover scalar/fast agreement for planet, secondary-EB, and
  even/odd models, including explicit overrides in actual TP/EB/EBx2P evidence
  closures. The default matches explicit 20-subsample evaluation exactly.
- Wheel import isolation and bundled-table checks passed. All 32 preserved
  research source files still match their extraction hashes.
- The same environment was used, without installing/upgrading dependencies.
  The expanded tests emitted 1017 inherited dependency/array-to-scalar
  deprecation warnings; none were numerical test failures.

Retained artifacts:

```text
dist/pentaceratops-0.1.0.dev1-py3-none-any.whl
SHA-256: 0ccb6bb493eaeee264c0609227e087bc75e8175c1026afe4dd79f10c101aab85
docs/exposure-wheel-tests.xml
```

The initial `dev0` artifacts remain intact. No production sampling or catalog
queries were run. These checks establish numerical equivalence at the selected
integration count, not convergence of that count or full benchmark equivalence.

## Download/preparation examples: `0.1.0.dev2` (2026-09-29)

The [candidate examples](../examples/README.md) default to NASA Exoplanet
Archive ephemerides, with an explicitly selected Bayesian catalogue option
for TESS. They prepare MAP-detrended windows, not all-scenario evidence or
FGP-covariance likelihood inputs.

- **167 tests passed** against the standalone wheel in 21.47 seconds with
  single-threaded numerical libraries; the source suite also passed. The
  existing 1017 inherited deprecation warnings remain.
- The 28 added tests cover identifier and catalogue selection, time units,
  companion masks, catalogue snapshots/offline reuse, product selection,
  per-product dilution restoration, identical protection/folding assignments,
  coverage failures, prepared-cache/plot replay, and saved final FGP fit state.
- A synthetic actual-FGP test caught and now exercises the repaired fallback
  when optional `celerite` cannot initialize. Unsmoothed interpolation had
  left zero residual variance; the fallback now smooths training data only.
- Live NASA lookups succeeded for TOI-4616.01 / TIC 258796169 and
  KOI-2841.01 / KIC 8758204. A live Bayesian lookup succeeded for TOI-4616.01,
  using its absolute `Epoch` and `Duration` fields rather than `Phase`/`Tau`.
- Download-only checks fetched TESS Sector 17 and Kepler Quarter 4 for those
  targets and verified product identity, times, and aperture restoration.
  All catalogue/FITS/native caches are under scratch. These were **not**
  full real-data FGP or FPP runs; real-target detrending still needs inspection.
- Wheel import isolation, bundled-table checks, undefined-name linting and
  Slurm-template shell syntax passed. All 32 preserved research source files
  still match their extraction hashes. No shared dependencies were changed.

Retained artifacts (earlier versions remain intact):

```text
dist/pentaceratops-0.1.0.dev2-py3-none-any.whl
SHA-256: 10bbada2a3dbe17cbbb4a49676c994d03cea771ce3763c3c0bdb8b875e646bb4
docs/candidate-wheel-tests.xml
```

## PyTransit upgrade: `0.1.0.dev3` (2026-09-29)

- **194 tests passed** against the standalone wheel in 28.00 seconds, using
  PyTransit 2.9.2, MeepMeep 1.1.0 and setuptools 81.0.0 in a scratch-only
  virtualenv overlay. NumPy remains 2.2.6 and Numba 0.61.2. The 1023 warnings
  are inherited scalar conversions and deprecated calls in the preserved
  reference engine; no `pkg_resources` warning occurs in the new stack.
- The dependency bump alone failed scalar/fast EB agreement. The sparse
  backend was then migrated to the same analytic MeepMeep expansion and
  asymmetric contact bounds as PyTransit. Existing likelihood tolerances
  were not relaxed. Scalar calls now use the public `evaluate` API.
- Twenty-five added compatibility tests cover scalar API equivalence and
  strict/fast projection agreement with the public model, including orbital
  wrapping, duplicate times, asymmetric bounds and exposure integration.
  Two additional API tests cover persisted dependency versions and optional
  runtime setuptools reporting.
- Separate old/new environments produced 240 template-array comparisons and
  two tiny seeded TP evidence checks. Largest template differences were
  0.00635 ppm for planets and 0.36581 ppm for EBs. Log-evidence differences in
  those low-SNR checks were below `2e-9`. These are limited numerical checks,
  not a population-level FPP rerun; see [upgrade notes](pytransit.md).
- Import isolation, bundled tables, focused linting and source audit pass.
  All 32 original files are unchanged, as is the `exoprob` dependency stack.
  The new wheel removes the old runtime setuptools pin and requires PyTransit
  2.9.2–2.9.x, MeepMeep 1.1.x, and NumPy 2.0–2.2.

Retained artifacts (earlier wheels remain intact):

```text
dist/pentaceratops-0.1.0.dev3-py3-none-any.whl
SHA-256: b2c6d7dbefd890465a5bbc80fb831e33be5d59e8b554f8bb73a1bac94022ef03
docs/pytransit-wheel-tests.xml
```

## Shared TP model: `0.1.0.dev4` (2026-09-29)

The real/Fourier TP wrappers now compose the same host model, priors, geometry,
sampler call, and result packing. Their public signatures and scientific
normalizations are unchanged. The fast real-space TP path consumes explicit
model fields; other scenarios retain their previous implementations.

- **241 tests passed** against the standalone wheel in 56.40 seconds; the
  source suite also passed (34.11 seconds). Imports and bundled tables work
  independently of the research checkout. The same isolated PyTransit 2.9.2
  stack was used, without changing the shared research environment.
- Forty-seven new tests cover real/Fourier scalar likelihood and prior equality
  against the preserved code, fixed/ranged periods, TESS/Kepler coefficients,
  both host-mass radius-prior branches, flat priors, scalar/per-point errors,
  even/odd Fourier lengths, result packing and its inherited fallback.
- Existing seeded real/Fourier TP evidence and returned posterior arrays remain
  exactly equal to the reference. Full weighted pool recording and save/load
  still pass. Scalar, prior-only, and fully batched sampler modes all execute
  the shared TP model, and fast exposure comparisons retain their tolerances.
- Fast TP dispatch is tested with closure inspection deliberately disabled;
  Fourier callbacks are explicitly rejected by the real-space fast adapter.
- Focused linting, whitespace checks, and the source audit pass. All 32
  preserved source files remain unchanged. The 3655 warnings are inherited
  scalar conversions and deprecated API calls in reference code, now exercised
  by more differential tests.

This is a structural regression check, not new scientific validation or a
production FPP rerun. Other planet-host families and the EB families still
need staged extraction and their own differential tests. See
[architecture](architecture.md#shared-scenario-components).

Retained artifacts (earlier wheels remain intact):

```text
dist/pentaceratops-0.1.0.dev4-py3-none-any.whl
SHA-256: 8339689e20ff8fe5d38ce86fac66f2c54865d99ca3fadb1fcc0616867bc2838e
docs/tp-refactor-wheel-tests.xml
```

## Composable scenario pilot: `0.1.0.dev5` (2026-09-30)

TP/STP and the ordinary explicit-secondary EB/SEB wrappers now compose a host
with a planet or binary system, in both real and Fourier space. Their public
arguments, prior support, likelihood normalization, timing options, and
posterior-output policies are preserved. Other hosts, the twin-split routines,
and x2P paths remain unchanged. See the
[component boundary](architecture.md#shared-scenario-components).

- **292 tests passed** against the unpacked wheel in 58.00 seconds using
  single-threaded numerical libraries in the isolated PyTransit 2.9.2 stack.
  The source suite passed all 283 then-existing tests; the nine subsequently
  added sampler-mode smoke tests also passed before wheel verification.
- The 51 new tests cover both composition axes, scalar prior/likelihood
  equality, fixed/ranged periods, TESS/Kepler coefficients, low-mass hosts,
  optional MOLUSC populations (including an empty filtered population),
  contrast curves, per-point errors, and odd/even Fourier lengths.
- Tiny seeded STP/EB/SEB evidence calculations reproduce the reference outputs
  exactly in both domains. Full samples, weights, log targets, and evidence
  survive saving/loading and match the preserved sampler pools. Separate
  fixed-pool tests check historical posterior resampling and packing.
- Scalar, batched-prior-only, and fully batched sampler modes execute the
  composed scenarios and restore their hooks. Fast covariance likelihoods
  keep the existing scalar-agreement tolerance; closure inspection is disabled
  deliberately when testing migrated dispatch.
- Compatibility choices are explicit: real and Fourier companion LDC lookup
  rules are not unified numerically, nor are target/host inputs to size priors
  or the combined-light contrast weighting for SEB.
- Wheel import isolation, bundled tables, linting, formatting, and whitespace
  checks passed. All 32 preserved research files still match their extraction
  hashes. No production reruns, catalogue downloads, or shared-environment
  changes were made. The 9070 warnings are inherited scalar conversions and
  deprecated API calls, mostly from the preserved reference comparisons.

The final README and one adapter docstring were updated after the full wheel
suite; the final wheel is rebuilt and smoke-checked with no numerical changes.
These tests establish structural compatibility on the selected cases, not
production evidence precision or complete benchmark equivalence.

Retained artifacts (earlier wheels remain intact):

```text
dist/pentaceratops-0.1.0.dev5-py3-none-any.whl
SHA-256: ce40372b0fed6085186f96487c49ddd6ca06cf2f6ccfe52f2b601e30cb3d0685
docs/scenario-components-wheel-tests.xml
```

## All scenario recipes: `0.1.0.dev6` (2026-09-30)

All 52 public evidence wrappers now use shared host, planet/binary, and orbit
components. This includes T/P/S/D/B, explicit-secondary and x2P recipes,
legacy primary-only single/twin branches, and specialized unknown/evolved N
entry points in both domains. Known nearby stars continue to reuse target
recipes. Recipe-specific preparation and posterior packing remain in the
wrappers; their public signatures and output conventions are unchanged.

- **529 tests passed** against the unpacked standalone wheel in 77.03 seconds
  and in the source suite (61.25 seconds), with one numerical-library thread.
  The isolated PyTransit 2.9.2 environment and preserved reference were used.
- The 237 added tests include 156 recipe comparisons: every wrapper with
  default, catalogue/contrast-curve, and low-mass configurations. Scalar priors,
  likelihood values, sampler settings, and physical posterior outputs match
  exactly. Batched transforms/physical columns retain their stated tolerances.
- Every wrapper also has a tiny seeded sampling/save-load comparison. Evidence,
  complete samples, weights, and log targets match the preserved engine
  exactly, including both pools from legacy two-branch calls.
- Scalar/fast covariance comparisons cover all 15 standard T/P/S/D/B
  planet/secondary/x2P combinations, plus unknown/evolved-neighbor planet
  recipes, with closure inspection disabled. Invalid batched particles are
  rejected without indexing invalid catalogue rows.
- Nearby-star tests cover empty magnitude-selected populations and hosts
  failing the logg/temperature cut. The latter remain in the prior population
  rather than being removed and renormalized.
- Compatibility details remain explicit: eccentricity is drawn before period
  doubling; x2P collision policies differ by family; only legacy twins use the
  q=.95 split; B's distance correction and target-mass radius prior remain;
  specialized NEB retains its 1-Msun mass-ratio prior. Existing Fourier timing
  constraints and the legacy scalar `sigma_veto` are forwarded unchanged.
- Import isolation, bundled tables, linting, formatting, whitespace, and
  source audit pass. All 32 preserved research source files are unchanged.
  No production candidate runs, catalogue queries, or dependency changes were
  made. The 56,718 warnings are inherited scalar conversions and deprecated
  calls exercised heavily in the preserved reference comparisons.

These are structural compatibility tests, not production evidence-convergence
or population-level FPP validation. Direct Fourier parity-swap calls still
require matching grids; legacy primary-only EB requires a scalar veto error.
The fast backend remains experimental, real-space, fixed-period, and excludes
legacy primary-only binaries. No sampling defaults or demographic odds changed.

Retained artifacts (earlier versions remain intact):

```text
dist/pentaceratops-0.1.0.dev6-py3-none-any.whl
SHA-256: d4366ddb63334e922a847579f16709745b52ccb66bc414035979eb3addc64b4c
docs/all-scenarios-wheel-tests.xml
```

## Folded HZ inference: `0.1.0.dev7` (2026-10-05)

Prepared-input interfaces now integrate optimized folded Real and full-orbit
folded Fourier inference. Historical Kepler full-P/2P and covariance-only
updates remain explicit, preserving their original grids, scalar sampler,
priors, and timing limits. See [folded HZ runs](hz_runs.md).

- **612 tests passed**, none skipped, against the standalone wheel in 63.61
  seconds. All preserved-engine comparisons ran. Wheel imports and bundled
  tables passed without installing packages or importing the research engine
  at runtime. The same isolated PyTransit 2.9.2 stack was used.
- Tests cover covariance propagation through gaps and duplicate fold bins,
  native exposure weights, mission-specific coefficients, all physical
  scenario families, nearby-host aperture frames, primary-only inputs,
  historical grids, actual tiny sampling calls, saved pools and scalar replay.
- All **172 saved Real best-fit photometric gains** across the seven retained
  TESS targets replay exactly. All seven Real FPPs reconstruct from the saved
  evidences to the stated `2e-12` absolute tolerance.
- All **21 TOI-7390.01 folded Fourier** best-fit models and photometric gains
  replay exactly. Its reconstructed FPP is `0.11453120078512388` with eta=0.1.
- All **18 KOI-2719.02 seven-transit Fourier** best-fit models and log targets
  replay exactly, retaining original priors and EB timing restrictions. Its
  reconstructed FPP is `0.02963812370424862` with eta=0.1.
- The inherited reference test's dispatcher inventory now explicitly allows
  the additional observed-data entry point; every original function body
  remains checked. Existing numerical tolerances were not relaxed.
- The 62,979 warnings are inherited scalar-conversion and deprecated-API
  warnings from the expanded reference/scenario coverage. They are not failed
  numerical comparisons.

The [machine-readable validation](hz-validation-20261005.json) identifies the
wheel and runtime source hashes; the [production replay](hz-replay-20261005.json)
retains individual comparisons. The wheel and pytest XML are retained under
ignored `dist/`. No paper data or frozen research inputs were modified.

These checks establish prepared-input integration, regression compatibility,
and saved-production replay. They do not constitute new production sampling,
evidence-convergence certification, a catalogue-to-FPP pipeline, thread-safe
inference, or a fresh-environment dependency matrix. The previous licensing
and attribution release gates also remain.

```text
dist/pentaceratops-0.1.0.dev7-py3-none-any.whl
SHA-256: ad8b0155f36e8019af88157441cc5dfed1a1b7239d23b0cff50f79297cbc361f
dist/hz-dev7-wheel-tests.xml
```

## Automatic eclipse timing: `0.1.0.dev8` (2026-10-05)

New recorded runs derive local x2P support from eclipse contacts and supplied
window/exposure boundaries. Full-period Fourier models include both eclipses
wherever they fall on either half grid, without a fixed duration-based cutoff.
Explicit legacy settings preserve archived model placement and timing limits.
The existing ordinary-EB synthetic-secondary Gaussian penalty in folded Real
is unchanged under both timing settings; no new 1.5-sigma veto was added.
See [timing and observation coverage](timing.md).

- **632 tests passed**, none skipped, against the standalone wheel in 97.54
  seconds. This includes all preserved-engine comparisons and 20 timing tests.
- Timing tests cover x2P edge overlap, asymmetric windows, exposure boundaries,
  periodic wrapping, gaps, scalar/optimized equality, preserved ordinary-EB
  secondary constraints, and independent full-orbit model comparisons.
- The ordinary-EB scalar methods also match their pre-change syntax trees.
- Actual tiny Fourier sampling checks retained pools and timing-policy
  recording. These are software tests, not production evidence calculations.
- In explicit legacy mode, all **172 Real**, **21 TOI-7390.01 Fourier**, and
  **18 KOI-2719.02 seven-transit Fourier** saved fits replay exactly. Stored
  evidences reconstruct their FPPs within `2e-12` absolute tolerance.
- The 63,239 warnings are inherited scalar-conversion/deprecated-API warnings.
  No numerical comparison tolerances were relaxed.

The [validation record](hz-timing-validation-20261005.json) and
[individual replay results](hz-timing-replay-20261005.json) identify the final
runtime source and wheel. The wheel's 63 Python source files match the checked
source exactly. No production sampler, paper table, or frozen input was updated;
FPPs under the new timing policy require new inference.

```text
dist/pentaceratops-0.1.0.dev8-py3-none-any.whl
SHA-256: df9ec7bfdee4fcd36677becef222382d7fc4476ddec7cc605bef5b40d85eda9c
dist/hz-dev8-wheel-tests.xml
```
