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
