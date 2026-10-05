# Developer workflow

## Software versus experiments

Develop in `pentaceratops/`. Runners, frozen engines, outputs, and manuscript
remain under `pentaceratops_research/`. Do not rerun the import tool to update
sources: make reviewed changes with tests. Do not redirect production imports
before complete benchmark comparisons pass.

Runtime code imports `pentaceratops`, never the old `triceratops` package.
Reference tests load the latter only with an explicit reference checkout.
Runtime code has no hard-coded home or scratch paths.

## Local tests

Activate a suitable environment, or use its interpreter explicitly:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/path/to/scratch/pentaceratops/cache/matplotlib
export NUMBA_CACHE_DIR=/path/to/scratch/pentaceratops/cache/numba
python -m pytest tests -q -p no:cacheprovider
```

Without an editable installation, add `PYTHONPATH=src`. Paths above are
placeholders. Small fixtures go under the system temporary directory. No
catalog queries or production runs are needed. Reference comparisons:

```bash
python -m pytest tests -c pyproject.toml -q -p no:cacheprovider \
  --reference-root /path/to/preserved/triceratops
```

Explicit `tests -c pyproject.toml` prevents the reference path from becoming
pytest's root directory. Comparisons include numerical function bodies,
physical templates, a seeded toy sampler, and tiny low-SNR TP real/Fourier
evidence calculations. Tiny effort tests migration, not scientific precision.
All 52 evidence wrappers delegate to shared host, system, and orbit components.
Their public signatures remain fixed. `tests/test_all_scenarios.py` compares
every wrapper's priors, likelihoods, result packing, tiny seeded evidence, and
full saved pools against the preserved implementation. It includes x2P,
legacy twins, discrete background/neighbor populations, and empty neighbor
populations. `tests/test_target_planet.py` and `tests/test_scenario_components.py`
retain the more detailed pilot tests, including MOLUSC/contrast-curve inputs,
covariance hooks, and all three sampler modes. Numerical kernels, dispatcher,
and sampling-policy bodies still undergo the original syntax-tree comparison.
Long all-scenario or candidate regressions must use the research Slurm
submission conventions and scratch caches, not a login node.

`tests/test_exposure.py` compares scalar and fast matched-window models at
multiple exposure resolutions. It tests primary, secondary, and even/odd
models, direct overrides, and actual evidence-closure plumbing without running
a production sampler. See [exposure integration](exposure.md).

## Build and verify without modifying the research environment

In a dedicated development environment:

```bash
python -m build
python tools/verify_distribution.py dist/pentaceratops-0.1.0.dev10-py3-none-any.whl
```

If the optional build frontend is unavailable but setuptools is installed,
an offline check can invoke the backend directly:

```bash
python -c 'from setuptools.build_meta import build_wheel; print(build_wheel("/tmp/pentaceratops-dist"))'
python tools/verify_distribution.py /tmp/pentaceratops-dist/pentaceratops-0.1.0.dev10-py3-none-any.whl
```

Verification unpacks a trusted locally built wheel into a temporary directory,
imports from that copy, exercises bundled tables, and checks no research-engine
or catalog client was imported. It does not install or upgrade anything. This
tests import isolation, not dependency resolution in a fresh environment.

To exercise the full suite against the unpacked wheel, add
`--tests tests --reference-root /path/to/preserved/triceratops` to the verification
command. `--junitxml /path/to/report.xml` retains a machine-readable test report.
`python tools/audit_source.py` checks the original imported source files against
their extraction hashes without modifying them.

## Compatibility constraints

- Current tests use Python 3.13; the PyTransit upgrade is tested in a separate
  scratch virtual environment, leaving `exoprob` unchanged. Other advertised
  Python versions still need a CI matrix.
- PyTransit is constrained to `>=2.9.2,<2.10`, and MeepMeep to `>=1.1,<1.2`.
  Its quadratic kernel's orbit implementation is also used by our fast path;
  do not relax these bounds without the compatibility tests. See
  [PyTransit upgrade](pytransit.md). There is no runtime setuptools constraint.
- Inherited size-one-array-to-scalar conversions emit NumPy deprecation
  warnings. NumPy is temporarily bounded to `>=2.0,<2.3`; the upgrade check
  holds NumPy at 2.2.6 instead of simultaneously changing that numerical stack.
- The engine supports process parallelism, not concurrent threads.
- Real-space effort defaults are provisional, not guarantees of `std(log Z)`.
  Fourier dispatcher defaults retain their historical signature and are not
  silently replaced by the real-space policy.

## Review checklist

1. Does a change alter physical priors, likelihood frames, null references,
   parity, masks, or RNG sequences? Document and test it separately from refactoring.
2. Are full weights/log targets and physical context retained? Do not call an
   unnormalized log target a posterior density.
3. Are time/flux/error ordering and cross-window covariance preserved?
4. Does the wheel contain small required tables but no experiment products,
   caches, symlinks, or research imports?
5. Are scratch/cache/output paths explicit and portable?

Companion-table attribution/licensing and corrected benchmark integration
remain prerequisites for a public release.
