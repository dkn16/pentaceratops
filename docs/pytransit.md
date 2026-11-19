# PyTransit 2.9 upgrade

The standalone package now targets `pytransit>=2.9.2,<2.11` and
`meepmeep>=1.1.0,<1.3`. These are bounded because the experimental fast
backend uses internal routines that must match the scalar model. PyTransit
2.10.1 with MeepMeep 1.2.1 was checked on 2026-11-18; see
[PyTransit 2.10](#pytransit-210). This is a
development upgrade, not a rerun or replacement of published results.

Upstream: [PyTransit 2.9.2](https://pypi.org/project/PyTransit/2.9.2/),
[PyTransit source](https://github.com/hpparvi/PyTransit).

## Why more than a dependency bump was needed

PyTransit 2.9 uses MeepMeep's analytic orbital expansion and separate ingress
and egress contact bounds. Our previous sparse kernel used the older
finite-difference coefficients and a symmetric duration cut. Merely changing
the dependency caused an EB scalar/fast log-likelihood discrepancy of about
0.0114 in the first failing exposure fixture. The comparison tolerance was
**not** loosened to accept it.

The sparse kernel now uses exactly the orbital functions imported by
PyTransit's quadratic model: `solve2d`, `sep_c`, and `bounding_box`. A symmetric
envelope only accelerates index lookup; the final per-point test retains the
actual asymmetric bounds. Exposure integration, dilution, covariance, parity,
priors, and sampling effort are unchanged.

Calls to the deprecated `evaluate_ps` are replaced by the public `evaluate`
API. A test confirms identical scalar outputs for those two methods under the
new dependency. The compatibility engine's historical scientific conventions
are not otherwise rewritten.

PyTransit now uses `importlib.resources`, so the old **runtime** requirement
`setuptools<81` has been removed. Setuptools remains a build dependency.
PyTransit requires NumPy 2; the package temporarily uses `>=2.0,<2.3`, retaining
the tested numerical stack while inherited size-one-array conversion warnings
are still unresolved. Other dependency combinations need separate validation.

## Cross-version scientific check

The usual migration tests load the old and new *engine code* under the same
installed dependency. They cannot by themselves detect changes in PyTransit.
`tools/check_pytransit_upgrade.py` therefore saves snapshots in separate
environments and compares them afterward:

```bash
OLD_PYTHON tools/check_pytransit_upgrade.py snapshot /scratch/before.npz
NEW_PYTHON tools/check_pytransit_upgrade.py snapshot /scratch/after.npz
NEW_PYTHON tools/check_pytransit_upgrade.py compare /scratch/before.npz /scratch/after.npz
```

Use `PYTHONPATH=src` for an uninstalled checkout. `OLD_PYTHON`/`NEW_PYTHON` are
placeholders for interpreter paths. Snapshot files are no-clobber `RunResult`
bundles with physical inputs and dependency versions. All numerical-library
threads should be set to one. This small diagnostic is not a production sweep.

On 2026-09-29, 240 model-array comparisons covered short/long periods,
circular/eccentric orbits, grazing geometries, near-equal stellar radii,
dilution/host assignments, 2/30-minute exposures, and 1/5/20/50 subsamples:

| Model output | Largest absolute old-to-new change [ppm] |
| --- | ---: |
| Planet transit | 0.00635 |
| EB primary | 0.36581 |
| EB secondary | 0.17961 |
| x2P even | 0.001304 |
| x2P odd | 0.000000220 |

Two tiny, low-SNR TP evidence checks (`N=8`, `steps=2`, seed 73) changed log Z
by `1.92e-9` in real space and `1.75e-9` in Fourier space. These are smoke
checks, **not** high-precision convergence tests or proof that every target's
FPP is unchanged. No all-scenario production population was rerun.

The upgrade environment uses Python 3.13, PyTransit 2.9.2, MeepMeep 1.1.0,
NumPy 2.2.6, Numba 0.61.2, and setuptools 81.0.0. It is an isolated virtualenv
overlay inheriting the other packages from the research environment, not a
fresh dependency-resolution matrix. `pip check` passes. The original
`exoprob` environment remains on PyTransit 2.6.14/MeepMeep 0.7.2.

Result metadata now records PyTransit, MeepMeep, and Numba versions alongside
the existing Python/NumPy/SciPy versions and source fingerprint. Keep the old
environment when reproducing old results; there is no guarantee of bitwise
equality across the dependency upgrade. Batching remains opt-in.

See [validation](validation.md) for the complete suite and wheel checks.

## PyTransit 2.10

PyTransit 2.10.1 and MeepMeep 1.2.1 were tested on 2026-11-18 and the upper
bounds were raised to admit them. The four internal routines the fast path
imports (`eval_quad_z_s`, `solve2d`, `sep_c`, `bounding_box`) are still present;
`solve2d` gained an optional trailing argument that the package does not use.

- **Tests:** all 469 tests pass with PyTransit 2.10.1, MeepMeep 1.2.1 and
  Numba 0.68.1, including every scalar-against-fast comparison.
- **Model outputs:** the 240-fixture comparison of
  `tools/check_pytransit_upgrade.py` between 2.9.2 and 2.10.1 changes no model
  by more than `2e-9` ppm, and the two evidence checks by `9e-16` and `0`.
- **Real targets:** the complete `evidence()` call was run for TOI-6612.01 and
  TOI-700.02 with both likelihoods, default sampling and seed 42, on one core
  each of one node, twice per environment.

| Environment | PyTransit | MeepMeep | Numba | Total time, relative |
| --- | --- | --- | --- | ---: |
| Matched, old | 2.9.2 | 1.1.0 | 0.63.1 | 1.000 |
| Matched, new | 2.10.1 | 1.2.1 | 0.63.1 | 0.999 |
| Newest Numba | 2.10.1 | 1.2.1 | 0.68.1 | 1.020 |

The matched pair differs only in PyTransit and MeepMeep. Its log Bayes factors
are identical in every scenario, and its run times agree to 0.1 per cent,
within the 1 to 3 per cent scatter between repeated passes. With Numba 0.68.1
the log Bayes factors differ by at most `6e-7` and the run is about 2 per cent
slower; that difference comes from Numba. MeepMeep 1.1.0 cannot be installed
together with Numba 0.68.1, so PyTransit 2.9.2 was not tested with it.

This is one machine and two targets, not a dependency matrix. Other
combinations still need their own check.
