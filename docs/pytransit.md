# PyTransit 2.9 upgrade

The standalone package now targets `pytransit>=2.9.2,<2.10` and
`meepmeep>=1.1.0,<1.2`. These are bounded because the experimental fast
backend uses internal routines that must match the scalar model. This is a
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
