# Installation

Pentaceratops is installed from its repository. It needs Python 3.10 or later.

```bash
git clone https://github.com/dkn16/pentaceratops.git
cd pentaceratops
python -m pip install -e '.[catalogs]'
```

Use a fresh virtual or conda environment. Several dependencies are pinned to
narrow ranges (see below), so installing into an environment that other
projects rely on can change their versions.

## Optional extras

| Extra | Adds | Needed for |
| --- | --- | --- |
| `catalogs` | `astroquery`, `lightkurve` | Downloading light curves and querying catalogues |
| `dev` | `pytest`, `build`, `ruff` | Running the tests |
| `docs` | MkDocs and its plugins | Building this site |

Without `catalogs` the package can still run the evidence calculation on
inputs that are already prepared.

Combine extras with a comma, for example `'.[dev,catalogs]'`.

## Check the installation

```bash
python -m pentaceratops doctor
```

This prints the installed version of every dependency and exits with a non-zero
status if a required one is missing. It makes no catalogue queries or
downloads; importing the package never does either.

```bash
python -m pentaceratops sampling-policy
```

This prints the default sampling effort for each scenario. See
[sampling and convergence](concepts/sampling.md).

## Dependency ranges

| Package | Range | Reason |
| --- | --- | --- |
| NumPy | `>=2.0,<2.3` | Inherited code still relies on scalar conversions that newer NumPy deprecates |
| PyTransit | `>=2.9.2,<2.11` | The fast model path uses PyTransit internals that must match the reference model; 2.9.2 and 2.10.1 are tested |
| MeepMeep | `>=1.1.0,<1.3` | Orbit routines shared with PyTransit; 1.1.0 and 1.2.1 are tested |

## Storage and runtime settings

Light curves, catalogue responses and prepared inputs are cached on disk and
can be large. Point the cache at scratch storage:

```bash
export PENTACERATOPS_CACHE=/path/to/scratch/pentaceratops
```

The download and preparation helpers use, in order: an explicit `cache_dir` or
`--cache-dir`, `PENTACERATOPS_CACHE`, a subdirectory of `PSCRATCH` or `SCRATCH`,
and finally the system temporary directory.

The evidence calculation is single-threaded by design. When running many
targets in parallel, use one process per target and keep numerical libraries at
one thread each:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
```

Do not run several targets as threads inside one process: the model caches are
not thread-safe.

## Running the tests

```bash
python -m pip install -e '.[dev]'
python -m pytest tests -q -p no:cacheprovider
```

The tests use synthetic data and need no network access. See the
[developer workflow](development.md) for the comparison tests against the
original research code.
