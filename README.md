# Pentaceratops

Statistical validation of transiting planet candidates using competing planet
and eclipsing-binary scenarios, real-space or Fourier-space likelihoods, and
persistent sampling.

**Development version: `0.1.0.dev3`.** The standalone numerical engines,
posterior recording, preprocessing utilities, and experimental CPU-batching
adapters are implemented. The corrected end-to-end research pipelines are not
fully integrated yet: a bare `Target.calc_probs(...)` call is **not** a
replacement for the production benchmark runners. In particular, consistent
flux frames, null-evidence corrections, and covariance setup remain the
runner's responsibility. See [architecture and limitations](docs/architecture.md).

## Installation

From this directory, in a separate development environment:

```bash
python -m pip install -e '.[dev]'
python -m pentaceratops doctor
python -m pentaceratops sampling-policy
```

Do not upgrade the frozen research environment in place. For catalog-backed
target construction, install `'.[dev,catalogs]'` instead. Package imports and
`doctor` perform no catalog queries or downloads; constructing a `Target` can
access services and download data. Set its `lightkurve_cache_dir` explicitly
to a scratch location.

The package declares Python >=3.10; the tested environment uses Python 3.13.
The package now targets PyTransit **2.9.2** and MeepMeep **1.1.x**, with matching
scalar and fast orbital calculations. The old runtime setuptools restriction
is removed. NumPy is temporarily limited to `>=2.0,<2.3` while inherited scalar
conversion deprecations remain. Use a separate development environment, not an
in-place upgrade of a frozen research run. See the [upgrade notes](docs/pytransit.md)
and [development](docs/development.md) for tests and compatibility limits.

## Download and preprocess a candidate

Two editable Python examples now resolve an identifier, download official
PDCSAP light curves, restore the aperture flux frame, mask other catalogued
planets, detrend each sector/quarter, and plot the folded input windows:

```bash
# Install the catalogs extra first; run from this checkout.
export PENTACERATOPS_CACHE="$PSCRATCH/pentaceratops/candidate_examples"
python examples/tess_candidate.py --target TOI-4616.01
python examples/kepler_candidate.py --target 'KIC 8758204'
```

Alternatively, change `TARGET` near the top of either script. The default
ephemeris source is **NASA Exoplanet Archive** (TOI and KOI tables). For TESS,
`--catalog-source bayesian` explicitly selects Bayesian Exoplanets. Sources
are not silently mixed. Period, epoch, duration, source records, and overrides
are saved; ambiguous TIC/KIC hosts require a `--candidate` TOI/KOI selection.

Use `--stage resolve` to check ephemerides, `--stage download` to prefetch,
or the default `--stage prepare` for detrending and plots. Bulk caches go to
scratch; `--plot-dir` sets a separate plot location. Real-data FGP processing
and target sweeps should use a compute job; an example Slurm script is provided.

This is **preparation only**, not an automatic FPP run: stellar-field setup,
corrected scenario orchestration, FGP covariance propagation, and full-period
Fourier-likelihood inputs are not yet integrated. Plotted errors currently
include measurement errors only. See [examples and all options](examples/README.md).

## Run and save an evidence calculation

`run_evidence` accepts an evidence function and its normal numerical arguments.
It records weighted sampler pools without changing the sampler's settings or
taking extra random draws. This example assumes an already prepared,
normalized light curve for an isolated target, with no additional aperture
dilution; the input variables must be supplied by the caller:

```python
from pentaceratops import run_evidence, RunResult
from pentaceratops.evidence.real import lnZ_TTP
from pentaceratops.sampling.policy import sampling_config

# Optional N/steps overrides are independent: sampling_config("TP", N=200).
config = sampling_config("TP")
result = run_evidence(
    lnZ_TTP,
    time=time_days, flux=normalized_flux, sigma=flux_error,
    P_orb=period_days, M_s=mass_solar, R_s=radius_solar,
    Teff=temperature_kelvin, Z=metallicity_dex,
    N=config["N"], steps=config["steps"], mission="TESS",
    exptime=exposure_days, nsamples=20,
    seed=123,
    output_path="/path/to/scratch/target_tp.npz",
)
print(result.output["lnZ"])
saved = RunResult.load("/path/to/scratch/target_tp.npz")
```

A `RunResult` has three main fields:

- `output`: scenario evidence and physical parameter outputs, or a scenario
  table for the compatibility dispatchers.
- `sampling`: evidence-call inputs and full weighted sampler pools, including
  per-particle log-target values.
- `metadata`: versions, seed, callable identity, and a source fingerprint.

Posterior pools are retained by default. Saving produces **one compressed,
pickle-free NPZ file**, with no overwrite of an existing result. Omit
`output_path` to keep the result in memory, or save later with
`result.save(path)`. Automatic failed-run checkpointing is not implemented.
Use the weighted pools for posterior analysis; some legacy physical-output
arrays deliberately place a best-fit sample in the first row.

Inspect a saved result without fitting again:

```bash
python -m pentaceratops inspect /path/to/scratch/target_tp.npz
```

The recorded compatibility interfaces `Target.calc_probs(...)` and
`calc_probs_fourier(...)` also return `RunResult`, with a scenario table in
`.output`. The real-space compatibility interface warns about the missing
benchmark orchestration. Details: [results and posterior records](docs/results.md).

## From evidence to false-positive probability

A single-scenario evidence is not an FPP. Given a table of comparable,
**unweighted** scenario log evidences with the required frame/null corrections:

```python
from pentaceratops import scenario_probabilities

probabilities = scenario_probabilities(
    scenario_table, evidence_column="lnBF", eb_eta=0.1,
)
print(probabilities.attrs["FPP"])
print(probabilities.attrs["FPP_EB"])
```

Standard FPP counts all scenarios except TP, PTP, and DTP as false positives;
STP, BTP, and NTP therefore contribute. `FPP_EB` counts only eclipsing-binary
scenarios, while `NFPP` counts scenarios on nearby stars.

`eb_eta` multiplies EB scenario weights once, before normalization. It is an
explicit demographic-odds choice, not an empirically calibrated probability.
Do not apply it to an already-weighted evidence column. This helper neither
repairs inconsistent likelihood frames nor substitutes raw sampler evidence
for corrected scenario evidence.

## Fast CPU backend and exposure integration

The fast backend is **experimental and opt-in**, not the default. It batches
prior transformations and physical/photometric calculations for the
matched-window real-space workflow, with or without FGP covariance. It is
not a different scientific likelihood.

| Fast model path | Standard scenario labels |
| --- | --- |
| Planet | TP, PTP, STP, DTP, BTP, NTP |
| EB with secondary window | EB, PEB, SEB, DEB, BEB, NEB |
| Even/odd EB at twice the candidate period | EBx2P, PEBx2P, SEBx2P, DEBx2P, BEBx2P, NEBx2P |

Nearby-source scenarios reuse the target-star kernels with the appropriate
host inputs. This is kernel support, not a promise that the standalone
catalog runner already reproduces every corrected research result. A fast
Fourier backend, legacy primary-only EB likelihood, and specialized
unknown/evolved-host routines are not included in this adapter.

The fast adapter requires a **fixed P**, or **2P** for even/odd scenarios.
Exposure integration is configurable:

```python
from pentaceratops.experimental.v2_adapter import OptimizedAdapter

# data is an already prepared matched-window data dictionary.
adapter = OptimizedAdapter(data, nsamples=10)
```

`nsamples` is the positive integer number of subsamples within each exposure,
default **20**. It is neither the sampler particle count `N` nor the exposure
duration `data["exptime"]` (days). One duration is used per adapter; mixed
per-point durations are not supported. A smaller count needs a numerical
convergence check before production use.

Scalar and batched primary, secondary, and even/odd models use the same
resolved count. Use `scenario_kwargs` to pass the adapter default or an
explicit override to the evidence function; direct low-level evidence calls
otherwise retain their own default of 20. Constructing an adapter alone does
not activate batching: the experimental engine/sampler contexts are also
required. See [exposure integration](docs/exposure.md) for the configuration
examples and precedence rules.

## Planned one-line workflow

The next integration milestone is one call each to `preprocess`, `validate`,
and `report`, with optional advanced arguments at every stage. **These
high-level functions and corresponding runner/plotting commands are not
implemented yet.** The current CLI provides `doctor`, `sampling-policy`,
and `inspect`.

The [interface contract](docs/user_api.md) specifies explicit argument/CLI
overrides over configuration files over versioned defaults, independent
likelihood/backend selection, saved resolved settings, and plotting from
stored results without resampling. The download/preprocessing examples above
are available under `pentaceratops.preprocessing`; they stop before the
validation stage and do not replace the planned three-stage interface.

## Running and storing experiments

- Use one process per target, not concurrent threads: legacy model caches,
  module globals, and experimental hooks are not thread-safe.
- Keep numerical-library threads at one when packing parallel workers:
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1`.
- Submit long sampling and broad target sweeps through Slurm on NERSC;
  small synthetic unit tests can run locally.
- Keep light curves, catalog caches, prepared arrays, and posterior bundles
  on configurable scratch paths. Set `MPLCONFIGDIR` and `NUMBA_CACHE_DIR`
  there as well. Small plots, summaries, and source code can remain in home.
- Sampling effort defaults are provisional, not guarantees of evidence
  precision. Fourier dispatcher defaults are not silently replaced by the
  real-space per-scenario policy.

## Code layout

```text
src/pentaceratops/
  api.py, results.py       Recorded interfaces and portable result bundles
  likelihoods/            Transit models and real/Fourier residual costs
  evidence/               Scenario-specific priors and evidence integrals
  sampling/               Persistent sampler and provisional effort policy
  preprocessing/          Aperture-frame restoration and sector FGP utilities
  companions/, data/      Stellar-companion machinery and small model tables
  experimental/           Correlated likelihood and CPU-batching adapters
tests/                    Unit and opt-in research-reference comparisons
examples/                 TESS/Kepler download and preparation scripts
```

No experiment outputs, light curves, cached catalogs, or scratch posterior
archives are bundled. The `research/` symlink is only a local convenience and
is ignored by Git and packaging.

## Tests and provenance

The `0.1.0.dev3` wheel passed **194 tests** on 2026-09-29 with PyTransit 2.9.2,
including preserved-engine comparisons, result save/load, scalar/fast exposure
regressions, and download/preparation tests. A separate 240-template
old/new-dependency comparison is recorded in the [upgrade notes](docs/pytransit.md).
Earlier live NASA lookups and one-product downloads for both missions, plus
an optional Bayesian TESS lookup, also succeeded.
This is not yet full corrected benchmark equivalence or a multi-version CI
matrix. See the [validation record](docs/validation.md).

```bash
python -m pytest tests -q -p no:cacheprovider
python -m pytest tests -c pyproject.toml -q -p no:cacheprovider \
  --reference-root /path/to/preserved/triceratops
```

Without `--reference-root`, the preserved-engine comparisons are skipped.
Use `PYTHONPATH=src` if running from source without an editable installation.

The extraction used the working research checkout, including its uncommitted
fixes, not an older Git snapshot. Original hashes and Git state are in
[`docs/source_manifest.json`](docs/source_manifest.json). The import tool is
one-time migration tooling and refuses to overwrite imported files.

The preserved engine, original Triceratops snapshot, experiments, and paper
remain in `../pentaceratops_research`; they are not modified by this package.
`LICENSE` preserves the upstream MIT notice; experimental exposure kernels
also carry GPL-3.0-or-later notices. Public-release licensing and bundled-table
attribution review remain pending; see `NOTICE` and file-level notices.
