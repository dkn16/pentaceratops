# Run evidence from prepared files

For a direct Python call without JSON, load the output of the preprocessing
script using the exact NPZ path it prints after `Prepared data:`:

```python
from pentaceratops import RunResult, load_target
from pentaceratops.evidence import evidence

prepared = RunResult.load("/path/to/cache/prepared/<candidate>_<hash>.npz")
target = load_target("/path/to/new_field")
results = evidence(prepared, target=target, likelihood="fourier", eb_eta=0.1)
```

`prepare_target` creates a new TESS field; `load_target` reloads its saved
files. See [target setup](target_setup.md) for both routes and for loading
adopted fields from existing HZ runs.

See the [Python interface](evidence_api.md) to create `prepared` directly with
`prepare_candidate` instead. The configuration-based CLI below is optional.

After preprocessing and supplying the stellar field/population inputs, run:

```bash
pentaceratops evidence target.json
```

The equivalent command without the installed executable is
`python -m pentaceratops evidence target.json`. From a source checkout without
an editable installation, prefix it with `PYTHONPATH=src`. The Python interface
is `pentaceratops.run_prepared("target.json")`, returning a `RunResult`.

Copy [examples/evidence.json](../examples/evidence.json) to `target.json` and
replace its example ID and file paths with your actual inputs. All paths inside
the JSON are relative to that JSON file, so it can be launched from any directory.
The command performs no catalog query, download, PSD estimation or detrending fit.

For a saved output from `examples/tess_candidate.py` or `kepler_candidate.py`:

```json
{
  "schema_version": 1,
  "mission": "TESS",
  "target_id": 123,
  "stars": "stars.csv",
  "trilegal": "trilegal.csv",
  "input": {"format": "candidate", "path": "prepared/candidate.npz"},
  "likelihood": "real",
  "settings": {"N": 500, "steps": 50, "seed": 42, "eb_eta": 0.1},
  "output": "results/real.npz"
}
```

`123` is a placeholder. For candidate bundles, the configured mission and
target ID must match the saved ephemeris (TIC for TESS, KIC for Kepler).
No candidate identifier is guessed from a filename.

The same candidate preparation can also supply folded full-orbit Fourier:

```bash
pentaceratops evidence target.json --likelihood fourier --output results/fourier.npz
```

An explicit `--output` is relative to the working directory. Outputs must be
new `.npz` files; existing results are rejected before covariance building or
sampling. The command prints the output path, scenario count, FPP, EB FPP and
nearby-source FPP. A scenario subset is explicitly labeled conditional.

## Stellar field and populations

The light-curve preprocessing scripts do not construct these inputs. Supply
the already prepared stellar table as CSV, target first, then nearby sources:

```text
ID,mass,rad,Teff,plx,Tmag,Jmag,Hmag,Kmag,tdepth,fluxratio
```

Use solar units for mass/radius, kelvin for temperature and milliarcseconds for
parallax. `fluxratio` is each host's fraction of aperture light, in `(0,1]`.
`tdepth` is the host-eligibility depth from the field/dilution calculation;
positive values identify eligible hosts. Do not set every row positive by hand
or omit nearby sources to obtain a target-only FPP. The likelihood evaluates
all eligible hosts in the same aperture flux frame.

J/H/K magnitudes and parallax must be finite for the target. They may be
missing (`NaN`) for resolved neighbors: NTP/NEB/NEBx2P do not use them.
Eligible neighbors still require mass, radius, temperature, TESS magnitude
and aperture flux fraction. Missing infrared photometry never excludes a host.

An existing field object can be exported with
`target.stars.to_csv("stars.csv", index=False)`. Supply the same TRILEGAL file
used for that field. An optional top-level `"molusc": "molusc.csv"` selects the
prepared target companion population. No population or stellar parameters are
silently fabricated. Missing stellar parameters fail by default; the existing
explicit `missing_host_policy="solar"` only permits documented neighbor
substitutions, never target substitutions.

## What happens to the saved photometry

For `candidate` inputs, the command finishes covariance preparation using the
state saved by the example scripts:

- **Real:** reconstruct the fixed-PSD conditional coefficient posterior from
  the saved final fit input and training mask; apply the original weighted
  folding operator to its mean and covariance. Preserve covariance between
  the even, odd and secondary panels. Retain the HZ convention of one median
  measurement error per panel, separately from the trend covariance.
- **Fourier:** use every saved raw observed sample, normalized by its saved
  sector baseline. Fold at 2P, retaining alternating eclipses and absent bins,
  and propagate each sector's saved PSD through the same folding operator.
  The PSD white floor enters once; the finite DC prior uses the lowest-frequency
  total power. No native dense covariance or inverse is needed.

Both use the existing optimized HZ evidence routines. Normalization, masks,
ephemerides and the Real folding operator are checked against the preparation.
These are new inferences using those supplied data and PSDs; they do not update
or automatically reproduce previously saved paper results.

Covariance preparation and evidence can be expensive. Run real candidates in
a compute allocation. The command processes scenarios sequentially; use
separate processes for different targets and one numerical-library thread per
worker. The Fourier folded covariance is dense, even though its construction uses
FFT products without a native covariance matrix.

## Existing likelihood-ready HZ inputs

To skip covariance preparation, change only the input description:

```json
"input": {"format": "folded_real", "prepared": "prepared.npz", "posterior": "folded_posterior.npz"}
```

This uses `load_folded_real`. Explicit primary-only preparations also require
`"primary_only": true` in `settings`. To reproduce the saved Real timing rule,
set `"timing_policy": "legacy"` there. The ordinary-EB secondary penalty is
preserved under both settings.

For a directory saved by `FoldedFourierData.save`:

```json
"input": {"format": "folded_fourier", "directory": "folded_fourier"}
```

Set `"likelihood": "fourier"` as well. This interface uses the full-orbit
folding operator. Original half-grid Fourier archive recipes remain available
through `run_folded_baseline` or `calc_probs_fourier(..., timing_policy="legacy", weighting="legacy")`;
the command does not reinterpret those archives as full-orbit inputs.

## Settings and records

CLI overrides include `--N`, `--steps`, `--nsamples`, `--seed`,
`--posterior-samples`, and `--eb-eta`. They override the JSON settings. Absent
`N`/`steps` settings (or JSON `null`) use the [shared scenario policy](evidence_api.md#arguments).
An explicit integer overrides that field for every scenario independently;
`"N": 500, "steps": 50` reproduces the previous uniform effort settings.
Other defaults are seed 42, 2000 posterior draws, profile parity and eta=1;
exposure integration uses 20 subsamples for Real or 7 for Fourier.
The example explicitly chooses the paper's eta=0.1.
Particle counts are effort settings, not evidence-convergence guarantees.

The JSON also accepts `scenarios`, `parity`, `filt`, `missing_host_policy`, and,
for Real, `primary_only`, `include_gp` and `timing_policy`. Unknown or incompatible
settings fail explicitly. Omit `scenarios` to run all eligible scenario families.

Each result retains the full weighted sampling pools, best-fit replay records,
resolved settings, source versions, original JSON and hashes of all input files.
The large covariance is reproducible from those inputs rather than duplicated
in the evidence result. Keep the preparation files alongside the result.
