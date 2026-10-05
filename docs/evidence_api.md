# Evidence from Python

Create `prepared` with the provided preprocessing function and load `target`
from the saved stellar field for the same candidate. The field contains the
host, nearby sources, aperture dilution, and a TRILEGAL background population.
If you do not have these files yet, follow [target setup](target_setup.md)
first; it shows catalog queries, dilution, population preparation, and saving.

```python
from types import SimpleNamespace
import pandas as pd

from pentaceratops.preprocessing.candidate import prepare_candidate
from pentaceratops.evidence import evidence

prepared, prepared_path = prepare_candidate("TOI-700.02", cache_dir="/path/to/cache")
# Load the stellar field saved for this candidate (see target setup below).
candidate = prepared.metadata["settings"]["candidate"]
target = SimpleNamespace(
    ID=int(candidate["host_id"]), mission=candidate["mission"],
    stars=pd.read_csv("/path/to/field/stars.csv"),
    trilegal_fname="/path/to/field/trilegal.csv",
)
results = evidence(prepared, target=target, likelihood="fourier", N=500, steps=50, eb_eta=0.1)
```

`prepare_candidate` resolves the ephemeris, downloads photometry when needed,
detrends and folds it, and saves a preparation bundle. It returns **two values**:
`prepared` is the in-memory `RunResult`; `prepared_path` is the saved `.npz`
file. Unpack the pair as shown rather than passing the pair to `evidence`.
Use your candidate identifier and cache directory in place of the example.
This preparation step can be expensive; run real targets in a compute allocation.

Use `likelihood="real"` for folded Real evidence. This is a direct Python
function: it accepts objects, returns a `RunResult`, and needs no JSON file or
CSV export. The explicit import keeps the existing `pentaceratops.evidence`
package and its low-level scenario modules available.

## Inputs

`prepared` can be any of these:

| Input | Likelihood |
| --- | --- |
| The `RunResult` returned by `prepare_candidate`, or its saved `.npz` path | Real or Fourier |
| The dictionary returned by `load_folded_real` | Real |
| A `FoldedFourierData` object, or its saved directory | Fourier |

## Reuse preprocessing that has already finished

If you used the provided scripts, for example:

```bash
python examples/tess_candidate.py --target TOI-700.02 --cache-dir /path/to/cache
```

copy the exact filename from the script's `Prepared data: ...` output. The
file is under `<cache_dir>/prepared/` and includes a settings hash; do not
substitute the native-download cache or guess the hash. The Kepler example
script prints the same kind of preparation path.

Load that saved bundle without downloading or detrending again:

```python
from types import SimpleNamespace
import pandas as pd

from pentaceratops import RunResult
from pentaceratops.evidence import evidence

prepared_path = "/path/to/cache/prepared/<candidate>_<hash>.npz"  # Copy the printed path.
prepared = RunResult.load(prepared_path)
# Load the stellar field saved for this candidate (see target setup below).
candidate = prepared.metadata["settings"]["candidate"]
target = SimpleNamespace(
    ID=int(candidate["host_id"]), mission=candidate["mission"],
    stars=pd.read_csv("/path/to/field/stars.csv"),
    trilegal_fname="/path/to/field/trilegal.csv",
)
results = evidence(prepared, target=target, likelihood="fourier", eb_eta=0.1)
```

You may also pass `prepared_path` directly as the first argument to `evidence`.
The same candidate preparation works with `likelihood="real"` or `"fourier"`;
the evidence function constructs the corresponding covariance from its saved
PSD and fit state. No preprocessing rerun is needed to switch likelihoods.

For existing likelihood-ready HZ files, reuse the matching `target` field
from above and create `prepared` with the matching loader below. These blocks
continue the preceding setup; the folded files do not contain a stellar field.

```python
from pentaceratops import load_folded_real
from pentaceratops.evidence import evidence

prepared = load_folded_real("/path/to/prepared.npz", "/path/to/folded_posterior.npz")
results = evidence(prepared, target=target, likelihood="real", eb_eta=0.1)
```

```python
from pentaceratops import FoldedFourierData
from pentaceratops.evidence import evidence

# Directory previously written by FoldedFourierData.save(...).
prepared = FoldedFourierData.load("/path/to/folded_fourier")
results = evidence(prepared, target=target, likelihood="fourier", eb_eta=0.1)
```

The paired Real files and folded Fourier directory are different formats from
the candidate-script bundle. See [folded HZ preparation](hz_runs.md) for how
those likelihood-ready products are constructed.

## Stellar-field input

The [target setup guide](target_setup.md) shows how to construct a new TESS
field and save it, and how to load the field from an existing HZ run.

`target` supplies `.mission`, `.stars` and `.trilegal_fname`; it can be an existing
prepared `Target` or an object with those attributes. `.stars` is the in-memory
stellar DataFrame with the target first, then nearby sources. Its required
columns are `ID, mass, rad, Teff, plx, Tmag, Jmag, Hmag, Kmag, tdepth, fluxratio`.
`tdepth > 0` selects eligible hosts, and `fluxratio` is each host's aperture
flux fraction. The call does not construct or download this field.

If the population path is not attached to the object, pass
`trilegal_fname="/path/to/trilegal.csv"`. Optional `molusc_file` supplies the
prepared companion population. Candidate bundles must match the target ID
and mission. Supplied objects are not modified.

For candidate bundles, the function constructs the conditional Real covariance
or full-orbit Fourier covariance from the saved PSD and fit state, without
rerunning detrending. Likelihood-ready dictionaries and folded objects bypass
that step. See [covariance handling](evidence_command.md#what-happens-to-the-saved-photometry).

## Results and saving

```python
table = results.output                          # ID, scenario, lnBF, lnZ, prob, ...
fpp = table.attrs["FPP"]
eb_fpp = table.attrs["FPP_EB"]
weighted_pools = results.sampling
fits = results.metadata["scenario_records"]

results.save("/path/to/results.npz")             # Optional later save
```

Or save in the same call with `output_path="/path/to/results.npz"`. If omitted,
the function only returns the result; it creates no result/configuration files.
Existing output files are rejected before covariance building or sampling.
Resolved arguments, preparation metadata, source versions and hashes of the
population and any loaded preparation files are retained in the result.

## Arguments

The explicit signature accepts `N`, `steps`, `nsamples`, `seed`,
`posterior_samples`, `parity`, `eb_eta`, `scenarios`, `filt`,
`missing_host_policy`, `trilegal_fname`, `molusc_file`, and `output_path`.
Defaults remain 500 particles, 50 steps, seed 42, 2000 posterior draws, profile
parity, and eta=1. Exposure integration defaults to 20 subsamples for Real and
7 for Fourier. The example explicitly chooses the paper odds, eta=0.1.

Real-only options are `primary_only`, `include_gp` and `timing_policy`, defaulting
to `False`, `True` and `"observed"`. Explicit false values are honored; these
options are rejected for Fourier. Primary-only mode requires a matching
likelihood-ready Real preparation and cannot discard a candidate bundle's
secondary observations. Use `timing_policy="legacy"` for the archived Real
timing rule. The existing ordinary-EB secondary penalty remains unchanged.

Omit `scenarios` for all eligible hypotheses. An explicit subset produces a
conditional FPP. The sampler settings are effort controls, not a convergence
guarantee. Inference is sequential within a target; use isolated processes,
not concurrent threads, for multiple targets.
