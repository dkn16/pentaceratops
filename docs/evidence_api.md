# Evidence from Python

```python
from pentaceratops.evidence import evidence

results = evidence(prepared, target=target, likelihood="fourier", N=500, steps=50, eb_eta=0.1)
```

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

With the provided Python preprocessing routine:

```python
from pentaceratops.preprocessing.candidate import prepare_candidate
from pentaceratops.evidence import evidence

prepared, prepared_path = prepare_candidate("TOI-700.02", cache_dir="/path/to/cache")
# target is your already prepared stellar-field object.
results = evidence(prepared, target=target, likelihood="fourier", eb_eta=0.1)
```

When using the example preprocessing scripts, pass their saved preparation
path directly in place of `prepared`; no configuration file is needed.
When calling `prepare_candidate` in Python, unpack its `(result, path)` pair
as shown above.

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
