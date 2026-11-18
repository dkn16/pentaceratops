# Quick start

This page takes one TESS candidate from its identifier to a false-positive
probability. It assumes the package is [installed](install.md) with the
`catalogs` extra and that the machine has network access for the first two
steps.

!!! note "Where to run it"
    Preparing a real light curve and fitting all scenarios takes minutes to
    hours on one core, depending on the amount of data and the likelihood. On a
    shared cluster, run these steps in a compute job, not on a login node.

## 1. Prepare the light curve

```python
from pentaceratops.preprocessing.candidate import prepare_candidate

prepared, prepared_path = prepare_candidate("TOI-700.02", cache_dir="/path/to/cache")
```

`prepare_candidate` looks up the ephemeris, downloads the PDCSAP light curves,
restores the aperture flux, masks other catalogued planets on the same star,
fits a variability model to each sector, and folds the transit windows.

It returns two values. `prepared` is the in-memory result; `prepared_path` is
the `.npz` file it was saved to, under `<cache_dir>/prepared/`. Running the same
call again reuses the saved file.

The identifier can be a TOI, a KOI, or a TIC/KIC host. A host with several
candidates needs `candidate="TOI-700.02"` to select one. See
[prepare a candidate light curve](guide/candidate.md) for the catalogue
sources, overrides and preprocessing options.

## 2. Build the stellar field

```python
from pentaceratops import prepare_target

target = prepare_target(prepared, output_dir="/path/to/new_field")
```

`prepare_target` queries the TESS Input Catalog for the host and its
neighbours, computes how much of each star's light falls in the photometric
aperture, and requests a TRILEGAL background population for that sky position.
It writes `stars.csv`, `trilegal.csv` and `field.json` to `output_dir`, which
must not exist yet.

This step currently supports TESS. For Kepler, or to use adopted stellar
parameters, supply the table yourself as described in
[stellar field and target](target_setup.md).

## 3. Fit the scenarios

```python
from pentaceratops.evidence import evidence

results = evidence(prepared, target=target, likelihood="fourier", eb_eta=0.1)
```

`likelihood` selects one of [the two likelihoods](concepts/likelihoods.md):

| Value | Name in the paper | Use |
| --- | --- | --- |
| `"real"` | Subtracted likelihood | Faster; a good first assessment |
| `"fourier"` | Marginalized likelihood | Accounts for uncertainty in the stellar variability; preferred for a final FPP |

The same `prepared` object works for both, so switching does not repeat the
preprocessing.

`eb_eta=0.1` down-weights every eclipsing-binary scenario by a factor of ten
relative to the planet scenarios. The function's default is `eb_eta=1`; see
[scenarios and the FPP](concepts/scenarios.md#demographic-odds-eb_eta).

For one reference target (TOI-700.02, 33 scenario fits, one core, default
sampling), the subtracted likelihood took about 4 minutes and the marginalized
likelihood about 22 minutes. Targets with more sectors take longer.

## 4. Read the result

```python
table = results.output
print(table[["ID", "scenario", "lnBF", "prob"]].sort_values("prob", ascending=False))

print("FPP     ", table.attrs["FPP"])       # everything except a planet on the target
print("FPP_EB  ", table.attrs["FPP_EB"])    # eclipsing-binary scenarios only
print("NFPP    ", table.attrs["NFPP"])      # scenarios on resolved nearby stars
```

For TOI-700.02 with the subtracted likelihood the first rows look like this
(illustrative values; they depend on the data and settings):

```text
       ID scenario       lnBF         prob
150428135       TP 107.348018 8.010272e-01
150428135      PTP 105.257800 9.905508e-02
150428135      DTP 104.782400 6.157627e-02
150428135      STP 104.307135 3.828323e-02
150428138      NEB  99.616407 3.514398e-05
150428135      BEB  99.188726 2.291456e-05
150428129      NEB  93.607116 8.630765e-08
150428135      BTP  90.122704 2.647209e-08
```

Each row is one scenario on one star:

| Column | Meaning |
| --- | --- |
| `ID` | Catalogue ID of the star hosting the signal in this scenario |
| `scenario` | Scenario label, such as `TP` or `BEBx2P` |
| `lnBF` | Log Bayes factor of the scenario against a flat light curve |
| `prob` | Posterior probability of the scenario after applying `eb_eta` |
| `sampling_N`, `sampling_steps` | Sampling effort actually used |

A commonly used criterion for calling a candidate statistically validated is
`FPP < 0.015` together with `NFPP < 0.001` (Giacalone et al. 2021). An FPP is a
model-dependent probability, so treat a value near the threshold with care and
check that it is stable against the sampling settings and the random seed.

The fitted parameters of each scenario are kept with the result:

```python
record = results.metadata["scenario_records"][f"{target.ID}:TP"]
record["bestfit"]["parameters"]      # best-fitting physical parameters
record["physical_result"]["R_p"]     # posterior draws of the planet radius [Earth radii]
record["weights"]                    # importance weights of the full sampler pool
```

See [results and posterior records](results.md) for the full layout, including
which arrays are weighted pools and which are resampled draws.

## 5. Save and reload

```python
results.save("/path/to/results.npz")

from pentaceratops import RunResult, load_target
results = RunResult.load("/path/to/results.npz")
prepared = RunResult.load(prepared_path)
target = load_target("/path/to/new_field")
```

Saving never overwrites an existing file. `load_target` reloads a field written
by `prepare_target` without any network access, so later runs of the same
candidate, for example with the other likelihood, need no new queries.

## Next steps

- Understand what was fitted: [scenarios and the FPP](concepts/scenarios.md).
- Decide how much sampling effort the target needs:
  [sampling and convergence](concepts/sampling.md).
- Run a light curve that is already on disk, including injected signals:
  [use your own light curve](guide/supplied_lightcurve.md).
- See every setting: [options](reference/options.md).
