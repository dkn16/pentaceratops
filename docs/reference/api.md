# Python API

The functions most users need. Each signature is generated from the source;
the text below it is the function's own docstring. For a guided introduction,
start with the [quick start](../quickstart.md).

Importing `pentaceratops` performs no catalogue queries or downloads.

## Light-curve preparation

```python
from pentaceratops.preprocessing.candidate import (
    prepare_candidate, preprocess_blocks, PreparationOptions,
)
```

::: pentaceratops.preprocessing.candidate.prepare_candidate

::: pentaceratops.preprocessing.candidate.preprocess_blocks

### PreparationOptions

`pentaceratops.preprocessing.candidate.PreparationOptions` is a frozen dataclass
holding the preprocessing settings. Its fields, defaults and units are listed
under [light-curve preparation options](options.md#light-curve-preparation).

```python
options = PreparationOptions(bin_minutes=30.0, window_durations=4.0)
```

### Candidate

`pentaceratops.preprocessing.catalog.Candidate` is a frozen dataclass holding the
ephemeris of one transit signal.

| Field | Meaning |
| --- | --- |
| `mission` | `"TESS"` or `"Kepler"` |
| `host_id` | TIC or KIC identifier of the host star |
| `identifier` | Label of the signal, for example `"TOI-700.02"` |
| `period_days` | Orbital period in days |
| `epoch_bjd` | Reference transit time in full BJD (not BTJD or BKJD) |
| `duration_hours` | Transit duration in hours |
| `depth_ppm` | Optional transit depth in parts per million |

See [use your own light curve](../guide/supplied_lightcurve.md#2-candidate-ephemeris)
for how each field is used.

## Stellar field

```python
from pentaceratops import prepare_target, load_target
```

::: pentaceratops.preprocessing.field.prepare_target

::: pentaceratops.preprocessing.field.load_target

## Evidence and FPP

```python
from pentaceratops.evidence import evidence
from pentaceratops import scenario_probabilities, run_prepared
```

::: pentaceratops.prepared.evidence

::: pentaceratops.results.scenario_probabilities

::: pentaceratops.prepared.run_prepared

::: pentaceratops.evidence.joint_fourier.resolve_epoch_sigma

## Results

```python
from pentaceratops import RunResult
```

::: pentaceratops.results.RunResult
    options:
      members:
        - save
        - load
        - check_destination

## Likelihood-ready inputs

These interfaces take data that are already folded and carry their own
covariance. `evidence` calls them after building those inputs from a candidate
preparation. See [likelihood-ready folded inputs](../hz_runs.md).

```python
from pentaceratops import (
    calc_probs_folded_real, calc_probs_folded_fourier,
    load_folded_real, prepare_folded_fourier, FoldedFourierData,
)
```

::: pentaceratops.hz.calc_probs_folded_real

::: pentaceratops.hz.calc_probs_folded_fourier

::: pentaceratops.preprocessing.folded.load_folded_real

::: pentaceratops.preprocessing.folded.prepare_folded_fourier

## Sampling policy

::: pentaceratops.sampling.policy.sampling_policy

::: pentaceratops.sampling.policy.sampling_config
