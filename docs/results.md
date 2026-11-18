# Results and posterior records

## `RunResult`

- `output`: the original numerical return value. For a scenario calculation
  this is a physical-parameter/evidence dictionary (sometimes a single/twin
  pair). For the high-level interfaces it is a scenario DataFrame.
- `sampling`: chronological evidence events and sampler pools.
- `metadata`: package/Python/NumPy/SciPy versions, seed, callable identity,
  and a SHA-256 fingerprint of package Python source. The fingerprint is
  cached per process; restart after editing installed code.

Each evidence event stores its function, input arguments, status, and result.
Callbacks are described by name, never pickled. Sampler records reference their
enclosing event and retain:

```text
samples             physical prior-transform coordinates, shape (pool, ndim)
weights             normalized posterior importance weights
log_target          values evaluated by persistent sampling
log_evidence        raw sampler evidence, before outer scenario corrections
finite_result       false for an impossible-scenario placeholder
```

Coordinate order is defined by the recorded prior-transform function. Named,
derived physical quantities are in the evidence event output. Single/twin
calls may have two pools; callback names and event ordering distinguish them.
Nearby hosts can share an evidence function; the high-level table supplies
host labels and event inputs supply numerical context. A stable per-scenario-ID
record schema for compatibility calls remains a future improvement. The folded
HZ interfaces already provide `metadata["scenario_records"]["ID:scenario"]`,
including best fits and weighted/equal-weight physical pools, separately from
pandas attributes. See [folded HZ runs](hz_runs.md).

`log_target` is deliberately **not** called `log_posterior`: some scenarios
include additional companion/population weights in the integrand. Nor should
raw sampler evidence replace final scenario evidence containing outer
null/frame corrections.

Use weighted pools for posterior calculations. Old physical-output resampling
conventions are preserved: some routines force a best-fit sample into their
first returned row. Such an array is not an entirely unbiased equal-weight
posterior sample. Full pools avoid losing weights or confusing that first row
with a random draw.

## Scenario table and records

The high-level interfaces (`evidence`, `calc_probs_folded_real`,
`calc_probs_folded_fourier`) return a table in `results.output` with one row
per host star and scenario:

| Column | Meaning |
| --- | --- |
| `ID`, `scenario` | Host star and scenario label |
| `lnBF` | Log Bayes factor against a flat light curve for the same data; the unweighted evidence used for probabilities |
| `lnZ`, `null_loglike` | `lnBF` plus the log-likelihood of the flat model, and that log-likelihood |
| `aperture_fraction` | Fraction of the aperture light contributed by the host |
| `seed`, `sampling_N`, `sampling_steps` | Random seed and sampling effort used for this row |
| `prob`, `eta` | Scenario probability, and the `eb_eta` weight applied to the row |
| `epoch_offset`, `epoch_offset_err` | Only with `free_epoch=True`: posterior mean and standard deviation of the transit-time offset, in days |

`results.output.attrs` holds `FPP`, `FPP_EB`, `NFPP`, `scenario_scope` and the
resolved settings of the run.

As an example, this is TOI-700.02 with the subtracted likelihood, default
sampling and `eb_eta=0.1`. The table has 33 rows: fifteen scenarios on the
target (TIC 150428135) and three on each of six eligible neighbours. Sorted by
probability, the first eight are:

```python
table = results.output
print(table[["ID", "scenario", "lnBF", "prob"]].sort_values("prob", ascending=False).head(8))
```

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

```python
>>> {key: table.attrs[key] for key in ("FPP", "FPP_EB", "NFPP", "scenario_scope")}
{'FPP': 0.0383, 'FPP_EB': 5.8e-05, 'NFPP': 3.5e-05, 'scenario_scope': 'all'}
```

Here the three planet-on-target scenarios (TP, PTP, DTP) hold 96 per cent of
the probability, and almost all of the remaining 3.8 per cent is STP, a planet
on an unresolved bound companion. The numbers illustrate the format; they
depend on the preparation, the stellar field and the settings, and are not a
published result.

The record of one fit, here the planet on the target:

```python
>>> record = results.metadata["scenario_records"]["150428135:TP"]
>>> record["weights"].shape                 # the whole sampler pool
(1100,)
>>> record["posterior_parameters"].shape    # 2000 equal-weight draws of 5 sampled coordinates
(2000, 5)
>>> float(np.median(record["physical_result"]["R_p"]))   # planet radius, Earth radii
1.18
```

`results.metadata["scenario_records"]` maps `"<ID>:<scenario>"` to a dictionary
for each fit:

| Key | Contents |
| --- | --- |
| `physical_parameters`, `weights`, `log_target` | The complete sampler pool: sampled coordinates, normalized importance weights, and the log target of each particle |
| `posterior_parameters`, `posterior_log_target`, `posterior_source_index` | Equal-weight draws from that pool (`posterior_samples` of them) and their row indices in the pool |
| `physical_result` | Named physical quantities of the resampled posterior, such as `R_p`, `inc`, `ecc`, `argp`, `b`, `M_EB`, `R_EB` |
| `bestfit` | The highest-target particle: its named `parameters`, its `model` light curve, and an independent re-evaluation (`scalar_replay`) that must agree with the sampler's value |
| `epoch_offset`, `posterior_epoch_offset` | Only with `free_epoch=True`: the offset of every pool particle and of every posterior draw, in days |
| `lnBF`, `lnZ`, `seed`, `N`, `steps`, `sampling_stats` | The evidence and how it was obtained |

Angles are in degrees, `R_p` in Earth radii and stellar quantities in solar
units. `argp` is the stored periastron angle; the transit model uses
`90° - argp`.

## Storage

`result.save(path)` writes one NPZ with numeric arrays and UTF-8 JSON metadata.
`RunResult.load(path)` uses `allow_pickle=False`; object arrays are rejected.
Publication is atomic, with a temporary file in the destination directory,
and refuses to overwrite. Store large bundles on scratch; small plots and
summaries can remain in home.

No file is written without `output_path`. Automatic partial-run recovery is
not implemented: a failed high-level invocation raises without automatically
publishing completed work. For direct debugging, `capture_sampling()` retains
completed records in memory even if a later call raises. Automatic checkpointing
is a future feature, not a promise of the current recorder.

## FPP conventions

`scenario_probabilities(table, evidence_column=..., eb_eta=...)` requires a
comparable **unweighted** log-evidence column. It does not automatically prefer
raw `lnZ` over null-referenced `lnBF`.

Standard FPP sums all hypotheses except TP, PTP, and DTP. STP, BTP, and NTP
therefore count as false positives. `FPP_EB` sums only eclipsing binaries,
including x2P and nearby EBs. `NFPP` sums NTP, NEB, and NEBx2P. These values are
returned in the output DataFrame's `attrs`.

`eb_eta` is applied once to the selected log-evidence column; old `prob` or
`eta` columns are ignored. Using an already-weighted evidence column would
double-count the prior. Negative/nonfinite eta, unknown labels, NaN/+infinite
evidence, or a completely zero-weight set raise an error. An individual
impossible scenario may have evidence `-inf`. This helper normalizes; it does
not calibrate a population prior or correct an invalid likelihood.
