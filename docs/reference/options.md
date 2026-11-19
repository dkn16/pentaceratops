# Options

All settings in one place. Units are stated for every quantity; where a setting
has its own page, the last column links to it.

## Evidence calculation

These are keyword arguments of
[`evidence`](api.md#pentaceratops.prepared.evidence) and keys of the `settings`
block of the [command-line configuration](../evidence_command.md).

### Scientific choices

| Option | Default | Meaning | More |
| --- | --- | --- | --- |
| `likelihood` | `"subtracted"` | `"subtracted"` or `"marginalized"`. The earlier names `"real"` and `"fourier"` are accepted as aliases. | [Likelihoods](../concepts/likelihoods.md) |
| `eb_eta` | `1.0` | Weight multiplying every eclipsing-binary scenario. `0.1` is the value used for real candidates in the paper. | [Scenarios](../concepts/scenarios.md#demographic-odds-eb_eta) |
| `scenarios` | all | List of scenario labels to fit. A subset gives a conditional FPP. | [Scenarios](../concepts/scenarios.md#fitting-a-subset-of-scenarios) |
| `free_epoch` | `False` | Sample a transit-time offset from the supplied epoch. | [Transit-time offset](../concepts/likelihoods.md#transit-time-offset) |
| `epoch_sigma` | 3 minutes | Width of the Gaussian prior on that offset, **in days**. Only valid with `free_epoch=True`. | |
| `parity` | `"profile"` | How the two eclipse assignments of period-doubled scenarios are combined: `"profile"` or `"marginalize"`. | [Parity](../concepts/likelihoods.md#parity) |
| `filt` | from mission | Bandpass for limb darkening: TESS for TESS targets, Kepler otherwise. | |

### Sampling

| Option | Default | Meaning | More |
| --- | --- | --- | --- |
| `N` | per scenario | Particles per sampler stage. | [Sampling](../concepts/sampling.md) |
| `steps` | per scenario | Metropolis sweeps per stage. | |
| `seed` | `42` | Random seed of the call; a non-negative 32-bit integer. | [Seeds](../concepts/sampling.md#seeds-and-reproducibility) |
| `posterior_samples` | `2000` | Equal-weight posterior draws kept per scenario. | |
| `nsamples` | 20 (subtracted), 7 (marginalized) | Points used to average a model over one exposure. | [Exposure integration](../exposure.md) |

### Inputs

| Option | Default | Meaning |
| --- | --- | --- |
| `target` | required | Stellar field: an object with `.mission`, `.stars` and `.trilegal_fname`. |
| `trilegal_fname` | from `target` | Path to the TRILEGAL background population. |
| `molusc_file` | none | Optional prepared population of bound companions for the target. |
| `missing_host_policy` | `"error"` | `"solar"` substitutes solar values for missing parameters of **neighbours** only; the target is never substituted. |
| `output_path` | none | Save the result to this new `.npz` file. Existing files are never overwritten. |

### Subtracted likelihood only

Passing any of these with `likelihood="marginalized"` is an error.

| Option | Default | Meaning |
| --- | --- | --- |
| `primary_only` | `False` | Fit only the even and odd transit windows. Requires a preparation made without a secondary window. |
| `include_gp` | `True` | Include the uncertainty of the subtracted variability in the covariance. |
| `timing_policy` | `"observed"` | Coverage rule for period-doubled models; `"legacy"` restores the earlier rule. |

## Light-curve preparation

Fields of `PreparationOptions`, passed as `options=` to `prepare_candidate` or
`preprocess_blocks`. The example scripts expose the same settings as flags.

| Field | Default | Script flag | Meaning |
| --- | --- | --- | --- |
| `bin_minutes` | `10` | `--bin-minutes` | Width of the regular time grid, in minutes. Never finer than the longest exposure. The Kepler script defaults to 30. |
| `window_durations` | `4` | `--window-durations` | Half-width of the primary and secondary windows, in transit durations. |
| `max_half_period_fraction` | `0.25` | `--max-half-period-fraction` | Upper limit on the window half-width, as a fraction of the period. |
| `companion_mask_durations` | `1.5` | `--other-planet-mask-durations` | Half-width of the mask around other planets' transits, in their durations. |
| `fgp_iterations` | `3` | `--fgp-iterations` | Iterations of the variability fit. |
| `fgp_cutoff_per_day` | `3` | `--fgp-cutoff-per-day` | Highest frequency of the variability model, in cycles per day. |
| `padding_fraction` | `0.1` | `--padding-fraction` | Padding added to each segment for the Fourier model, as a fraction of its length. |
| `minimum_window_bins` | `8` | `--minimum-window-bins` | Fewest bins a folded window may contain. |

Arguments of `prepare_candidate` itself:

| Argument | Default | Meaning |
| --- | --- | --- |
| `target` | required | TOI, KOI, TIC or KIC identifier. |
| `cache_dir` | required | Directory for downloads and prepared files. |
| `candidate` | none | Which TOI or KOI to use when the host has several. |
| `segments` | all | Sectors or quarters to use. |
| `cadence_seconds` | automatic | Select a specific archive cadence, for example `20`. |
| `source` | `"nasa"` | Ephemeris catalogue: `"nasa"`, or `"bayesian"` for TESS. |
| `catalog_file` | none | Local copy of the Bayesian TESS catalogue. |
| `overrides` | none | Dictionary replacing catalogue values: `period_days`, `epoch_bjd` (full BJD, not BTJD or BKJD), `duration_hours`, `depth_ppm`. |
| `mask_other_planets` | `True` | Mask the transits of other catalogued planets on the same star. |
| `extra_companions` | none | Additional signals to mask, as `Candidate` objects. |
| `offline` | `False` | Use only existing caches; fail instead of downloading. |

See [prepare a candidate light curve](../guide/candidate.md).

## Stellar field

Arguments of `prepare_target` are listed in
[stellar field and target](../target_setup.md#optional-supplied-inputs).

## Environment

| Variable | Meaning |
| --- | --- |
| `PENTACERATOPS_CACHE` | Default cache directory of the example scripts and the `evidence` command. |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS` | Set to `1` when running several targets in parallel. |
