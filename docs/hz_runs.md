# Folded HZ runs

For a candidate bundle saved by the example preprocessing scripts, load
`prepared` from the NPZ filename printed after `Prepared data:`:

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

The [Python interface](evidence_api.md) also shows how to create `prepared`
directly with `prepare_candidate`. The paired Real caches and folded Fourier
directories used by the adopted HZ runs have their own loaders, shown below.

For a command-line entry point using these same routines, run
`pentaceratops evidence target.json`. The [evidence command](evidence_command.md)
accepts the prepared files below or saved example-script bundles plus explicit
stellar-field/population inputs.

The prepared-input HZ interfaces were introduced in `0.1.0.dev7`; `0.1.0.dev8`
adds automatic observation-based timing as the default. They integrate
the optimized production likelihood adapters with recorded evidence and saved
posterior pools. They do not fetch a new catalogue, select transits, or estimate
a new PSD. Reusing the paper's inputs therefore preserves those decisions.

## Supported conventions

| Interface | Observations and covariance | Model and sampling |
| --- | --- | --- |
| `calc_probs_folded_real` | Folded even/odd/secondary conditional means; `diag(sigma**2) + F F.T`, including covariance between panels | Optimized fixed-period physical models; exposure integration; original physical priors |
| `calc_probs_folded_fourier` | Complete observed orbit folded at 2P; `sum_s A_s C_s A_s.T`; empty bins omitted | The same A applied to native exposure predictions; optimized models retain alternating eclipses |
| `calc_probs_fourier` | Full-P/2P Fourier input arrays and PSD conventions | Scalar sampler and complete orbital eclipse models; explicit legacy mode for reproduction |
| `run_folded_baseline` | A prepared P or 2P Gaussian block with the selected transits/covariance | Original Fourier physical grids, scalar sampler, timing limits, and parity policy |

The original folded Fourier implementation is shared by **TESS and Kepler**.
Six of the seven adopted TESS results used the implementation now preserved by
`calc_probs_fourier(..., timing_policy="legacy")`: TOI-700.02,
700.04, 715.01, 904.02, 2257.01, and 6714.01. TOI-7390.01 uses the later
full-orbit folding operator and propagated covariance. The original path is
therefore a supported paper workflow for both missions, not a Kepler-only
fallback. The archived matched TESS and Kepler completion campaigns both
used `max_anomaly_shift = 3 * duration`. The adopted KOI-2719.02 record confirms
0.4177313703647727 days = 3 times its 0.1392437901215909-day duration.
An earlier description of a four-duration Kepler cutoff was incorrect.

This cutoff is an explicit event-selection restriction, not a bound derived
from the supplied Fourier grid. Its offset is the secondary's displacement
from half an orbital period. In EBx2P models, the two eclipse centers move by
minus/plus half that offset. For an eclipse of full duration D to overlap a
symmetric window of full width W, the corresponding offset bound is W + D
for EBx2P, or (W + D)/2 for an ordinary secondary. Full containment uses
W - D or (W - D)/2 instead. These formulas assume contiguous symmetric windows;
model-dependent durations, exposure boundaries, and gaps require explicit
contact/observation checks. A local event-selection window must be declared
separately from the complete orbital data span. The frozen three-duration
settings remain unchanged for explicit reproduction of the adopted results.
New runs use the [observed timing policy](timing.md), without that manual cap.

These are explicit representations of the same Gaussian-noise framework.
Folding is a chosen compression and is not guaranteed to preserve all native
information. A switch between these interfaces is not automatically a
covariance-only change. Use the historical path when matching the original
Kepler experiments, particularly the adopted KOI-2719.02 seven-transit run.

## Target and field inputs

Supply a prepared `Target`, or an object with `.mission`, `.stars` and
`.trilegal_fname`. `.mission` must be `TESS`, `Kepler`, or `K2`; it controls
limb darkening. `.stars` is a pandas table in the campaign order, with the
target first, then nearby sources. Required fields include `ID`, `mass`, `rad`,
`Teff`, `plx`, `Tmag`, `Jmag`, `Hmag`, `Kmag`, `tdepth`, and `fluxratio`.
`fluxratio` is each host's fraction of the aperture light; `tdepth > 0`
marks an eligible host. The target cannot be replaced by the first eligible
neighbor. Supply the same TRILEGAL and optional MOLUSC population used in the
experiment. No network access is required for inference on prepared inputs.

Missing stellar parameters fail before sampling by default. The explicit
`missing_host_policy="solar"` reproduces the research runner's substitutions
for neighbors (1 solar mass, 1 solar radius, 5777 K); each substitution is
recorded. It never substitutes the target's parameters.

All hypotheses are compared in the same **aperture flux frame** against the
same data null. Use the resulting `lnBF` directly. Do not add another host
normalization, divide the likelihood by the aperture fraction, or apply the
Original-Triceratops correction again.

## Folded real-space inference

```python
from pentaceratops import load_folded_real, calc_probs_folded_real

data = load_folded_real(prepared_path, posterior_path)
real = calc_probs_folded_real(
    target, data, N=500, steps=50, nsamples=20, seed=42,
    parity="profile", eb_eta=0.1,
    output_path="/path/to/scratch/real.npz",
)
print(real.output.attrs["FPP"])
```

`prepared_path` is the campaign's reference preparation NPZ;
`posterior_path` is its paired `folded_posterior.npz`. The loader checks time,
MAP flux, and error alignment, preserves the adopted median measurement error
within each panel, and stacks the conditional trend factors with their shared
coefficient columns. It does not replace that error model with per-point errors.
Do not include trend variance in `sigma` as well as the factor.

For explicitly prepared inputs with no secondary observations, pass
`primary_only=True` to **both** loading and inference. Existing secondary
observations cannot be silently dropped. The three-window Real likelihood
retains the existing synthetic flat-secondary penalty when an ordinary EB's
secondary center falls outside the secondary window, under either timing
policy. No 1.5-sigma cutoff is added. Its x2P support follows modeled contacts
and exposure boundaries; set `timing_policy="legacy"` explicitly to reproduce
the archived x2P center-only gate. Primary-only Real inference and full-orbit
Fourier inference add no synthetic secondary observations.
See [timing and coverage](timing.md).

`include_gp=False` is an explicit diagnostic that removes the trend factor.
It does not represent the adopted covariance-aware real likelihood.

## Full-orbit folded Fourier inference

```python
from pentaceratops import (
    prepare_folded_fourier, FoldedFourierData, calc_probs_folded_fourier,
)

# blocks and spectra are the aligned per-sector prepare_sector outputs.
folded = prepare_folded_fourier(
    blocks, spectra=spectra, period=period_days, epoch=epoch_days,
    bin_days=10/1440,
)
folded.save("/path/to/scratch/folded_inputs")
folded = FoldedFourierData.load("/path/to/scratch/folded_inputs")
fourier = calc_probs_folded_fourier(
    target, folded, N=500, steps=50, nsamples=7, seed=42,
    parity="profile", eb_eta=0.1,
    output_path="/path/to/scratch/fourier.npz",
)
```

`prepare_sector` is in `pentaceratops.preprocessing.joint_fourier`. Supply its
matching native observed blocks and spectral records, with the chosen PSD and
mask. In a spectral record, `total_fft_power` means `E[|rFFT(noise)|**2]`,
including measurement white noise exactly once. Its `observed` boolean mask
selects the same regular-grid times, in the same order, as the block. Sector
covariances are independent. The current adapter requires a common exposure
duration; mixed exposure durations fail explicitly.

The preparation folds every supplied observation once with equal contributor
weights. It creates no eclipse windows, fills no missing phase bins, and
imposes no phase-coverage threshold. It retains the input PSD's finite DC
prior, as in the adopted TOI-7390.01 run. Native times and weights remain
attached to the block so averaging model predictions uses exactly the data's
operator, including exposure integration. Without `spectra`, preparation
inverts each block's positive-definite precision; projected singular blocks
belong to the separate historical interface.

Dense folded covariance can still be large. Prepare it once in a compute job,
save it, and load its precision as a read-only memory map in each process.
Keep these prepared inputs alongside the result; the result bundle retains
the observation/folding map but does not duplicate the full covariance matrix.

## Reproducing the historical Kepler update

Keep the original recipe arguments, including the physical half grids,
`max_shift`, exposure integration, and sampler effort. Build the selected
transit fold using `preprocessing.folded_covariance.folding_operator`, propagate
the archived PSD using `propagate_fourier_covariance`, and remove **only** the
original per-half DC/Nyquist nuisance modes using `dropped_fourier_modes` and
`marginalized_mode_block`. Then call each original single-branch recipe:

```python
from pentaceratops import run_folded_baseline
from pentaceratops.evidence.fourier import lnZ_TTP_fourier

tp = run_folded_baseline(
    lnZ_TTP_fourier, block=prepared_block, fold_indices=retained_indices,
    fold_length=original_fold_length, fold_roll=original_fold_roll,
    period=period_days, epoch=epoch_days, aperture_fraction=host_fraction,
    seed=original_seed, posterior_samples=2000,
    **original_recipe_arguments,
)
```

Use the analogous explicit-secondary and even/odd recipes for EB and EBx2P.
The retained indices select the original full grid **after** its P roll or
2P parity concatenation. Empty bins remain absent; they are not local windows.
This interface retains the original scalar sampler and does not substitute
the newer native-exposure renderer. Collect every eligible scenario's `lnBF`
before calling `scenario_probabilities(..., eb_eta=0.1)`.

## Results, odds, and reproducibility

Both `calc_probs_folded_*` functions return `RunResult`. `.output` is the
scenario table with unweighted `lnBF`, normalized `lnZ`, host fraction, seed,
and posterior probability. Its attributes include `FPP`, `FPP_EB`, `NFPP`,
mission, sampler settings, parity policy, timing policy, and likelihood frame.
`.sampling` contains the complete weighted sampler pools.
`.metadata["scenario_records"]["ID:scenario"]` contains physical pools,
equal-weight posterior draws, and the independently replayed scalar best fit.
Versions, code fingerprint, stellar inputs, and population paths are recorded.

`N`, `steps`, and `nsamples` are separate controls. The defaults 500/50 are
convenient effort settings, not a claim that every adopted paper run used them
or that evidence has converged. The real/Fourier exposure defaults are 20/7.
Use the frozen campaign configuration to reproduce a particular row.
The default demographic multiplier remains `eb_eta=1`; specify **0.1** for
the paper's reported odds. It is applied once to EB scenario weights.
`parity="profile"` preserves the adopted maximum-over-parity convention;
`"marginalize"` is an explicit scientific change.

Omit `scenarios` for all 15 target-host families and three families per eligible
nearby star. A subset is marked `conditional_subset`: its FPP is conditional
on omitted scenarios having zero weight. Seeds are `seed + 1009*ordinal` in
the full eligible scenario order, including skipped labels. Array-valued
posterior records live outside pandas attributes so table display and slicing
remain safe.

The public target calls are sequential. For campaign parallelism, use isolated
processes (up to the allocation's CPU/memory limits), as in the research runners;
the existing per-scenario `evidence.joint_fourier.sample_joint_scenario` accepts
either optimized adapter. Module hooks and model caches are not thread-safe.
Use one numerical-library thread per worker. A 128-CPU node does not justify
128 simultaneous dense covariance builds; share prepared read-only blocks.

## Verification

`tests/test_hz.py` covers covariance propagation against a dense reference,
native folding maps, missions, faint-host likelihood frames, all 15 physical
families, and actual tiny recorded real/Fourier sampler calls.
`tests/test_folded_baseline_api.py` checks the original-grid recorded path.
The broader suite retains the existing physical/prior differential tests.

The read-only `tools/replay_adopted_hz.py` replays all 21 saved TOI-7390.01 and
18 saved KOI-2719.02 seven-transit Fourier best fits, plus 172 real-space
best fits across all seven retained TESS targets, then reconstructs their FPPs
from stored evidences. It requires the archived campaign and scratch inputs:

```bash
python tools/replay_adopted_hz.py --research-root /path/to/pentaceratops_research \
  --scratch-root /path/to/user/scratch --output /path/to/new/replay.json
```

This checks model/log-target replay and probability bookkeeping. It does not
rerun the production sampler or establish evidence convergence for all targets.
