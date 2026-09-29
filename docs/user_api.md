# Planned three-stage user interface

This is the implementation contract for the next integration milestone, not
documentation of functions already available. The current package exposes
`run_evidence`, the recorded compatibility interfaces, and lower-level building
blocks. Do not add placeholder `preprocess`, `validate`, or `report` functions
that appear to run the corrected pipeline before that integration is complete.

The [download/preparation examples](../examples/README.md) now implement an
initial, explicitly limited preparation stage. They default to NASA TOI/KOI
ephemerides and allow the Bayesian catalogue for TESS. They do not yet provide
the complete corrected validation or reporting interfaces described below.

## Simple by default, configurable when needed

The normal workflow should require one call per stage:

```python
data = penta.preprocess("target.yaml")
result = penta.validate(data)
figures = penta.report(result)
```

Each stage must also accept optional, documented keyword arguments. Users
should not have to edit a configuration file, modify engine globals, or change
the source to override a setting. Illustrative expanded calls:

```python
data = penta.preprocess(
    "target.yaml",
    sectors=[42, 43],
    cache_dir="/path/to/scratch/cache",
    output_dir="/path/to/scratch/prepared",
)
result = penta.validate(
    data,
    likelihood="real_covariance",
    backend="scalar",
    N=200,
    steps=30,
    nsamples=20,  # integration points within each exposure, not sampler particles
    scenario_settings={"STP": {"N": 500, "steps": 50}},
    eb_eta=0.1,
    seed=123,
    output_path="/path/to/scratch/result.npz",
)
figures = penta.report(
    result,
    scenarios=["TP", "STP"],
    output_dir="figures",
    formats=["pdf", "png"],
    show=True,
)
```

These examples define the intended interface, not finalized scientific default
values or currently executable calls. Candidate-specific information such as
an unambiguous ephemeris must come from inputs/configuration or a validated
catalog lookup; it must not be silently guessed to achieve a one-line call.

## Override rules

1. Explicit call arguments (or CLI flags).
2. User configuration, including settings inherited from the preceding stage.
3. Named, versioned package defaults.

Within each layer, a scenario-specific setting takes precedence over that
layer's global setting. Layers are applied in order of increasing priority:
an explicit `N=200` overrides an older per-scenario N in the configuration;
an explicit `scenario_settings={"STP": {"N": 500}}` then specializes it.
Unspecified fields retain their resolved values: overriding N does not also
reset steps, and overriding a figure format does not reset plot selections.

Implementation should distinguish an omitted argument from an explicit `None`,
`False`, zero, or empty selection. For example, `show=False` must override
`show=True` in a configuration file. Do not use truth-value tests to decide
whether an override was supplied. Parameter-specific validation still applies:
zero steps may be invalid even though zero is an explicitly supplied value.

Reject unknown options and incompatible combinations before fetching data or
starting sampling. Use explicit signatures or validated option objects, not
unchecked `**kwargs` passed through to whichever low-level function happens
to receive them. Keep the advanced low-level API for custom experiments.

## Option groups

| Stage | Optional controls to expose |
| --- | --- |
| Preprocessing | Target/ephemeris overrides; sectors and data products; other-planet masks; eclipse protection and input windows; FGP/PSD settings; binning; cache, prepared-data and diagnostic-plot paths |
| Validation | Likelihood; computational backend; scenario selection; global and scenario-specific N/steps; exposure subsample count; demographic odds; timing constraints; companion/background inputs; seed; posterior retention and output paths |
| Reporting | Scenario/host selection; best-fit panels and posterior summaries; even/odd and secondary panels; units/axis limits; uncertainty display; output formats, resolution and paths; interactive display |

Use descriptive parameter names and state units in help/docstrings. Keep the
scientific model separate from the computational implementation:
`likelihood="real_covariance"` and `backend="scalar"` answer different questions.
For unsupported backend/likelihood combinations, fail clearly when a backend
is explicitly requested; any `auto` selection/fallback must be recorded.
The existing optimized adapters remain opt-in until their promotion gates pass.
Their configurable exposure integration is already implemented at the lower
level; see [exposure integration](exposure.md). The high-level validation
interface must propagate that setting consistently to scalar evaluation,
batched evaluation, and saved-model replay.

## Persistence and reproducibility

- Every stage returns a structured result usable by the next stage and can
  reload a saved artifact without fetching or fitting again.
- Save the complete resolved configuration, default-policy version, explicit
  overrides, actual selected backend, seed and input/source fingerprints.
- Record preprocessing choices with the prepared data. Changing a likelihood
  window or a PSD must not silently reuse incompatible masks/covariances.
- Posterior retention is enabled by default. An explicit opt-out must be
  recorded; reports should identify missing products instead of fitting again.
- Cache and bulk arrays/posteriors belong in configurable scratch storage;
  plots may go to a separate home-directory location.
- Reporting reads saved results only. Changing display options must not rerun
  the sampler or alter the evidence. Any optional prior reweighting must be
  explicit and preserve the original stored probabilities and prior settings.

## Acceptance tests for integration

- A minimal call works from a complete target configuration.
- Omitted options match the documented, versioned defaults.
- Direct/CLI overrides beat file settings, including explicit false values.
- Global and scenario-specific partial overrides obey the same rules.
- Invalid options fail before expensive work; outputs are not overwritten.
- Resolved settings survive save/load and are sufficient to identify the run.
- Reports work from stored results with no sampler invocation or catalog query.
- Both scalar and supported optimized backends use identical scientific inputs;
  promotion to `auto` requires the corrected-runner equivalence tests.
