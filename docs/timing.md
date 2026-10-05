# Eclipse timing and observation coverage

New recorded folded runs use `timing_policy="observed"` by default in
`0.1.0.dev8`. No transit-duration multiplier needs to be supplied. This changes
the allowed models and observation treatment relative to the archived runs;
new FPPs require new inference. Existing paper tables and frozen campaigns are
not overwritten by a package update.

## Local folded Real windows

The ordinary (non-x2P) EB treatment is unchanged under either timing policy.
When the secondary's center lies outside the supplied secondary window, the
observed secondary panel is modeled as flat and an independent synthetic
centered secondary is compared with flat flux using that panel's covariance.
This adds the existing continuous Gaussian non-detection penalty; there is no
new 1.5-sigma depth veto. The center-based classification, including its
treatment of partial edge overlap, is preserved. This constraint is an
assumption about non-detection elsewhere, not additional observed data.
Explicitly primary-only Real inputs retain their existing behavior: no
secondary panel or synthetic secondary constraint.

The local x2P recipe associates alternating eclipses with the supplied primary
even/odd windows. Its automatic support check uses the model's primary and
secondary contact times, the actual endpoints of each window, exposure
half-widths, and periodic wrapping. Contacts use the same asymmetric orbital
calculation as the transit renderer. A model remains eligible when either
eclipse can overlap either primary window under a parity assignment.
Interior gaps do not trigger rejection; the likelihood evaluates only the
actual samples. Nonfinite contact estimates are passed conservatively to the
renderer rather than rejected by this support shortcut.

For a symmetric window of full width W and eclipse of full duration D, the
x2P overlap condition reduces to `abs(offset) <= W + D`, where the two eclipse
centers move by `-offset/2` and `+offset/2`. Exposure integration widens the
eligible interval appropriately. The implementation uses the actual contacts
and separate grids, rather than assuming symmetry or a catalogue duration.
The original local x2P panel representation and profile/marginalize parity
choice remain; this is not a reconstruction of missing native folding maps.

```python
from pentaceratops import calc_probs_folded_real

result = calc_probs_folded_real(target, prepared_real, eb_eta=0.1)
assert result.metadata["timing_policy"] == "observed"
```

Both the scalar best-fit path and optimized particle evaluations apply the
same x2P overlap rule and ordinary-EB secondary treatment. The timing policy is retained
in result metadata and the scenario table's attributes.

## Full-orbit Fourier inputs

`calc_probs_folded_fourier` already evaluates the complete orbit on native
exposures and applies the data's folding operator to the model. Its default
remains free of an extra local timing cutoff. Missing phase bins remain absent.

The recorded full-P/2P array interface `pentaceratops.calc_probs_fourier` now
also uses complete orbital EB predictions by default. Both eclipses are
evaluated on each fitted half/parity grid. This matters for eccentric systems:
simply removing the old timing bound could otherwise leave an eclipse outside
the half in which the old renderer placed it, losing an observable feature.
The retained FFT bins, PSD conventions, null normalization, scalar sampler,
physical priors and maximum-over-parity choice are preserved.

```python
from pentaceratops import calc_probs_fourier

# Supply the usual full-period arrays and other numerical arguments.
result = calc_probs_fourier(target, **fourier_arguments)
assert result.metadata["timing_policy"] == "observed"
```

Do not include `max_anomaly_shift` in normal `fourier_arguments`. A non-None
value raises an error instead of silently adding a timing prior to an
observed-mode run. Full-orbit observation coverage is not a local event window.
Any additional candidate-association prior would be a separate, explicit
scientific assumption.

## Reproducing archived results

Use explicit reproduction settings when checking adopted paper results:

```python
real = calc_probs_folded_real(target, prepared_real, timing_policy="legacy",
                             eb_eta=0.1, **original_real_settings)
fourier = calc_probs_fourier(target, timing_policy="legacy",
                            **original_fourier_arguments)
```

Here `original_fourier_arguments` includes the saved timing limit, if present.
The checked TESS matched and Kepler completion campaigns both used three
catalogue transit durations. Legacy mode retains the original center-only
x2P guard and Fourier model placement. The Real synthetic-secondary treatment
is retained under both policies.
The dedicated `run_folded_baseline` interface is also explicitly a reproduction
path, preserving its original model grids and timing choices.

Low-level compatibility routines in `pentaceratops.fourier`, the original
residual functions, and experimental adapters retain their previous defaults
for existing research callers. For direct use of `ModelAdapter`,
`OptimizedAdapter` or `PrimaryOnlyAdapter`, pass `timing_policy="observed"` to
select the new behavior. Prefer the recorded public interfaces for new runs.
The production replay tool requests legacy Real behavior explicitly.

## Checks

`tests/test_observed_timing.py` covers x2P edge overlap, finite exposure
integration, asymmetric windows, periodic wrapping, interior gaps, preserved
ordinary-EB constraints, absent secondaries, scalar/optimized equality, and
full-period boundary crossings.
Complete orbital predictions are compared with the independent native-time
renderer. Tests also exercise actual Fourier sampling, saved weighted pools,
policy recording, invalid mixed settings and restoration after exceptions.

The original archived-mode regression and saved-production replay checks
remain separate from these new-behavior tests. Reproducing stored scores is
not a new evidence-convergence calculation.
