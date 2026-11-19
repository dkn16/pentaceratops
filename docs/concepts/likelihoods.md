# The two likelihoods

Every scenario predicts a light curve. The likelihood measures how well that
prediction matches the photometry, given a model of the noise. Pentaceratops
offers two noise treatments, selected with the `likelihood` argument of
`evidence`.

| | Subtracted likelihood | Marginalized likelihood |
| --- | --- | --- |
| `likelihood=` | `"real"` | `"fourier"` |
| Stellar variability | Estimated and subtracted; its uncertainty in the folded windows is propagated | Integrated over, using its power spectrum |
| Data used | Folded windows around the primary transit (even and odd) and the secondary phase | Every observed sample, folded at twice the candidate period |
| Secondary eclipse away from phase 0.5 | Only if it falls inside the window | Anywhere in the orbit |
| Cost | Lower | Higher (about five times, for the reference target) |
| Recommended for | Screening and benchmarks | The final FPP |

Both are built from the same preprocessing output, so a candidate prepared once
can be run with either.

## Subtracted likelihood

The preprocessing fits a Fourier Gaussian-process model of the stellar
variability to each sector or quarter, using only data outside the protected
transit windows, and subtracts it. The likelihood then compares each scenario
with three folded windows:

- the **even** transits,
- the **odd** transits,
- a window at the **secondary** phase, half a period after the transit.

The windows extend four transit durations either side of their centre by
default, capped below a quarter of the period so they cannot overlap. The noise
covariance combines the measurement errors with the uncertainty of the
subtracted trend, including its correlation between the three windows.

![The even, odd and secondary windows of TOI-6612.01 after the variability is removed: a 1 per cent transit in the even and odd windows and a flat secondary window](../assets/likelihood_input_subtracted_light.svg#only-light)
![The even, odd and secondary windows of TOI-6612.01 after the variability is removed: a 1 per cent transit in the even and odd windows and a flat secondary window](../assets/likelihood_input_subtracted_dark.svg#only-dark)

*What the subtracted likelihood is given, for TOI-6612.01 (a 1 per cent
transit every 2.32 days). Each window holds 29 half-hour bins spanning 7 hours
either side of its centre. The error bars are the measurement errors, 200 to
300 ppm; the uncertainty of the subtracted trend is carried separately in the
covariance.*

Keeping even and odd transits apart is what lets the period-doubled binary
scenarios be tested: a difference between the two folds favours a binary at
twice the period, and agreement constrains it. The secondary window does the
same for an ordinary eclipsing binary, whose predicted secondary eclipse must
be consistent with the data there.

If an eccentric binary's predicted secondary falls outside the window, the
model is penalised as if that eclipse had been looked for and not found. This
assumes there is no unmodelled eclipse elsewhere in the orbit, which the
windowed data cannot check.

## Marginalized likelihood

The stellar variability is not subtracted. It is treated as a Gaussian process
whose covariance is given by the power spectral density estimated during
preprocessing, and the likelihood integrates over it. The data are all observed
samples, folded at twice the candidate period so that alternating eclipses stay
distinct. Gaps stay gaps: the covariance is propagated through the actual
observation mask and folding, and nothing is interpolated.

![Top: all 224 bins of TOI-6612.01 folded at twice the period, showing the even and odd transits and slow stellar variability. Bottom: the correlation of the noise between two bins against their separation, falling from 1 to zero over about ten hours](../assets/likelihood_input_marginalized_light.svg#only-light)
![Top: all 224 bins of TOI-6612.01 folded at twice the period, showing the even and odd transits and slow stellar variability. Bottom: the correlation of the noise between two bins against their separation, falling from 1 to zero over about ten hours](../assets/likelihood_input_marginalized_dark.svg#only-dark)

*What the marginalized likelihood is given, for the same candidate. Top: all
224 half-hour bins of the fold at twice the period. The stellar variability is
still in the data, as the slow drift between the transits, and the error bars,
about 600 ppm, include it. Bottom: the noise model treats that variability as
correlated noise. Neighbouring bins are correlated at 0.68, and the correlation
dies away over about ten hours.*

Because the whole orbit is fitted, there is no window to choose and a secondary
eclipse can lie at any phase. The price is a dense covariance matrix, which
makes this likelihood slower, especially for targets with many sectors.

The power spectrum is held fixed at its estimate; its own uncertainty is not
marginalized.

## Which one to use

Run the subtracted likelihood first when exploring a sample: it is cheaper and
its result is often close. Use the marginalized likelihood for the number you
report, particularly for long-period, shallow transits, where the uncertainty
of the removed variability matters most.

If the two disagree strongly, look at the scenario that changed. A common
pattern is weak evidence for the event itself, which shows up as probability
moving between host stars, not as support for a specific false positive.

## Settings shared by both

### Exposure integration

Models are averaged over each exposure using `nsamples` points. The default is
20 for the subtracted likelihood and 7 for the marginalized one. This is
separate from the sampler's particle count `N`. See
[exposure integration](../exposure.md).

### Transit-time offset

By default the transit time is fixed to the supplied epoch. With
`free_epoch=True`, every scenario samples one extra parameter: the transit time
minus the supplied epoch, with a zero-mean Gaussian prior of width
`epoch_sigma` in days (3 minutes if not given). All eclipses of a scenario move
together.

This matters at high signal-to-noise. Catalogue epochs can be off by about a
minute, which is larger than the timing precision of a deep, well-sampled
transit. With a fixed epoch the fit absorbs the error through eccentricity,
which biases the fitted eccentricity and argument of periastron and changes the
balance between planet and binary scenarios. Set `epoch_sigma` to the epoch
uncertainty at the time of the observations, for example the catalogue epoch
and period errors propagated to the middle of the data.

### Parity

For the period-doubled scenarios the model does not know which of the two
alternating events is the primary eclipse. `parity="profile"` (default) takes
the better assignment at each parameter draw; `parity="marginalize"` averages
the two.

## Settings of the subtracted likelihood only

| Option | Default | Meaning |
| --- | --- | --- |
| `primary_only` | `False` | Use only the even and odd windows, for data with no coverage of the secondary phase. Needs a matching preparation; see [primary-only subtracted likelihood](../primary-only-v2.md). |
| `include_gp` | `True` | Include the uncertainty of the subtracted trend in the covariance. |
| `timing_policy` | `"observed"` | How the period-doubled models are checked against the window coverage. `"legacy"` restores an older rule; see [eclipse timing and coverage](../timing.md). |

## Further detail

- [Likelihood-ready folded inputs](../hz_runs.md): building the folded data
  products directly.
- [Joint Fourier likelihood with gaps](../joint_fourier.md): the covariance and
  observation operator behind the marginalized likelihood.
- [Folded Fourier weighting](../fourier_weighting.md): how parity folds are
  combined for scenarios that repeat every period.
