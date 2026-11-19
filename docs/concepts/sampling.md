# Sampling and convergence

The evidence of each scenario is an integral of the likelihood over that
scenario's prior. Pentaceratops estimates it with **persistent sampling**
(Karamanis et al. 2025), a tempered particle method.

## How it works, briefly

The sampler starts with `N` particles drawn from the prior and moves them in
stages towards the posterior. At each stage it raises the weight given to the
likelihood, resamples the particles, and diversifies them with `steps` sweeps
of a Metropolis sampler. Particles from earlier stages stay in a pool and are
reweighted, so their likelihood evaluations are reused.

The run ends when the full likelihood is reached. It returns the evidence
estimate and the weighted pool, which is a sample from the posterior.

### Watch one run

The animation below replays an actual fit: the planet-on-target scenario for
TOI-700.02 with the subtracted likelihood. Press play.

<video class="explainer only-light" controls loop muted playsinline preload="metadata"
       aria-label="Animation of persistent sampling on a real transit: particles concentrate from the prior onto the posterior over twelve stages">
  <source src="../../assets/persistent_sampling_light.webm" type="video/webm">
  <source src="../assets/persistent_sampling_light.webm" type="video/webm">
  Your browser cannot play this video; open the final frame below.
</video>

<video class="explainer only-dark" controls loop muted playsinline preload="metadata"
       aria-label="Animation of persistent sampling on a real transit: particles concentrate from the prior onto the posterior over twelve stages">
  <source src="../../assets/persistent_sampling_dark.webm" type="video/webm">
  <source src="../assets/persistent_sampling_dark.webm" type="video/webm">
  Your browser cannot play this video; open the final frame below.
</video>

On the left, each dot is one particle, placed by its planet radius and impact
parameter; the area of a dot is its current weight. On the right are the folded
transit data with light curves drawn from the weighted pool. Three things to
watch:

- **β** is the weight given to the likelihood. It rises from 0 (the prior) to 1
  (the posterior) in steps the sampler chooses itself, small at first and
  larger later.
- **Raising β reweights every particle kept so far.** Particles whose models
  miss the data shrink away; those that fit grow. The light curves on the right
  tighten around the data at the same time.
- **Each stage then adds new particles** (orange) by resampling from the
  weighted pool and moving them with a Metropolis sampler. Earlier particles
  stay in the pool, which is what "persistent" refers to.

At β = 1 the weighted pool is the posterior, here a planet of about 1.2 Earth
radii, and the normalization of the weights is the evidence.

This run used `N=400` and `steps=20`, more particles than the default, so that
the picture is dense; it took twelve stages and kept 4,800 particles.

??? note "Final frame as an image"
    ![Final frame: the weighted pool concentrated at about 1.2 Earth radii, and model light curves matching the transit](../assets/persistent_sampling_light.png#only-light)
    ![Final frame: the weighted pool concentrated at about 1.2 Earth radii, and model light curves matching the transit](../assets/persistent_sampling_dark.png#only-dark)

Two settings control the effort:

| Setting | Meaning | More of it |
| --- | --- | --- |
| `N` | Particles per stage | Lowers the scatter and bias of the evidence; cost grows in proportion |
| `steps` | Metropolis sweeps per stage | Helps particles spread within a stage; cost grows in proportion |

Both are different from `nsamples`, the number of points used to average a
model over one exposure.

## Default effort

When `N` and `steps` are not given, each scenario uses its own default:

| Scenarios | `N` | `steps` |
| --- | ---: | ---: |
| TP, PTP, DTP, NTP, SEBx2P | 100 | 20 |
| SEB, BEBx2P | 200 | 50 |
| STP, BEB | 500 | 50 |
| All others | 200 | 30 |

An explicit value overrides that setting for every scenario and leaves the
other at its default. `N=300` keeps each scenario's default `steps`;
`N=500, steps=50` gives every scenario the same effort.

The values actually used are recorded in the `sampling_N` and `sampling_steps`
columns of the result table. Print the current defaults with
`python -m pentaceratops sampling-policy`.

## These defaults are not a convergence guarantee

The defaults come from a sweep over two targets and are provisional. How much
effort is needed depends on the data: a strong signal confines the posterior to
a much smaller part of the prior, and the same effort is then not enough.

!!! warning "High signal-to-noise targets need more effort"
    With the defaults, the evidence of deep, well-sampled transits can be
    unconverged, and an FPP near 0 or 1 from such a run should not be trusted
    without a check.

For a large TESS sample we scaled the effort with the catalogue signal-to-noise
of the candidate:

| Catalogue SNR | `N` | `steps` |
| --- | ---: | ---: |
| Below 30 | defaults | defaults |
| 30 to 100 | 500 | 50 |
| 100 and above | 1000 | 100 |

Treat this as a starting point for your own data, not as a rule.

## Checking a result

Reaching the end of the tempering schedule does not show that the estimate is
accurate. Two inexpensive checks:

1. **Repeat with another seed.** Run the same target with a different `seed`
   and compare the FPPs. If the validation decision changes between seeds, the
   effort is too low or the candidate is close to the threshold.
2. **Increase the effort.** Double `N` and see whether `lnBF` of the leading
   scenarios moves by more than their difference.

Look at `lnBF` of the two or three most probable scenarios, not only at the
FPP: an FPP can be stable because one scenario dominates, while the evidences
underneath are still noisy.

## Seeds and reproducibility

`seed` (default 42) fixes the random numbers of the whole call. Each scenario
gets its own seed derived from it, in a fixed order, so fitting a subset of
scenarios does not change the seeds of the ones that remain. The seed used for
each row is recorded in the table's `seed` column.

`posterior_samples` (default 2000) sets how many equal-weight draws are kept
for each scenario. The complete weighted pool is kept regardless.

## Cost and parallelism

Scenarios run one after another within a call. To process many targets, start
one process per target and keep numerical libraries at one thread each (see
[installation](../install.md#storage-and-runtime-settings)). Cost grows with
the amount of data: targets observed in many sectors are by far the most
expensive, particularly with the marginalized likelihood.
