# Pentaceratops

Pentaceratops estimates the **false-positive probability (FPP)** of a transiting
planet candidate. It takes the candidate's light curve and the stars in and
around the photometric aperture, fits eighteen competing astrophysical
scenarios, and reports how probable each one is.

[![Method overview. Top: the pipeline from the folded photometry, ephemeris and stellar scene, through the scenario bank, the prior and likelihood and the persistent sampler, to scenario scores, demographic weights and the validation outputs. Bottom: a folded candidate, the scenario families, the eclipse windows, the marginalized likelihood, persistent sampling, and radius posteriors.](assets/method_overview.svg){ .overview-figure }](assets/method_overview.svg)

*How a candidate becomes an FPP: the pipeline (top) and what each stage works
with (bottom). Click the figure to enlarge it.*

It follows the scenario framework of
[TRICERATOPS](https://github.com/stevengiacalone/triceratops)
(Giacalone et al. 2021) and changes three parts of the calculation:

- **The likelihood uses more of the light curve, and the priors are more
  informative.** Eclipsing-binary scenarios are tested against the secondary
  eclipse and against the difference between even and odd transits, not only the
  primary transit. The priors can include a companion population conditioned on
  the target's Gaia astrometry, and demographic odds between planets and
  eclipsing binaries.
- **Stellar variability can be marginalized.** In addition to the usual
  likelihood on a detrended light curve, a second likelihood integrates over a
  Gaussian model of the variability described by its power spectrum.
- **Evidence comes from persistent sampling.** Each scenario's Bayesian evidence
  is estimated with a tempered particle sampler, which also returns posterior
  samples of the physical parameters. [Watch one run](concepts/sampling.md#watch-one-run).

!!! warning "Development version"
    The current version is `0.1.0.dev14`. Interfaces can still change, the
    default sampling settings are provisional, and the [accompanying
    paper](#citing) is in preparation. Check the [sampling notes](concepts/sampling.md) before
    relying on an individual FPP.

## A complete run in three calls

```python
from pentaceratops import prepare_target
from pentaceratops.preprocessing.candidate import prepare_candidate
from pentaceratops.evidence import evidence

# 1. Download, detrend and fold the light curve.
prepared, prepared_path = prepare_candidate("TOI-700.02", cache_dir="/path/to/cache")

# 2. Build the stellar field: nearby stars, aperture dilution, background population.
target = prepare_target(prepared, output_dir="/path/to/new_field")

# 3. Fit every scenario and combine them into an FPP.
results = evidence(prepared, target=target, likelihood="marginalized", eb_eta=0.1)

print(results.output[["ID", "scenario", "lnBF", "prob"]])
print("FPP:", results.output.attrs["FPP"])
```

The [quick start](quickstart.md) walks through these calls, what they need, and
how to read the result.

## What you need

| Input | Where it comes from |
| --- | --- |
| Light curve | Downloaded for a TOI or KOI by `prepare_candidate`, or [your own arrays](guide/supplied_lightcurve.md) |
| Ephemeris (period, epoch, duration) | NASA Exoplanet Archive by default, or values you supply |
| Stellar field and aperture dilution | `prepare_target` for TESS, or a [table you supply](target_setup.md) |
| Background-star population | A TRILEGAL query made by `prepare_target`, or an existing file |

Building a new stellar field with `prepare_target` currently supports **TESS**.
Kepler candidates run through the same evidence calculation with a supplied
stellar table and population.

## Where to go next

<div class="grid cards" markdown>

- **[Installation](install.md)**

    Requirements, optional extras and a dependency check.

- **[Quick start](quickstart.md)**

    One TESS candidate from identifier to FPP.

- **[Scenarios and the FPP](concepts/scenarios.md)**

    The eighteen scenarios and how they combine into a probability.

- **[The two likelihoods](concepts/likelihoods.md)**

    Subtracted and marginalized likelihoods, and when to use each.

- **[Options](reference/options.md)**

    Every setting of the evidence calculation in one table.

- **[Use your own light curve](guide/supplied_lightcurve.md)**

    Observed or injected photometry that is already on disk.

</div>

## Citing

If you use Pentaceratops, please cite the method paper, which is in preparation:

- Diao, K., Robnik, J., Adolff, J., Karamanis, M., & Seljak, U., in preparation,
  *Statistically Validating Habitable Transiting Planets with pentaceratops:
  Improved Posterior Modeling and Implications*

```bibtex
@unpublished{pentaceratops,
  author = {Diao, Kangning and Robnik, Jakob and Adolff, Julius and
            Karamanis, Minas and Seljak, Uro{\v{s}}},
  title  = {Statistically Validating Habitable Transiting Planets with
            \texttt{pentaceratops}: Improved Posterior Modeling and Implications},
  note   = {in preparation},
  year   = {2026},
  url    = {https://github.com/dkn16/pentaceratops}
}
```

This entry will be replaced by the journal reference once the paper is out.
The scenario framework comes from TRICERATOPS, which should be cited as well:

- Giacalone, S., Dressing, C. D., Jensen, E. L. N., et al. 2021, AJ, 161, 24
