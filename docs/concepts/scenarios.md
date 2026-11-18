# Scenarios and the FPP

A transit-like dip can be produced by a planet on the target star or by several
other configurations. Pentaceratops fits each configuration to the same data,
computes its Bayesian evidence, and normalizes the evidences into
probabilities.

## The eighteen scenarios

In every scenario the dip is produced by a transiting planet (TP) or an
eclipsing binary (EB). What changes is which star hosts the signal and what
else dilutes it.

| Host of the signal | Planet | EB | EB at twice the period |
| --- | --- | --- | --- |
| Target star | **TP** | EB | EBx2P |
| Target star, diluted by a bound companion | **PTP** | PEB | PEBx2P |
| Bound companion of the target | STP | SEB | SEBx2P |
| Target star, diluted by a background star | **DTP** | DEB | DEBx2P |
| Background star | BTP | BEB | BEBx2P |
| Resolved nearby star | NTP | NEB | NEBx2P |

The three scenarios in bold are the ones in which a planet transits the target
star. Every other scenario counts as a false positive, including a planet on a
companion, background or nearby star.

**Period-doubled variants (`x2P`).** A binary with two similar stars produces
primary and secondary eclipses of nearly equal depth. A transit search can then
report half the true orbital period, and the folded light curve looks like a
single planetary transit. The `x2P` scenarios model a binary at twice the
candidate period, with the two eclipses falling on alternating events.

**Nearby stars.** The first five rows are evaluated once, for the target. The
last row is evaluated separately for every resolved neighbour that is bright
enough in the aperture to produce the observed depth, so a crowded field has
more than eighteen rows in its result table.

## From evidence to probability

For each scenario the sampler estimates the evidence, the likelihood averaged
over that scenario's parameter prior. The result table reports it as `lnBF`,
the log Bayes factor against a flat light curve for the same data. Using the
same reference for every scenario makes the values directly comparable.

Scenario probabilities are the normalized evidences, with one optional weight
on the eclipsing-binary scenarios:

```text
prob(scenario) ∝ exp(lnBF) × (eb_eta if the scenario is an EB, otherwise 1)
```

Three summaries are stored in `results.output.attrs`:

| Attribute | Sum of the probabilities of |
| --- | --- |
| `FPP` | Every scenario except TP, PTP and DTP |
| `FPP_EB` | Every eclipsing-binary scenario, including `x2P` and nearby EBs |
| `NFPP` | The nearby-star scenarios NTP, NEB and NEBx2P |

`FPP` is the standard false-positive probability. `FPP_EB` is useful when the
question is only whether the signal is an eclipsing binary: a candidate can
have a high `FPP` because a planet on a neighbouring star fits well, with a low
`FPP_EB`.

## Demographic odds: `eb_eta`

Planets are more common than close stellar companions at the periods where
transits are found, so a planet scenario and a binary scenario that fit equally
well are not equally probable. `eb_eta` multiplies the weight of every
EB scenario once, before normalization.

- `eb_eta=1` is the function default and applies no extra weighting.
- `eb_eta=0.1` is the value adopted for real candidates in the paper.

Because the weight is applied after the fits, changing it needs no new
sampling:

```python
from pentaceratops import scenario_probabilities

reweighted = scenario_probabilities(results.output, evidence_column="lnBF", eb_eta=1.0)
print(reweighted.attrs["FPP"])
```

Always pass the unweighted `lnBF` column. Applying `eb_eta` to probabilities
that already include it would count the prior twice.

`eb_eta` is a stated assumption about relative occurrence, not a calibrated
probability. For a sample that was built to contain equal numbers of planets
and binaries, such as an injection test, `eb_eta=1` is the consistent choice.

## Fitting a subset of scenarios

`evidence(..., scenarios=("TP", "EB", "BEB"))` fits only the listed scenarios.
The returned FPP is then conditional on the omitted scenarios having zero
probability, and the result is labelled `conditional_subset`. Use a subset for
diagnostics, not for a final FPP.

## Parameters and priors

Each scenario samples its own orbital and physical parameters, for example the
inclination, eccentricity and argument of periastron, the planet radius or the
binary mass ratio, and the properties of any companion or background star. The
orbital period is fixed to the candidate period, or twice that for `x2P`. The
transit time is fixed to the supplied epoch unless
[`free_epoch`](likelihoods.md#transit-time-offset) is set.

The prior distributions follow TRICERATOPS and are listed in the paper's
appendix. Two inputs to the priors come from the stellar field:

- the **background population**, a TRILEGAL simulation at the target's sky
  position, which supplies the stars for the D and B scenarios;
- an optional **companion population** (`molusc_file`), a prepared sample of
  bound companions for this target, for example one conditioned on its Gaia
  astrometry, used in place of the analytic companion prior.
