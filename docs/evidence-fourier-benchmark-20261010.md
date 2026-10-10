# Full Fourier evidence timing: TOI-700.02

Measured 2026-10-10 with the saved TOI-700.02 photometry and stellar field,
using the shared scenario sampling defaults:

```python
results = evidence(prepared, target=target, likelihood="fourier", eb_eta=0.1)
```

All **33 scenarios** completed (15 target/background families plus three
hypotheses for each of six resolved neighbors). This is the entire call.

| Stage | Seconds |
| --- | ---: |
| Scientific imports | 21.041 |
| Load saved preparation and target field | 0.195 |
| Fourier covariance construction | 145.310 |
| All 33 scenario calls, including compilation | 1172.822 |
| Other work inside `evidence()` | 0.350 |
| **Complete `evidence()` call** | **1318.483** |

The full call took **21.97 minutes**, compared with
**4.33 minutes** for the subtracted likelihood
on the same input target and with the same scenario sampling settings.
Compilation took 5.793 s and is already included in the table;
it must not be added again.

Preprocessing the cached native photometry previously took 32.667 s and is
shared between the likelihoods. Adding that measured stage, imports, loading,
and this evidence call gives approximately **22.87 minutes end
to end**, excluding downloads and fresh catalog/TRILEGAL generation. This
stage-sum estimate is not a separately timed monolithic pipeline. Result saving
(1.058 s) is outside the evidence timer.

## Scenario timings

The [complete scenario timing table](evidence-fourier-scenario-times-20261010.csv) includes both likelihoods,
with each host, scenario, N and steps. The slowest Fourier fits were:

| Host | Scenario | Seconds |
| --- | --- | ---: |
| 150428135 | BEB | 144.638 |
| 150428135 | BEBx2P | 126.791 |
| 150428138 | NEBx2P | 71.225 |
| 736940565 | NEB | 66.074 |
| 736940558 | NEB | 64.315 |
| 736940566 | NEB | 63.146 |
| 150428129 | NEB | 61.086 |
| 736940565 | NEBx2P | 60.729 |

## Measurement conditions

- One AMD EPYC 7763 CPU core, one BLAS thread, sequential scenarios, optimized
  batched backend. Slurm recorded a peak resident memory of approximately 5 GiB.
- Node `nid004228`, Slurm CPU debug job `59643798`.
- Same saved photometry, target field, seed 42, 2000 posterior draws, eta=0.1,
  and source fingerprint as the [subtracted-likelihood timing](evidence-defaults-benchmark-20261010.md).
- Fresh Python process with an existing Numba disk cache, matching the earlier
  benchmark. Exposure subsamples use their likelihood defaults (Fourier 7,
  real 20).
- Fourier uses 10,780 full-orbit folded bins formed from
  112,626 native exposures. The real likelihood uses 489
  bins in the even, odd and secondary panels. This comparison therefore measures
  the complete default workflows, including their different input sizes.
- Every nonempty fit used batched prior and likelihood callbacks.
- Every scenario's actual N/steps was checked against the shared policy and
  against the corresponding real benchmark. Both runs have the same 33
  host/scenario pairs and source fingerprint `25742f719a9b070e1706060ec35cb09a7f8f942f0be0eef2e87c0f75f60c9f51`.
- One timing measurement per full-target call. These timings do not establish
  evidence convergence or replace any paper result.

Reusing a ready `FoldedFourierData` object skips covariance construction; the
call above starts from the saved candidate preparation and includes it.
