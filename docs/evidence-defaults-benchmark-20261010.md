# Evidence sampling-policy timing: TOI-700.02

Measured 2026-10-10 using the saved TOI-700.02 preparation and target field.
All 33 eligible host/scenario fits were included (15 target/background families
and three hypotheses for each of six resolved neighbors).

| Setting | Complete `evidence()` call | Compilation included in that time |
| --- | ---: | ---: |
| Previous uniform `N=500, steps=50` | 939.012 s | 4.899 s |
| Shared scenario defaults | 259.557 s | 4.969 s |

This run was **3.62 times faster**, a **72.4% reduction**
in wall time. Both measurements used one CPU core, one BLAS thread, the
optimized subtracted likelihood, the same saved photometry and target field,
seed 42, `eb_eta=0.1`, and an existing Numba disk cache in a fresh process.
Scenarios run sequentially. The baseline ran on `nid005907` (job
`59642752`), the policy run on `nid005367` (job `59643601`).
Each full-target timing is one measurement, not a repeated-run estimate.

The new call is:

```python
results = evidence(prepared, target=target, likelihood="real", eb_eta=0.1)
```

To request the previous uniform effort explicitly:

```python
results = evidence(prepared, target=target, likelihood="real", eb_eta=0.1,
                   N=500, steps=50)
```

The timed call includes covariance preparation from the saved candidate bundle,
compilation, and sampling. Scientific imports (6.053 s) and
loading the saved candidate/field (0.183 s)
were measured separately. Photometric preprocessing from cached native data
previously took 32.667 s on this target; that unchanged stage was not rerun.
Adding these measured stages gives approximately **4.97 minutes**
end to end, excluding downloads and fresh catalog/TRILEGAL generation. This is
a stage-sum estimate, not a separate monolithic pipeline timing. Result saving
is outside the timed evidence call.

## Validation and interpretation

- 103 targeted tests passed, covering both folded likelihoods,
  Python and JSON entry points, defaults, independent overrides, invalid input,
  resolved neighbors, and settings retained through save/load.
- All 33 benchmark fits completed using policy `scenario-defaults-2026-09-15`;
  the runner checked the actual N/steps passed for every scenario.
- Explicit N/steps preserve their previous numerical sampling behavior.
  The likelihoods, physical priors, parity convention, and scenario selection
  are unchanged. Lower default effort changes Monte Carlo precision; this
  timing check does not establish convergence or replace paper results.
  The reported FPP was 4.125% with uniform effort and
  3.312% with scenario defaults in this single-seed comparison.
- This is a full-target subtracted-likelihood timing. It does not measure a
  complete marginalized-likelihood run.

The default settings and recorded output fields are described in the
[Python evidence guide](evidence_api.md#arguments).
