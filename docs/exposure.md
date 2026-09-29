# Exposure integration with fixed orbital period

The experimental matched-window adapters accept any positive integer
`nsamples`: the number of midpoint integration samples **within each exposure**.
The default remains **20**. Both scalar and batched models use the same setting,
including secondary eclipses and both parity assignments of x2P models.

These are separate controls:

| Setting | Meaning |
| --- | --- |
| `data["period"]` | Fixed candidate period in days; x2P uses twice this value |
| `data["exptime"]` | Exposure duration in days; one common duration per adapter |
| `nsamples` | Positive integer subsamples per exposure; default 20 |
| `N`, `steps` | Persistent-sampling effort; independent of exposure integration |

`nsamples=1` evaluates the exposure midpoint. Larger values more finely
approximate the finite-exposure average; they do not change the exposure
duration, phase bins, data, period prior, or covariance. A lower count is not
automatically adequate: check convergence for the target's ingress/egress and
exposure duration before reducing it for production evidence calculations.

## Adapter configuration

For already prepared matched-window data:

```python
from pentaceratops.experimental.covariance import ModelAdapter
from pentaceratops.experimental.v2_adapter import OptimizedAdapter

scalar = ModelAdapter(data, nsamples=10)
fast = OptimizedAdapter(data, nsamples=10)
```

The existing scenario helper passes the adapter's setting to the actual
evidence function and its scalar callback:

```python
from pentaceratops.experimental.covariance import scenario_functions, scenario_kwargs

function = scenario_functions()["TP"]
kwargs = scenario_kwargs(
    function, fast, star, data["period"], trilegal_path, molusc_path,
    n=100, steps=20, posterior_n=1000,
)
assert kwargs["nsamples"] == 10

# Optional per-scenario override; fast.nsamples remains 10.
kwargs = scenario_kwargs(
    function, fast, star, data["period"], trilegal_path, molusc_path,
    n=100, steps=20, posterior_n=1000, nsamples=50,
)
```

This is configuration of the existing experimental workflow, not a complete
TIC-to-FPP runner. Its process-local engine/sampler contexts and data/frame/null
bookkeeping are still required. Use one process per target, not concurrent
threads. A bare evidence call does not activate batching.

An explicit evidence-function `nsamples` is honored by `likelihood_batch`,
even if it differs from the adapter default. Direct scalar callbacks and
`photometric_columns(..., nsamples=...)` also accept overrides. In those adapter
methods, an omitted value or `None` inherits the adapter setting. Constructor
values must be positive integers; booleans, floats, strings, zero, and negative
counts are rejected. NumPy integer scalars are accepted.

Low-level evidence functions retain their own historical default of 20.
When choosing a different adapter default, use `scenario_kwargs` or explicitly
pass the resolved count to the evidence function. Keep its `exptime` equal to
`data["exptime"]`; the fast projector uses the prepared-data duration. Save the
resolved count with the run arguments and use it for best-fit replay. Scalar
adapter snapshots include the actual count used.

## Scope and checks

- Fixed period is deliberately retained. The fast model still rejects periods
  other than the prepared P or, for x2P, 2P. No period-range support is added.
- This extends the common planet, secondary-EB, and even/odd model paths used
  by the supported standard scenarios. It does not add a fast Fourier backend,
  legacy primary-only EB backend, or per-point/mixed exposure durations.
- Batching remains experimental and opt-in. The preserved research code and
  historical results are unchanged.
- Synthetic regressions compare scalar and batched log likelihoods at counts
  1, 5, 10, 20, and 50 with 2-minute and 30-minute exposures. They cover GP
  covariance, no-GP operation, unequal even/odd grids, both parity conventions,
  dilution, secondaries inside/outside the window, and fixed-period rejection.
- Additional tests exercise counts 7 and 11, changing settings without stale
  caches, explicit evidence-closure overrides for TP/EB/EBx2P, and exact
  equivalence between omitted and explicit 20-subsample settings.

These are numerical-equivalence tests at a chosen resolution, not a guarantee
that any particular count has converged to the continuous exposure integral or
that low-effort evidence sampling has converged.
