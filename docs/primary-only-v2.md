# Primary-only conditional-FGP v2

`pentaceratops.experimental.primary_v2.PrimaryOnlyAdapter` is an explicit mode
for targets with observed even and odd primary windows but no observed secondary
window. The default three-window `OptimizedAdapter` remains unchanged.

The likelihood vector contains the observed even and odd folded flux. Its
covariance is `diag(sigma**2) + factor @ factor.T`, retaining within-panel and
cross-panel conditional FGP uncertainty. The observation error convention
remains one median folded error per panel, as in the existing v2 campaign.
No missing secondary values are added, and no flat-secondary or off-window
secondary penalty is applied. Empty secondary arrays are structural API
placeholders with zero rows in the covariance factor.

Planet and P-period binary models use the observed primary windows. The
existing 2P binary recipe still evaluates alternating eclipse depths and both
parity assignments on the observed even/odd windows. The established profile
parity choice, x2P timing guard, population priors, dilution, physical cuts,
and exposure integration remain in force. This mode does not replace the
separate full observed-flux joint Fourier likelihood.

Direct adapter construction retains its historical timing default. The public
`calc_probs_folded_real(..., primary_only=True)` interface instead derives x2P
support from model contacts and the supplied window/exposure boundaries in
`0.1.0.dev8`. Pass `timing_policy="legacy"` for archive reproduction. Both
settings preserve the primary-only ordinary-EB treatment described above;
see [timing and coverage](timing.md).

`pentaceratops.preprocessing.primary_v2.prepare_primary_v2` refits each sector
using the package FGP and its exact fixed-PSD conditional coefficient posterior.
It protects exactly the raw observations entering the even/odd folding
operators and requires at least 140 training bins per sector. Existing primary
grids and bin cadence are retained when feasible. Occupied bins omitted by the
old equal-parity grid can be restored; genuinely absent bins remain absent.
Windows may shorten only while retaining every observed nominal-eclipse point,
the whole nominal eclipse interval, and at least one duration on either side.
Selection uses ephemerides, times, and errors; no flux or FPP-based selection.

Every prepared sector saves its masks, training values, PSD, coefficient mean,
factor and draws. Folded posterior draws check the analytic covariance. The
adapter rejects nonempty secondary inputs rather than silently discarding them.

```python
from pentaceratops.preprocessing.primary_v2 import prepare_primary_v2
from pentaceratops.experimental.primary_v2 import load_primary_inputs, PrimaryOnlyAdapter

prepared, folded, audit = prepare_primary_v2(data, metadata, sector_output=output)
# Save prepared and folded dictionaries as separate NPZ files.
inputs = load_primary_inputs({"prepared_path": prepared_file, "folded_path": folded_file})
adapter = PrimaryOnlyAdapter(inputs, include_gp=True, parity="profile", nsamples=20)
```

Use the existing scenario kwargs, patched-engine context and batched sampler
with this adapter, in isolated worker processes. All scenarios for one target
must use the same data vector and covariance. Preprocessing is also process
based; threads sharing module-level transit caches are unsupported.

`tests/test_primary_v2.py` checks an independent multivariate Gaussian,
marginal versus conditioned covariance, cross-parity correlation, gapped
folding, exact GP holdout, posterior covariance draws, ignored unobserved
secondary depth, input rejection, and scalar/batched exposure-integrated models
for planet, binary and alternating-binary cases. Seventeen tests passed before
the first six-target campaign; production validation additionally checks every
actual star/scenario prior and likelihood against its scalar callback.
