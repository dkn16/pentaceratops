# Consistent folded Fourier weighting

`pentaceratops.calc_probs_fourier(...)` defaults to `weighting="consistent"`
and `backend="optimized"`. All target, companion, background and nearby-host
scenarios use the same aperture-frame data covariance and flat reference.
Dilution is applied to the physical models.

When full-period even/odd folds are supplied, they and their supplied PSDs
define the noise model for every scenario. They already include the nominal
secondary-phase observations. There is no extra flat-secondary likelihood.
For independent parity noise, at each retained Fourier mode define

\[
V_c=(V_e^{-1}+V_o^{-1})^{-1},\qquad
y_c=V_c(y_e/V_e+y_o/V_o).
\]

A P-period TP or EB predicts the same model in both parities. Its likelihood
ratio against the flat model is exactly the same on `(y_c,V_c)` as on both
parity folds. The parity-difference factor cancels against that reference.
The implementation uses the equivalent sum of precision operators, retaining
the common full-data normalization for reported `lnZ`. An EBx2P can predict
different parity profiles and is scored on both folds. The covariance is not
re-estimated separately for different scenario families.

The FFT retains the existing omission of DC and the real Nyquist mode.
Phase halves are not treated as independent Fourier datasets, and no
half-period PSD approximation is introduced. With no parity data, all available
P-period scenarios use the supplied full-period flux and PSD; x2P inference
requires parity data. `ntransits_secondary` is retained for API compatibility
and is used only in archive weighting.

This array interface assumes a complete uniform period and independent
diagonal-Fourier parity noise. For gaps, variable bin coverage, or cross-parity
covariance, use the prepared-data interface:

```python
from pentaceratops.evidence import evidence

results = evidence(prepared, target=target, likelihood="fourier", eb_eta=0.1)
```

That interface already uses the same full-orbit data, model folding operator
and propagated covariance `sum_s A_s C_s A_s.T` for every scenario. It does not
switch to the uniform approximation. See [preparing these inputs](evidence_api.md).

To reproduce an archived half-period calculation, select both conventions:

```python
from pentaceratops import calc_probs_fourier

archived = calc_probs_fourier(
    target, **original_fourier_arguments,
    weighting="legacy", timing_policy="legacy",
)
```

Timing and covariance settings are independent. A controlled covariance rerun
can retain `timing_policy="legacy"` and its saved `max_anomaly_shift` while
using `weighting="consistent"`. Setting `backend="scalar"` provides the
reference likelihood with the same corrected weights. Existing result files
and published FPPs are never silently relabelled or recomputed.
