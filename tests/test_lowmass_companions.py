"""Low-mass companion extension and Moe & Di Stefano binary eccentricity prior."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pentaceratops import priors, stellar

needs_new = pytest.mark.skipif(not (priors.LOWMASS_ENABLED and stellar.LOWMASS_ENABLED and priors.BINARY_ECC == "moe2017"),
                               reason="low-mass extension / moe2017 eccentricity switched off by environment")


def _legacy_values():
    """Run the priors with both switches off in a fresh interpreter."""
    code = (
        "import json, numpy as np\n"
        "from pentaceratops import priors, stellar\n"
        "u = np.linspace(0.001, 0.999, 41)\n"
        "out = dict(q=[priors.sample_q(u.copy(), m).tolist() for m in (0.2, 0.6, 1.3)],\n"
        "           qc=[priors.sample_q_companion(u.copy(), m).tolist() for m in (0.2, 0.6, 1.3)],\n"
        "           e=[priors.binary_eccentricity_ppf(u, p).tolist() for p in (3., 30.)],\n"
        "           r=stellar.stellar_relations(np.array([0.05, 0.09, 0.2]), np.full(3, 9.), np.full(3, 9e9))[0].tolist(),\n"
        "           f=np.atleast_1d(stellar.flux_relation(np.array([0.05, 0.2]))).tolist())\n"
        "print(json.dumps(out))\n")
    env = dict(os.environ, PENTACERATOPS_LOWMASS="0", PENTACERATOPS_BINARY_ECC="legacy")
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_legacy_switches_reproduce_previous_priors():
    out = _legacy_values()
    u = np.linspace(0.001, 0.999, 41)
    for m, q in zip((0.2, 0.6, 1.3), out["q"]):
        np.testing.assert_allclose(q, priors._sample_q_stellar(u.copy(), m), rtol=0, atol=0)
    for m, q in zip((0.2, 0.6, 1.3), out["qc"]):
        np.testing.assert_allclose(q, priors._sample_q_companion_stellar(u.copy(), m), rtol=0, atol=0)
    for p, e in zip((3., 30.), out["e"]):
        np.testing.assert_allclose(e, u**(1 / (0.2 if p <= 10 else 0.6)), rtol=1e-12)
    assert out["r"][0] == 0.1                       # old clamp below 0.1 Msun
    assert out["f"][0] == pytest.approx(10**stellar.flux_spline(0.05))


@needs_new
@pytest.mark.parametrize("M_s", [0.2, 0.6, 1.0, 1.6])
@pytest.mark.parametrize("fn", [priors.sample_q, priors.sample_q_companion])
def test_lowmass_mixture_is_monotone_continuous_and_reaches_13_mjup(M_s, fn):
    u = np.linspace(0, 1 - 1e-12, 200001)
    q = fn(u.copy(), M_s)
    assert np.all(np.diff(q) >= -1e-12), "inverse CDF must be monotone"
    assert q[0] == pytest.approx(priors.M_BD_MIN / M_s, rel=1e-6)
    assert q[-1] <= 1.0 + 1e-12
    # continuity of the quantile function (no gaps) across the segment boundaries
    assert np.max(np.diff(q)) < 0.01
    # stellar part unchanged in shape: conditional on M > floor it matches the old prior
    floor = priors._sample_q_stellar(np.array([0.]), M_s)[0] if fn is priors.sample_q \
        else priors._sample_q_companion_stellar(np.array([0.]), M_s)[0]
    old = (priors._sample_q_stellar if fn is priors.sample_q else priors._sample_q_companion_stellar)(
        np.linspace(0, 1 - 1e-12, 2001), M_s)
    new = q[q >= floor]
    np.testing.assert_allclose(np.quantile(new, np.linspace(0, 1, 2001)), old, atol=2e-3)


@needs_new
def test_brown_dwarf_weight_and_density_continuity_at_floor():
    M_s = 1.0
    seg = priors._lowmass_segments(M_s, priors._sample_q_stellar, 0.3)
    assert seg["m_bd"] == pytest.approx(priors.F_BD_REL)
    u = np.random.default_rng(0).random(2_000_000)
    q = priors.sample_q(u, M_s)
    frac_bd = np.mean(q * M_s < priors.M_HBURN)
    assert frac_bd == pytest.approx(seg["m_bd"] / seg["total"], rel=0.02)
    # density just below and just above the old floor agree (VLM continues the power law)
    f = seg["q_floor"]
    below = np.mean((q > f * 0.97) & (q < f)) / (0.03 * f)
    above = np.mean((q > f) & (q < f * 1.03)) / (0.03 * f)
    assert below == pytest.approx(above, rel=0.05)


@needs_new
def test_lowmass_relations_and_flux():
    m = np.array([0.0124, 0.03, 0.06, 0.075, 0.09, 0.0999, 0.1, 0.12])
    r, t = stellar.stellar_relations(m, np.full(len(m), 9.), np.full(len(m), 9e9))
    assert np.all((r[:4] > 0.078) & (r[:4] < 0.105))        # brown dwarfs ~0.8-1.0 R_J
    assert t[0] < 400 and t[3] > 1800 and t[5] > 2700
    assert r[5] == pytest.approx(0.124, abs=0.002) and r[6] == pytest.approx(0.12)   # ~3% join at 0.1
    f = stellar.flux_relation(m)
    assert f[6] == pytest.approx(10**stellar.flux_spline(0.1))
    assert f[5] == pytest.approx(f[6], rel=0.05)                                   # continuous at 0.1
    assert np.all(np.diff(f) > 0) and f[0] < 1e-8                                  # BDs dark in TESS band


@needs_new
def test_moe2017_binary_eccentricity():
    u = np.linspace(1e-6, 1 - 1e-6, 100001)
    assert np.all(priors.binary_eccentricity_ppf(u, 1.5) == 0)      # P <= 2 d circular
    assert np.all(priors.binary_eccentricity_ppf(u, 8.0) == 0)      # eta <= -1 -> circular
    e30 = priors.binary_eccentricity_ppf(u, 30.)
    emax = 1 - (30 / 2)**(-2 / 3)
    assert e30.max() <= emax + 1e-12 and np.median(e30) > 0.2
    # logpdf integrates to 1 on its support
    grid = np.linspace(1e-6, emax, 200001)
    pdf = np.exp(priors.binary_eccentricity_logpdf(grid, 30.))
    assert np.trapezoid(pdf, grid) == pytest.approx(1, rel=2e-3)
