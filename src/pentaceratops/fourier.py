"""Full-period Fourier FPP with a shared covariance for every scenario.

When parity folds are supplied, their full-period data and PSDs define the
likelihood for every host and scenario. P-period models use the exactly
equivalent inverse-variance combined statistic; x2P models retain both parities.
With no parity data, P-period models use the supplied full-period covariance.
The same retained Fourier modes and aperture-frame flat reference are used
throughout. A separate flat-secondary likelihood is not added.

weighting="legacy" explicitly reproduces the former independent half-period
PSD approximation and scenario-dependent fold weights. It is for archived
results only. timing_policy is independent of the covariance choice.
"""

import numpy as np
import pandas as pd

from .evidence import fourier as _MF
from .evidence import fourier_eclipses as _MEF


# ---------------------------------------------------------------------------
# data-prep helpers
# ---------------------------------------------------------------------------
def _fft_complex_bins(x):
    """DC- and Nyquist-dropped positive-frequency coefficients."""
    x = np.asarray(x)
    coeff = np.fft.rfft(x)[1:]
    if x.size % 2 == 0:
        coeff = coeff[:-1]
    return coeff


def renorm_fourier(flux_full, psd_folded, star_fluxratio):
    """Renormalise the full-period flux + PSD into one star's diluted frame.
    `renorm_flux` scales flux fluctuations by 1/fr, so the noise POWER scales by 1/fr**2."""
    fr = float(star_fluxratio)
    flux_s = (flux_full - (1.0 - fr)) / fr
    psd_s = psd_folded / fr**2
    return flux_s, psd_s


def build_fourier_windows(flux_full, psd_folded, phase, dt,
                          ntransits, ntransits_secondary, P_orb=None):
    """Prepare the half-split windows for one (renorm'd) flux_full.

    Returns a dict: t_pri, t_sec, prim, sec, var_pri, var_sec (the two P/2 halves,
    eclipse-centred, with per-half variance psd_folded[1::2]/2 / n_half), plus the
    full-period time_full / var_full (TP single-window option, unused by the half-split
    path). The grids carry the EXACT fractional eclipse positions: rolling snaps the
    primary to the nearest sample (offset i_pri*dt - phase, up to half a cadence) and,
    for odd N, the secondary window centre sits M*dt (not P/2) after the primary -- both
    offsets are folded into t_pri / t_sec so the model is evaluated where the data
    eclipses actually are (needs P_orb for the secondary grid)."""
    flux_full = np.asarray(flux_full, float)
    N = len(flux_full); M = N // 2
    i_pri = int(round(phase / dt))
    time_full = np.arange(N) * dt - phase
    rolled = np.roll(flux_full, M // 2 - i_pri)        # primary -> centre of first half
    prim = rolled[:M]
    sec = rolled[M:2 * M]                               # nominal secondary -> centre of 2nd half
    delta_pri = i_pri * dt - phase                      # grid-snap offset (<= dt/2)
    t_pri = (np.arange(M) - M // 2) * dt + delta_pri
    t_sec = t_pri + (M * dt - P_orb / 2.0) if P_orb is not None else t_pri.copy()
    psd_half = np.asarray(psd_folded, float)[1::2] / 2.0
    return dict(
        time_full=time_full, flux_full=flux_full,
        var_full=np.asarray(psd_folded, float) / ntransits,
        t_pri=t_pri, t_sec=t_sec, t_half=t_pri, prim=prim, sec=sec,
        var_pri=psd_half / ntransits, var_sec=psd_half / ntransits_secondary,
    )


def null_window_lnL(data, var):
    """Flat-model (null) Fourier log-evidence of one window: norm - data power."""
    data = np.asarray(data)
    ft = _fft_complex_bins(data)
    var = np.asarray(var, float)
    expected = len(np.fft.rfft(data)[1:])
    if len(var) != expected:
        raise ValueError(
            f"variance length {len(var)} does not match DC-dropped rfft length {expected}"
        )
    if data.size % 2 == 0:
        var = var[:-1]
    return float(-np.sum(np.log(np.pi * var)) - np.sum(np.abs(ft)**2 / var))


# backward-compatible alias (TP scenarios' null-secondary term)
null_secondary_lnL = null_window_lnL


def build_evenodd_windows(flux_eo, psd_folded, phase, dt, n_even, n_odd,
                          psd_even=None, psd_odd=None):
    """Prepare the x2P even/odd FULL-P folds for one (renorm'd) flux_even_odd_full.
    flux_eo = [even fold | odd fold], each length N; each scored against psd_folded
    (length N//2) with its OWN ntransits. Aligned grid = arange(N)*dt - phase."""
    flux_eo = np.asarray(flux_eo, float)
    N = len(flux_eo) // 2
    even, odd = flux_eo[:N], flux_eo[N:]
    t = np.arange(N) * dt - phase
    psd_default = np.asarray(psd_folded, float)
    psd_even = psd_default if psd_even is None else np.asarray(psd_even, float)
    psd_odd = psd_default if psd_odd is None else np.asarray(psd_odd, float)
    return dict(t_eo=t, even=even, odd=odd,
                var_even=psd_even / n_even, var_odd=psd_odd / n_odd)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def calc_probs_fourier(target, P_orb, flux_full, psd_folded, phase, dt,
                       ntransits, ntransits_secondary,
                       flux_even_odd_full=None, ntransits_even=None, ntransits_odd=None,
                       contrast_curve_file=None, filt="Kepler", molusc_file=None,
                       trilegal_fname=None,
                       N=50, steps=20, nsamples=7, flatpriors=False,
                       eb_eta=1.0, drop_scenario=(), verbose=1,
                       max_anomaly_shift=None, psd_even=None, psd_odd=None,
                       weighting="consistent", backend="optimized", timing_policy="legacy"):
    """Compute full-period Fourier-domain scenario evidences and FPP.

    Args mirror calc_probs where shared. `eb_eta` down-weights the EB-family occurrence
    prior (demographics; 1.0 = off). Returns a pandas.DataFrame (`.FPP`, `.NFPP` attrs).
    With weighting="consistent" (default), parity fluxes and their PSDs define
    the shared covariance; flux_full/psd_folded supply it only without parities.
    ntransits_secondary is used only by weighting="legacy". The optimized and
    scalar backends evaluate the same likelihood. This low-level entry point
    retains legacy model timing unless timing_policy="observed" is requested;
    the recorded package API defaults to observed timing.
    """
    if weighting not in ("consistent", "legacy"):
        raise ValueError("weighting must be consistent or legacy")
    if backend not in ("optimized", "scalar") or timing_policy not in ("observed", "legacy"):
        raise ValueError("Unknown Fourier backend or timing policy")
    stars = target.stars[target.stars["tdepth"] > 0].reset_index(drop=True)
    if len(stars) == 0:
        raise ValueError("no stars with tdepth > 0 -- run target.calc_depths first")
    mission = getattr(target, "mission", "Kepler")
    if trilegal_fname is None:
        trilegal_fname = getattr(target, "trilegal_fname", None)
    if max_anomaly_shift is not None:
        max_anomaly_shift = float(max_anomaly_shift)
        if not np.isfinite(max_anomaly_shift) or max_anomaly_shift <= 0.0:
            raise ValueError("max_anomaly_shift must be a finite positive duration in days")
    sig = 1e-3  # placeholder scalar (the secondary Fourier likelihood ignores it)

    flux_full = np.asarray(flux_full, float)
    if flux_full.ndim != 1:
        raise ValueError(f"flux_full must be a 1-D updated-format array; got {flux_full.shape}")
    if flux_even_odd_full is not None:
        flux_even_odd_full = np.asarray(flux_even_odd_full, float)
        if flux_even_odd_full.ndim != 1:
            raise ValueError(
                "flux_even_odd_full must be a 1-D updated-format array; "
                f"got {flux_even_odd_full.shape}")
        # Some npz products give the two folds lengths differing by one sample
        # (len = 2N+1). Normalise to exactly [even(N) | odd(N)], choosing which fold
        # carries the extra (dropped) sample by the recombination residual vs flux_full.
        N_f = len(flux_full)
        if len(flux_even_odd_full) == 2 * N_f + 1:
            a = flux_even_odd_full
            cands = [np.concatenate([a[:N_f], a[N_f:2 * N_f]]),          # extra sample ends the odd fold
                     np.concatenate([a[:N_f], a[N_f + 1:2 * N_f + 1]])]  # extra sample ends the even fold
            if ntransits_even is not None and ntransits_odd is not None:
                def _dev(c):
                    rec = (ntransits_even * c[:N_f] + ntransits_odd * c[N_f:]) \
                          / (ntransits_even + ntransits_odd)
                    return float(np.max(np.abs(rec - flux_full)))
                devs = [_dev(c) for c in cands]
                pick = int(np.argmin(devs))
            else:
                devs = [np.nan, np.nan]; pick = 0
            flux_even_odd_full = cands[pick]
            if verbose:
                print(f"  NOTE: even/odd folds differ by one sample (len=2N+1); split #{pick} "
                      f"chosen (recombination dev {devs[pick]:.2e})", flush=True)
        if ntransits_even is not None and ntransits_odd is not None:
            # For NOMINAL integer counts the per-fold ntransits should partition the transits
            # (n_even+n_odd=n_full). With EFFECTIVE (mean-R-recalibrated, non-integer) counts
            # this no longer holds, so only warn rather than hard-assert.
            if abs((ntransits_even + ntransits_odd) - ntransits) > 1e-6 and verbose:
                print(f"  NOTE: ntransits_even+odd ({ntransits_even+ntransits_odd:.3f}) != "
                      f"ntransits ({ntransits}) -- effective/recalibrated counts in use", flush=True)
            # x2P/P comparability requires the folds to recombine to flux_full:
            # n_e*even + n_o*odd == n_t*flux_full. Inconsistent folds make the same signal
            # worth systematically different evidence in the two representations.
            L_eo = len(flux_even_odd_full) // 2
            if L_eo == len(flux_full):
                recon = (ntransits_even * flux_even_odd_full[:L_eo]
                         + ntransits_odd * flux_even_odd_full[L_eo:]) / (ntransits_even + ntransits_odd)
                dev = float(np.max(np.abs(recon - flux_full)))
                dip = float(1.0 - np.min(flux_full))
                if dev > 0.25 * dip and verbose:
                    print(f"  WARNING: even/odd folds do not recombine to flux_full "
                          f"(max dev {dev:.2e} vs transit dip {dip:.2e}) -- x2P vs P-period "
                          f"evidences carry a data-inconsistency systematic", flush=True)

    adapter = None
    if weighting == "consistent":
        from .likelihoods.folded_fourier import prepare_folded_metric
        from .experimental.uniform_fourier import UniformFourierAdapter
        metric = prepare_folded_metric(flux_full, psd_folded, phase, dt, P_orb, ntransits,
            parity_flux=flux_even_odd_full, n_even=ntransits_even, n_odd=ntransits_odd,
            psd_even=psd_even, psd_odd=psd_odd)
        adapter = UniformFourierAdapter(metric, P_orb, phase, nsamples=nsamples,
            mission=mission, filt=filt, timing_policy=timing_policy,
            max_shift=max_anomaly_shift, backend=backend)

    def _call(function, *args, **kwargs):
        if adapter is not None:
            return adapter.call(function, *args, **kwargs)
        if timing_policy == "observed":
            from .likelihoods.observed_fourier import observed_fourier_engine
            with observed_fourier_engine():
                return function(*args, **kwargs)
        return function(*args, **kwargs)

    rows = []
    for i in range(len(stars)):
        ID = stars["ID"].values[i]
        fr = float(stars["fluxratio"].values[i])
        if not np.isfinite(fr) or not 0 < fr <= 1:
            raise ValueError("Eligible host fluxratio must be in (0, 1]")
        if adapter is not None:
            adapter.aperture_fraction = fr
        M_s = float(stars["mass"].values[i])
        R_s = float(stars["rad"].values[i])
        Teff = float(stars["Teff"].values[i])
        Z = 0.0
        flux_s, psd_s = renorm_fourier(flux_full, psd_folded, fr)
        w = build_fourier_windows(flux_s, psd_s, phase, dt, ntransits, ntransits_secondary,
                                  P_orb=P_orb)
        n_rows_star = len(rows)   # rows appended for THIS star get its null (see loop end)

        # Consistent inference uses the same aperture-frame flat reference for
        # every host and scenario. Only explicit archive weighting evaluates
        # the former host-frame half-period reference and representation shift.
        null_hs = (null_window_lnL(w["prim"], w["var_pri"])
                   + null_window_lnL(w["sec"], w["var_sec"])) if adapter is None else metric.null_loglike
        secondary_null = null_secondary_lnL(w["sec"], w["var_sec"]) if adapter is None else 0.

        # Recipe inputs remain available for priors/output compatibility.
        # The consistent adapter scores its shared metric; the archive branch
        # alone applies a representation-dependent null shift.
        do_x2p = (flux_even_odd_full is not None
                  and ntransits_even is not None and ntransits_odd is not None)
        if do_x2p:
            flux_eo_s = (np.asarray(flux_even_odd_full, float) - (1.0 - fr)) / fr
            psd_even_s = psd_s if psd_even is None else np.asarray(psd_even, float) / fr**2
            psd_odd_s = psd_s if psd_odd is None else np.asarray(psd_odd, float) / fr**2
            weo = build_evenodd_windows(
                flux_eo_s, psd_s, phase, dt, ntransits_even, ntransits_odd,
                psd_even=psd_even_s, psd_odd=psd_odd_s)
            null_eo = (null_window_lnL(weo["even"], weo["var_even"])
                       + null_window_lnL(weo["odd"], weo["var_odd"]))
            x2p_shift = null_hs - null_eo if adapter is None else 0.

        def _x2p(fn, label, star_num, extra, **kw):
            """Dispatch one x2P (even/odd) scenario with the comparability shift."""
            if not do_x2p:
                return
            if label in drop_scenario:
                rows.append(dict(ID=ID, scenario=label, star_num=star_num,
                                 lnZ=-np.inf, R_p=np.nan, eta=eb_eta))
                return
            res = _call(fn, weo["t_eo"], weo["even"], sig, weo["var_even"],
                     weo["t_eo"], weo["odd"], sig, weo["var_odd"], *extra,
                     N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                     exptime=dt, nsamples=nsamples,
                     max_shift=max_anomaly_shift, **kw)
            rows.append(dict(ID=ID, scenario=label, star_num=star_num,
                             lnZ=res["lnZ"] + x2p_shift, R_p=np.nan, eta=eb_eta))

        if i == 0:
            # ---- target star: full scenario enumeration ----
            plx = float(stars["plx"].values[i])
            Tmag = float(stars["Tmag"].values[i])
            Jmag = float(stars["Jmag"].values[i])
            Hmag = float(stars["Hmag"].values[i])
            Kmag = float(stars["Kmag"].values[i])
            if verbose:
                print(f"  target {ID}: TP/EB, PTP/PEB, STP/SEB, DTP/DEB, BTP/BEB ...")

            # ================= TP / EB (signal on the target) =================
            if "TP" in drop_scenario:
                rows.append(dict(ID=ID, scenario="TP", star_num=1, lnZ=-np.inf, R_p=np.nan, eta=1.0))
            else:
                res = _call(_MF.lnZ_TTP_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"], P_orb, M_s, R_s, Teff, Z,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples)
                lnZ_TP = res["lnZ"] + secondary_null
                rows.append(dict(ID=ID, scenario="TP", star_num=1, lnZ=lnZ_TP,
                                 R_p=res["R_p"][0], eta=1.0))

            if "EB" in drop_scenario:
                rows.append(dict(ID=ID, scenario="EB", star_num=1, lnZ=-np.inf, R_p=np.nan, eta=eb_eta))
            else:
                res = _call(_MEF.lnZ_TEB_secondary_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"],
                    w["t_sec"], w["sec"], sig, w["var_sec"],
                    P_orb, M_s, R_s, Teff, Z,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples, max_shift=max_anomaly_shift)
                rows.append(dict(ID=ID, scenario="EB", star_num=1, lnZ=res["lnZ"], R_p=np.nan, eta=eb_eta))

            # ================= P family (bound companion dilutes): PTP / PEB =================
            if "PTP" in drop_scenario:
                rows.append(dict(ID=ID, scenario="PTP", star_num=1, lnZ=-np.inf, R_p=np.nan, eta=1.0))
            else:
                res = _call(_MF.lnZ_PTP_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"], P_orb, M_s, R_s, Teff, Z,
                    plx, contrast_curve_file, filt,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples, molusc_file=molusc_file)
                lnZ_PTP = res["lnZ"] + secondary_null
                rows.append(dict(ID=ID, scenario="PTP", star_num=1, lnZ=lnZ_PTP,
                                 R_p=res["R_p"][0], eta=1.0))

            if "PEB" in drop_scenario:
                rows.append(dict(ID=ID, scenario="PEB", star_num=1, lnZ=-np.inf, R_p=np.nan, eta=eb_eta))
            else:
                res = _call(_MEF.lnZ_PEB_secondary_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"],
                    w["t_sec"], w["sec"], sig, w["var_sec"],
                    P_orb, M_s, R_s, Teff, Z,
                    plx, contrast_curve_file, filt,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples, molusc_file=molusc_file,
                    max_shift=max_anomaly_shift)
                rows.append(dict(ID=ID, scenario="PEB", star_num=1, lnZ=res["lnZ"], R_p=np.nan, eta=eb_eta))

            # ================= S family (planet/EB on the bound companion): STP / SEB =========
            if "STP" in drop_scenario:
                rows.append(dict(ID=ID, scenario="STP", star_num=2, lnZ=-np.inf, R_p=np.nan, eta=1.0))
            else:
                res = _call(_MF.lnZ_STP_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"], P_orb, M_s, R_s, Teff, Z,
                    plx, contrast_curve_file, filt,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples, molusc_file=molusc_file)
                lnZ_STP = res["lnZ"] + secondary_null
                rows.append(dict(ID=ID, scenario="STP", star_num=2, lnZ=lnZ_STP,
                                 R_p=res["R_p"][0], eta=1.0))

            if "SEB" in drop_scenario:
                rows.append(dict(ID=ID, scenario="SEB", star_num=2, lnZ=-np.inf, R_p=np.nan, eta=eb_eta))
            else:
                res = _call(_MEF.lnZ_SEB_secondary_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"],
                    w["t_sec"], w["sec"], sig, w["var_sec"],
                    P_orb, M_s, R_s, Teff, Z,
                    plx, contrast_curve_file, filt,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples, molusc_file=molusc_file,
                    max_shift=max_anomaly_shift)
                rows.append(dict(ID=ID, scenario="SEB", star_num=2, lnZ=res["lnZ"], R_p=np.nan, eta=eb_eta))

            # ================= D family (resolved/aligned star): DTP / DEB =================
            if "DTP" in drop_scenario:
                rows.append(dict(ID=ID, scenario="DTP", star_num=1, lnZ=-np.inf, R_p=np.nan, eta=1.0))
            else:
                res = _call(_MF.lnZ_DTP_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"], P_orb, M_s, R_s, Teff, Z,
                    Tmag, Jmag, Hmag, Kmag, trilegal_fname, contrast_curve_file, filt,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples)
                lnZ_DTP = res["lnZ"] + secondary_null
                rows.append(dict(ID=ID, scenario="DTP", star_num=1, lnZ=lnZ_DTP,
                                 R_p=res["R_p"][0], eta=1.0))

            if "DEB" in drop_scenario:
                rows.append(dict(ID=ID, scenario="DEB", star_num=1, lnZ=-np.inf, R_p=np.nan, eta=eb_eta))
            else:
                res = _call(_MEF.lnZ_DEB_secondary_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"],
                    w["t_sec"], w["sec"], sig, w["var_sec"],
                    P_orb, M_s, R_s, Teff, Z,
                    Tmag, Jmag, Hmag, Kmag, trilegal_fname, contrast_curve_file, filt,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples, max_shift=max_anomaly_shift)
                rows.append(dict(ID=ID, scenario="DEB", star_num=1, lnZ=res["lnZ"], R_p=np.nan, eta=eb_eta))

            # ================= B family (background star): BTP / BEB (NO Z arg) =================
            if "BTP" in drop_scenario:
                rows.append(dict(ID=ID, scenario="BTP", star_num=2, lnZ=-np.inf, R_p=np.nan, eta=1.0))
            else:
                res = _call(_MF.lnZ_BTP_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"], P_orb, M_s, R_s, Teff,
                    Tmag, Jmag, Hmag, Kmag, trilegal_fname, contrast_curve_file, filt,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples)
                lnZ_BTP = res["lnZ"] + secondary_null
                rows.append(dict(ID=ID, scenario="BTP", star_num=2, lnZ=lnZ_BTP,
                                 R_p=res["R_p"][0], eta=1.0))

            if "BEB" in drop_scenario:
                rows.append(dict(ID=ID, scenario="BEB", star_num=2, lnZ=-np.inf, R_p=np.nan, eta=eb_eta))
            else:
                res = _call(_MEF.lnZ_BEB_secondary_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"],
                    w["t_sec"], w["sec"], sig, w["var_sec"],
                    P_orb, M_s, R_s, Teff,
                    Tmag, Jmag, Hmag, Kmag, trilegal_fname, contrast_curve_file, filt,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples, max_shift=max_anomaly_shift)
                rows.append(dict(ID=ID, scenario="BEB", star_num=2, lnZ=res["lnZ"], R_p=np.nan, eta=eb_eta))

            # ================= x2P (period-doubled EB; even/odd folds) =================
            _x2p(_MEF.lnZ_TEB_evenodd_fourier, "EBx2P", 1, (P_orb, M_s, R_s, Teff, Z))
            _x2p(_MEF.lnZ_PEB_evenodd_fourier, "PEBx2P", 1,
                 (P_orb, M_s, R_s, Teff, Z, plx, contrast_curve_file, filt),
                 molusc_file=molusc_file)
            _x2p(_MEF.lnZ_SEB_evenodd_fourier, "SEBx2P", 2,
                 (P_orb, M_s, R_s, Teff, Z, plx, contrast_curve_file, filt),
                 molusc_file=molusc_file)
            _x2p(_MEF.lnZ_DEB_evenodd_fourier, "DEBx2P", 1,
                 (P_orb, M_s, R_s, Teff, Z, Tmag, Jmag, Hmag, Kmag,
                  trilegal_fname, contrast_curve_file, filt))
            _x2p(_MEF.lnZ_BEB_evenodd_fourier, "BEBx2P", 2,           # B family: NO Z
                 (P_orb, M_s, R_s, Teff, Tmag, Jmag, Hmag, Kmag,
                  trilegal_fname, contrast_curve_file, filt))

        else:
            # ---- nearby star: NTP / NEB (solar defaults for missing params) ----
            if np.isnan(Teff): Teff = 5777.0
            if np.isnan(M_s): M_s = 1.0
            if np.isnan(R_s): R_s = 1.0
            if verbose:
                print(f"  nearby {ID}: NTP / NEB ...")
            if "NTP" in drop_scenario:
                rows.append(dict(ID=ID, scenario="NTP", star_num=1, lnZ=-np.inf, R_p=np.nan, eta=1.0))
            else:
                res = _call(_MF.lnZ_TTP_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"], P_orb, M_s, R_s, Teff, Z,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples)
                lnZ_NTP = res["lnZ"] + secondary_null
                rows.append(dict(ID=ID, scenario="NTP", star_num=1, lnZ=lnZ_NTP,
                                 R_p=res["R_p"][0], eta=1.0))

            if "NEB" in drop_scenario:
                rows.append(dict(ID=ID, scenario="NEB", star_num=1, lnZ=-np.inf, R_p=np.nan, eta=eb_eta))
            else:
                res = _call(_MEF.lnZ_TEB_secondary_fourier,
                    w["t_pri"], w["prim"], sig, w["var_pri"],
                    w["t_sec"], w["sec"], sig, w["var_sec"],
                    P_orb, M_s, R_s, Teff, Z,
                    N=N, steps=steps, mission=mission, flatpriors=flatpriors,
                    exptime=dt, nsamples=nsamples, max_shift=max_anomaly_shift)
                rows.append(dict(ID=ID, scenario="NEB", star_num=1, lnZ=res["lnZ"], R_p=np.nan, eta=eb_eta))
            _x2p(_MEF.lnZ_TEB_evenodd_fourier, "NEBx2P", 1, (P_orb, M_s, R_s, Teff, Z))

        # stamp this star's flat-model null on every row it produced (see comment above)
        for r in rows[n_rows_star:]:
            r["null"] = null_hs

    df = pd.DataFrame(rows)
    df.attrs.update(weighting=weighting, timing_policy=timing_policy,
                    backend=backend if adapter is not None else "scalar")
    if adapter is not None:
        df.attrs["noise_model"] = dict(
            representation="full_period_parities" if len(metric.blocks) == 2 else "full_period",
            compression="inverse_variance_sufficient_statistic_for_P_models",
            mode_policy="drop_DC_and_real_Nyquist", aperture_frame=True,
            time=[b.time.copy() for b in metric.blocks], flux=[b.flux.copy() for b in metric.blocks],
            variance=[b.variance.copy() for b in metric.blocks], null_loglike=metric.null_loglike)
    if df["lnZ"].isna().any():
        bad = df.loc[df["lnZ"].isna(), "scenario"].tolist()
        print(f"  WARNING: NaN lnZ for scenarios {bad} -- treated as -inf (excluded from "
              f"the FPP); investigate the sampler/inputs", flush=True)
    # posterior probabilities: softmax over the null-referenced Bayes factors x eta.
    # lnBF = lnZ - null is invariant to the per-star renorm frame (Jacobian cancels).
    df["lnBF"] = df["lnZ"].values - df["null"].values
    with np.errstate(divide="ignore"):
        lnw = df["lnBF"].values + np.log(df["eta"].values)
    lnw = np.where(np.isnan(lnw), -np.inf, lnw)
    finite = np.isfinite(lnw)
    if not finite.any():
        df["prob"] = np.nan
        df.FPP = float("nan"); df.NFPP = float("nan")
        df.attrs["FPP"] = df.FPP; df.attrs["NFPP"] = df.NFPP
        print("  WARNING: no scenario has finite evidence -- FPP undefined (NaN)", flush=True)
        return df
    lnw = lnw - np.max(lnw[finite])
    p = np.exp(lnw)
    p = p / p.sum()
    df["prob"] = p
    # planet-on-the-target scenarios (mirrors calc_probs: prob[0]+prob[3]+prob[9])
    planet = df["scenario"].isin(["TP", "PTP", "DTP"])
    nearby = df["scenario"].isin(["NTP", "NEB", "NEBx2P"])
    FPP = float(1.0 - df.loc[planet, "prob"].sum())
    NFPP = float(df.loc[nearby, "prob"].sum())
    df.FPP = FPP; df.NFPP = NFPP                 # legacy attribute API (lost on df.copy())
    df.attrs["FPP"] = FPP; df.attrs["NFPP"] = NFPP   # survives copy on modern pandas
    if verbose:
        print(f"  FPP = {df.FPP:.4f}   NFPP = {df.NFPP:.4f}")
    return df


def calc_probs_joint_fourier(*args, **kwargs):
    """Opt-in joint Fourier inference on observed samples, preserving gaps and 2P.

    See :func:`pentaceratops.evidence.joint_fourier.calc_probs_joint_fourier`.
    The uniform calc_probs_fourier entry point also uses consistent scenario weights.
    """
    from .evidence.joint_fourier import calc_probs_joint_fourier as joint
    return joint(*args, **kwargs)
