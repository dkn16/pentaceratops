"""Download-first candidate preparation, independent of research directories.

This stage produces MAP-FGP windowed photometry and replay information. It
does not construct a stellar field, compute an FPP, or prepare a full-period
Fourier-likelihood product. Those must not be inferred from these windows.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .catalog import resolve_candidate
from .download import download_products, fingerprint
from .flux import APERTURE_FRAME, aperture_catalog_depth
from .fgp import detrend_regular_segment

PREPARATION_VERSION = "download-map-matched-windows-v1"
PANELS = ("even", "odd", "secondary")


@dataclass(frozen=True)
class PreparationOptions:
    bin_minutes: float = 10.0
    window_durations: float = 4.0
    max_half_period_fraction: float = .25
    companion_mask_durations: float = 1.5
    fgp_iterations: int = 3
    fgp_cutoff_per_day: float = 3.0
    padding_fraction: float = .1
    minimum_window_bins: int = 8

    def validate(self):
        for name in ("bin_minutes", "window_durations", "companion_mask_durations", "fgp_cutoff_per_day"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.max_half_period_fraction <= .25:
            raise ValueError("max_half_period_fraction must be in (0, .25]")
        if not np.isfinite(self.padding_fraction) or not 0 <= self.padding_fraction <= 1:
            raise ValueError("padding_fraction must be in [0, 1]")
        for name in ("fgp_iterations", "minimum_window_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def phase(time, epoch, period):
    cycles = np.floor((np.asarray(time) - epoch) / period + .5).astype(np.int64)
    return np.asarray(time) - (epoch + cycles * period), cycles


def regular_bin(time, flux, error, dt):
    time, flux, error = map(lambda x: np.asarray(x, float), (time, flux, error))
    valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    time, flux, error = time[valid], flux[valid], error[valid]
    if not len(time):
        raise ValueError("No usable samples remain after companion masking")
    origin = np.floor(np.min(time) / dt) * dt
    ids = np.floor((time - origin) / dt + .5).astype(int)
    size = int(ids.max()) + 1
    if size > 2_000_000:
        raise ValueError("Unreasonable regular-grid length; check time units/cadence")
    weights = 1 / error**2
    sums = np.bincount(ids, weights=weights, minlength=size)
    observed = sums > 0
    f, e = np.full(size, np.nan), np.full(size, np.nan)
    f[observed] = np.bincount(ids, weights=weights * flux, minlength=size)[observed] / sums[observed]
    e[observed] = 1 / np.sqrt(sums[observed])
    return origin + np.arange(size) * dt, f, e


def window_grid(candidate, dt, options):
    requested = options.window_durations * candidate.duration_days
    cap = options.max_half_period_fraction * candidate.period_days
    n = min(int(np.ceil(requested / dt)), int(np.ceil(cap / dt - .5)) - 1)
    if n < 1 or 2 * n + 1 < options.minimum_window_bins:
        raise ValueError("Too few bins within disjoint primary/secondary windows; reduce --bin-minutes")
    centers = np.arange(-n, n + 1) * dt
    half_width = (n + .5) * dt
    if half_width < candidate.duration_days:
        raise ValueError("Window cannot retain the full transit plus a duration margin; check period/duration")
    return centers


def assignments(time, observed, candidate, centers, dt):
    """Identical half-open bins for FGP protection and final folded inputs."""
    primary, cycles = phase(time, candidate.epoch_relative, candidate.period_days)
    secondary, _ = phase(time, candidate.epoch_relative + candidate.period_days/2, candidate.period_days)
    edges = np.r_[centers[0] - dt/2, centers + dt/2]
    result = {}
    for panel in PANELS:
        relative = secondary if panel == "secondary" else primary
        ids = np.searchsorted(edges, relative, side="right") - 1
        choose = np.asarray(observed, bool) & (ids >= 0) & (ids < len(centers))
        if panel != "secondary":
            choose &= cycles % 2 == int(panel == "odd")
        result[panel] = (ids, choose)
    multiplicity = np.sum([use for _, use in result.values()], axis=0)
    if np.any(multiplicity > 1):
        raise ValueError("A cadence appears in more than one likelihood panel")
    return result, multiplicity > 0


def fold(time, flux, error, candidate, centers, dt):
    assigned, _ = assignments(time, np.ones(len(time), bool), candidate, centers, dt)
    result = {}
    # All-primary is a diagnostic, never an extra independent likelihood term.
    e, o = assigned["even"], assigned["odd"]
    assigned["primary"] = (e[0], e[1] | o[1])
    for panel, (ids, use) in assigned.items():
        chosen = ids[use]
        weights = 1 / error[use]**2
        counts = np.bincount(chosen, minlength=len(centers))
        if np.any(counts == 0):
            raise ValueError(f"Incomplete {panel} window: {np.sum(counts == 0)} empty bins. "
                             "Select more sectors/quarters or explicitly adjust bin/window settings; no gaps were filled.")
        sums = np.bincount(chosen, weights=weights, minlength=len(centers))
        result[f"time_{panel}"] = centers.copy()
        result[f"flux_{panel}"] = np.bincount(chosen, weights=weights*flux[use], minlength=len(centers)) / sums
        result[f"err_{panel}"] = 1 / np.sqrt(sums)
        result[f"count_{panel}"] = counts
    return result


def preprocess_blocks(blocks, candidate, companions=(), options=None):
    """Pure array entry point; no catalogs, filesystem writes, or downloads."""
    options = PreparationOptions() if options is None else options
    options.validate()
    if not blocks:
        raise ValueError("No photometry supplied")
    dt = max(options.bin_minutes * 60, max(block["exposure_seconds"] for block in blocks)) / 86400
    centers = window_grid(candidate, dt, options)
    sectors, pieces, diagnostics = [], [], []
    for block in blocks:
        time, raw, error = (np.asarray(block[key], float) for key in ("time", "flux", "error"))
        if not (time.ndim == 1 and time.shape == raw.shape == error.shape):
            raise ValueError("Native time/flux/error shapes differ")
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
            raise ValueError("Native times must be finite and strictly increasing")
        keep = np.ones(len(time), bool)
        for other in companions:
            relative, _ = phase(time, other.epoch_relative, other.period_days)
            # Remove before regular-grid binning, with an exposure-edge margin.
            keep &= abs(relative) > options.companion_mask_durations * other.duration_days + dt/2
        t, f, e = regular_bin(time[keep], raw[keep], error[keep], dt)
        base_length = len(t)
        padding = int(np.ceil(base_length * options.padding_fraction))
        if padding:
            t = np.r_[t, t[-1] + np.arange(1, padding + 1) * dt]
            f, e = np.r_[f, np.full(padding, np.nan)], np.r_[e, np.full(padding, np.nan)]
        observed = np.isfinite(f) & np.isfinite(e) & (e > 0)
        _, protected = assignments(t, observed, candidate, centers, dt)
        training = observed & ~protected
        if training.sum() < 140:
            raise ValueError(f"Segment {block['segment']}: only {training.sum()} training bins after exact "
                             "window protection (need 140). Explicitly reduce --window-durations; masks were not narrowed.")
        if int(options.fgp_cutoff_per_day * dt * len(t)) >= (len(t)-1)//2:
            raise ValueError("FGP cutoff reaches Nyquist; reduce --fgp-cutoff-per-day")
        print(f"FGP segment {block['segment']}: {observed.sum()} observed, "
              f"{protected.sum()} protected, {training.sum()} training bins", flush=True)
        model = detrend_regular_segment(t, f, e, protected, dt,
                                        iterations=options.fgp_iterations,
                                        cutoff_per_day=options.fgp_cutoff_per_day)
        if not np.array_equal(model["fit_mask"], training):
            raise ValueError("FGP training/protection mismatch")
        if not np.all(np.isfinite(model["residual_flux"][observed])):
            raise ValueError("Nonfinite FGP residuals")
        sectors.append(dict(segment=int(block["segment"]), time=t, raw_flux=f, input_error=e,
                            native_keep=keep, base_length=base_length, **model))
        pieces.append(dict(time=t[observed], raw=f[observed], flux=model["residual_flux"][observed],
                           error=model["relative_error"][observed], protected=protected[observed],
                           segment=np.full(observed.sum(), int(block["segment"]))))
        diagnostics.append(dict(segment=int(block["segment"]), observed_bins=int(observed.sum()),
                                protected_bins=int(protected.sum()), training_bins=int(training.sum()),
                                masked_native_samples=int((~keep).sum()),
                                center=float(model["center"]), scale=float(model["scale"]),
                                mode_count=int(model["mode_count"]), training_likelihood_overlap=0))
    stream = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    order = np.argsort(stream["time"], kind="stable")
    stream = {key: value[order] for key, value in stream.items()}
    if np.any(np.diff(stream["time"]) <= 0):
        raise ValueError("Overlapping sector/quarter regular grids; duplicate science data must be resolved")
    windows = fold(stream["time"], stream["flux"], stream["error"], candidate, centers, dt)
    _, science = assignments(stream["time"], np.ones(len(order), bool), candidate, centers, dt)
    np.testing.assert_array_equal(science, stream["protected"])
    windows.update(period=candidate.period_days, duration_days=candidate.duration_days,
                   epoch_bjd=candidate.epoch_bjd, time_zero_bjd=candidate.time_zero,
                   exptime_days=dt, bin_days=dt, window_half_days=float(centers[-1]+dt/2),
                   flux_frame=APERTURE_FRAME)
    return dict(windows=windows, stream=stream, segments=sectors), diagnostics


def prepare_candidate(target, cache_dir, *, mission=None, candidate=None, segments=None,
                      cadence_seconds=None, overrides=None, options=None, offline=False,
                      mask_other_planets=True, extra_companions=(), source="nasa", catalog_file=None):
    """Resolve, download, and cache preparation. Does not invoke an FPP engine."""
    from .. import __version__
    from ..results import RunResult
    from ..api import _source_fingerprint
    options = PreparationOptions() if options is None else options
    options.validate()
    ephem, companions, catalog = resolve_candidate(target, cache_dir, mission=mission,
        candidate=candidate, overrides=overrides, offline=offline, mask_other_planets=mask_other_planets,
        source=source, catalog_file=catalog_file)
    companions.extend(extra_companions)
    for other in companions:
        if other.mission != ephem.mission or other.host_id != ephem.host_id:
            raise ValueError("Companion ephemeris belongs to another mission or host")
    blocks, native, native_path = download_products(ephem, cache_dir, segments=segments,
                                                   cadence_seconds=cadence_seconds, offline=offline)
    settings = dict(preparation_version=PREPARATION_VERSION, package_version=__version__,
                    source_sha256=_source_fingerprint(), candidate=asdict(ephem),
                    companions=[asdict(other) for other in companions], options=asdict(options),
                    native_selection=native["selection"], native_products=native["products"],
                    mask_other_planets=mask_other_planets, catalog_source=source,
                    catalog_provenance=catalog["queries"])
    path = Path(cache_dir) / "prepared" / f"{ephem.tag}_{fingerprint(settings)[:16]}.npz"
    if path.exists():
        result = RunResult.load(path)
        if result.metadata["settings"] != settings:
            raise ValueError("Prepared-cache configuration mismatch")
        return result, path
    prepared, diagnostics = preprocess_blocks(blocks, ephem, companions, options)
    if ephem.depth_ppm is not None:
        times, errors, labels = [], [], []
        for block, sector in zip(blocks, prepared["segments"]):
            use = sector["native_keep"]
            times.append(block["time"][use])
            errors.append(block["error"][use])
            labels.append(np.full(use.sum(), block["segment"]))
        depth, factor = aperture_catalog_depth(ephem.depth_ppm * 1e-6,
            np.concatenate(times), np.concatenate(errors), np.concatenate(labels), native["products"],
            ephem.epoch_relative, ephem.period_days, ephem.duration_days)
        prepared["windows"].update(catalog_transit_depth=ephem.depth_ppm*1e-6,
                                     transit_depth=depth, catalog_depth_aperture_scale=factor)
    result = RunResult(prepared, metadata=dict(artifact="candidate_preparation", settings=settings,
        catalog=catalog, native_path=str(native_path.resolve()), diagnostics=diagnostics,
        likelihood_ready=False, uncertainty="measurement errors only; FGP uncertainty not yet propagated",
        scope="MAP-FGP window preparation; no stellar field, scenario evidence or Fourier-likelihood product"))
    try:
        result.save(path)
    except FileExistsError:
        saved = RunResult.load(path)
        if saved.metadata["settings"] != settings:
            raise ValueError("Prepared-cache configuration mismatch")
        result = saved
    return result, path


def plot_preparation(result, plot_dir, *, stem=None):
    """Plot only saved arrays. No catalog access, detrending, or sampling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    windows = result.output["windows"]
    info = result.metadata["settings"]["candidate"]
    stem = stem or info["identifier"].replace(" ", "_")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True, constrained_layout=True)
    for ax, panel in zip(axes.flat, ("primary", "secondary", "even", "odd")):
        ax.errorbar(windows[f"time_{panel}"], (windows[f"flux_{panel}"]-1)*1e6,
                    yerr=windows[f"err_{panel}"]*1e6, fmt=".", ms=3, color=".3", ecolor=".7", lw=.6)
        ax.axhline(0, color=".7", lw=.6)
        ax.axvspan(-windows["duration_days"]/2, windows["duration_days"]/2, color="C0", alpha=.07)
        ax.set_title(panel.title() + (" (diagnostic only)" if panel == "primary" else ""))
        ax.grid(alpha=.15)
    for ax in axes[-1]:
        ax.set_xlabel("Time from window center [d]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized flux − 1 [ppm]")
    fig.suptitle(f"{info['identifier']} · MAP-FGP inputs (measurement errors only)")
    window_path = plot_dir / f"{stem}_input_windows.png"
    fig.savefig(window_path, dpi=160)
    plt.close(fig)
    paths = [window_path]
    for sector in result.output["segments"]:
        t, observed = sector["time"], sector["observed"]
        protected = sector["protected"]
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, constrained_layout=True)
        axes[0].plot(t[observed], (sector["raw_flux"][observed]-1)*1e3, ".", color=".65", ms=1, label="Aperture flux")
        trend = sector["center"] + sector["center"] * sector["model_flux"]
        axes[0].plot(t, (trend-1)*1e3, color="C0", lw=1, label="FGP trend")
        axes[1].plot(t[observed], (sector["residual_flux"][observed]-1)*1e6, ".", color=".45", ms=1)
        axes[1].plot(t[protected], (sector["residual_flux"][protected]-1)*1e6,
                     ".", color="C1", ms=1.5, label="Protected window samples")
        axes[0].set_ylabel("Flux − 1 [ppt]")
        axes[1].set_ylabel("Residual [ppm]")
        axes[1].set_xlabel(f"BJD − {windows['time_zero_bjd']:.0f}")
        for ax in axes:
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=.15)
        fig.suptitle(f"{info['identifier']} · {'sector' if info['mission'] == 'TESS' else 'quarter'} {sector['segment']}")
        path = plot_dir / f"{stem}_segment{sector['segment']}_fgp.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths
