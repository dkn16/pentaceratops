"""Shared CLI behind the two editable download/preprocessing example scripts."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def default_cache_dir():
    explicit = os.environ.get("PENTACERATOPS_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    scratch = os.environ.get("PSCRATCH") or os.environ.get("SCRATCH") or tempfile.gettempdir()
    return Path(scratch) / "pentaceratops" / "candidate_examples"


def configure_runtime(cache_dir):
    """Call before numerical/catalog imports; honor explicit user environment."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    for name, suffix in (("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba"),
                         ("XDG_CACHE_HOME", "xdg"), ("XDG_CONFIG_HOME", "config")):
        os.environ.setdefault(name, str(cache_dir / "runtime" / suffix))


def positive(value):
    import math
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return value


def segment_list(value):
    try:
        result = sorted({int(item) for item in value.replace(",", " ").split()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated sector/quarter integers") from error
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError("segments must be nonnegative integers")
    return result


def build_parser(mission, default_target):
    parser = argparse.ArgumentParser(description=f"Download and FGP-preprocess one {mission} candidate; no evidence sampling.")
    parser.add_argument("--target", default=default_target, help="TIC/TOI or KIC/KOI identifier")
    parser.add_argument("--candidate", help="TOI/KOI selection for a multi-candidate host")
    parser.add_argument("--catalog-source", choices=("nasa", "bayesian"), default="nasa",
                        help="NASA Exoplanet Archive by default; Bayesian Exoplanets is optional for TESS")
    parser.add_argument("--catalog-file", type=Path, help="Optional local Bayesian tois.csv snapshot")
    parser.add_argument("--stage", choices=("resolve", "download", "prepare"), default="prepare")
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--segments", "--sectors", "--quarters", type=segment_list,
                        help="Restrict the MAST query to these sectors/quarters")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--plot-dir", type=Path, default=Path("plots") / "candidate_examples")
    parser.add_argument("--plot-only", type=Path, metavar="PREPARED_NPZ",
                        help="Replot a saved preparation with no network or FGP fitting")
    parser.add_argument("--offline", action="store_true", help="Require existing catalog and native caches")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--cadence-seconds", type=positive, help="Select a specific available archive cadence")
    parser.add_argument("--period", type=positive, help="Override orbital period [days]")
    parser.add_argument("--epoch-bjd", type=positive, help="Override epoch [full BJD, NOT BTJD/BKJD]")
    parser.add_argument("--duration-hours", type=positive)
    parser.add_argument("--depth-ppm", type=positive)
    parser.add_argument("--keep-other-planets", action="store_true",
                        help="Diagnostic only: disable automatic masking of other catalogued planets")
    parser.add_argument("--mask-planet", action="append", default=[], metavar="PERIOD,BJD,DURATION_HOURS",
                        help="Add an uncatalogued companion ephemeris to mask; repeatable")
    parser.add_argument("--bin-minutes", type=positive, default=10. if mission == "TESS" else 30.)
    parser.add_argument("--window-durations", type=positive, default=4.,
                        help="Primary/secondary half-width in transit durations, capped below P/4")
    parser.add_argument("--max-half-period-fraction", type=positive, default=.25)
    parser.add_argument("--other-planet-mask-durations", type=positive, default=1.5)
    parser.add_argument("--fgp-iterations", type=int, default=3)
    parser.add_argument("--fgp-cutoff-per-day", type=positive, default=3.)
    parser.add_argument("--padding-fraction", type=float, default=.1)
    parser.add_argument("--minimum-window-bins", type=int, default=8)
    return parser


def main(mission, default_target, argv=None):
    parser = build_parser(mission, default_target)
    args = parser.parse_args(argv)
    args.cache_dir = args.cache_dir.expanduser().resolve()
    configure_runtime(args.cache_dir)
    from dataclasses import asdict
    from .catalog import catalog_rows, parse_identifier, resolve_candidate, candidate_from_row
    from .candidate import PreparationOptions, prepare_candidate, plot_preparation
    from .download import download_products
    from ..results import RunResult
    options = PreparationOptions(bin_minutes=args.bin_minutes, window_durations=args.window_durations,
        max_half_period_fraction=args.max_half_period_fraction,
        companion_mask_durations=args.other_planet_mask_durations,
        fgp_iterations=args.fgp_iterations, fgp_cutoff_per_day=args.fgp_cutoff_per_day,
        padding_fraction=args.padding_fraction, minimum_window_bins=args.minimum_window_bins)
    options.validate()
    if args.plot_only:
        result = RunResult.load(args.plot_only)
        if result.metadata.get("artifact") != "candidate_preparation":
            raise ValueError("--plot-only requires a candidate preparation bundle")
        for path in plot_preparation(result, args.plot_dir, stem=args.plot_only.stem):
            print(path)
        return 0
    if args.list_candidates:
        rows, _ = catalog_rows(parse_identifier(args.target, mission), args.cache_dir, offline=args.offline,
                               source=args.catalog_source, catalog_file=args.catalog_file)
        print(json.dumps(rows, indent=2))
        return 0
    overrides = dict(period_days=args.period, epoch_bjd=args.epoch_bjd,
                     duration_hours=args.duration_hours, depth_ppm=args.depth_ppm)
    ephem, companions, _ = resolve_candidate(args.target, args.cache_dir, mission=mission,
        candidate=args.candidate, overrides=overrides, offline=args.offline,
        mask_other_planets=not args.keep_other_planets, source=args.catalog_source, catalog_file=args.catalog_file)
    extra = []
    for index, specification in enumerate(args.mask_planet):
        try:
            period, epoch, duration = map(float, specification.split(","))
        except ValueError as error:
            raise ValueError("--mask-planet requires PERIOD,FULL_BJD,DURATION_HOURS") from error
        extra.append(candidate_from_row({}, mission, ephem.host_id,
            dict(period_days=period, epoch_bjd=epoch, duration_hours=duration), identifier=f"manual-{index+1}"))
    print(json.dumps(dict(candidate=asdict(ephem), masked_companions=[asdict(c) for c in companions+extra],
                          catalog_source=args.catalog_source,
                          cache_dir=str(args.cache_dir), options=asdict(options)), indent=2), flush=True)
    if args.stage == "resolve":
        return 0
    if args.stage == "download":
        _, _, path = download_products(ephem, args.cache_dir, segments=args.segments,
            cadence_seconds=args.cadence_seconds, offline=args.offline)
        print(f"Native cache: {path}", flush=True)
        return 0
    result, path = prepare_candidate(args.target, args.cache_dir, mission=mission, candidate=args.candidate,
        segments=args.segments, cadence_seconds=args.cadence_seconds, overrides=overrides,
        options=options, offline=args.offline, mask_other_planets=not args.keep_other_planets,
        extra_companions=extra, source=args.catalog_source, catalog_file=args.catalog_file)
    print(f"Prepared data: {path}", flush=True)
    print("Preparation only: no stellar-field model, FPP, or full-period Fourier fit has been calculated.")
    if not args.no_plots:
        for output in plot_preparation(result, args.plot_dir, stem=path.stem):
            print(f"Plot: {output}")
    return 0
