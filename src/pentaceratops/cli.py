"""Offline evidence, saved-result inspection and dependency reporting."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pentaceratops")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report installed dependencies; no queries or downloads")
    commands.add_parser("sampling-policy", help="Show shared per-scenario sampling defaults")
    summary = commands.add_parser("inspect", help="Summarize a saved result bundle")
    summary.add_argument("path")
    evidence = commands.add_parser("evidence", help="Run evidence from prepared files and a JSON configuration")
    evidence.add_argument("config", help="JSON configuration; relative input paths use its directory")
    evidence.add_argument("--likelihood", choices=("real", "fourier"))
    evidence.add_argument("--output", help="New result NPZ path, relative to the current directory")
    for name in ("N", "steps", "nsamples", "seed", "posterior-samples"):
        evidence.add_argument("--"+name, type=int, default=None)
    evidence.add_argument("--eb-eta", type=float)
    args = parser.parse_args(argv)
    if args.command == "evidence":
        from .preprocessing.example_cli import configure_runtime, default_cache_dir
        configure_runtime(default_cache_dir())
        from .prepared import run_prepared
        overrides = {key: getattr(args, key) for key in
                     ("N", "steps", "nsamples", "seed", "posterior_samples", "eb_eta")
                     if getattr(args, key) is not None}
        try:
            result = run_prepared(args.config, likelihood=args.likelihood,
                                  output_path=args.output, **overrides)
        except (ValueError, FileNotFoundError, FileExistsError) as error:
            parser.error(str(error))
        table = result.output
        print(json.dumps(dict(output=result.metadata["prepared_run"]["output_path"],
            likelihood=result.metadata["prepared_run"]["likelihood"], scenarios=len(table),
            **{key: table.attrs[key] for key in ("FPP", "FPP_EB", "NFPP", "scenario_scope")}), indent=2))
        return 0
    if args.command == "doctor":
        report = {"pentaceratops": __version__}
        required = ("numpy", "scipy", "pandas", "astropy", "pytransit", "meepmeep", "numba", "matplotlib",
                    "mechanicalsoup", "beautifulsoup4")
        missing = []
        for name in required + ("setuptools", "astroquery", "lightkurve"):
            try:
                report[name] = version(name)
            except PackageNotFoundError:
                report[name] = "not installed"
                if name in required:
                    missing.append(name)
        print(json.dumps(report, indent=2))
        return int(bool(missing))
    if args.command == "sampling-policy":
        from .sampling.policy import sampling_policy
        print(json.dumps(sampling_policy(), indent=2))
        return 0
    from .results import RunResult
    result = RunResult.load(args.path)
    samplers = [item for item in result.sampling if item["kind"] == "sampler"]
    print(json.dumps({"package_version": result.metadata.get("package_version"),
                      "function": result.metadata.get("function"),
                      "sampler_calls": len(samplers),
                      "stored_particles": sum(len(item["weights"]) for item in samplers)}, indent=2))
    return 0
