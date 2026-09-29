"""Small offline CLI; intentionally not a production TIC runner yet."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pentaceratops")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report installed dependencies; no queries or downloads")
    commands.add_parser("sampling-policy", help="Show provisional real-space sampling defaults")
    summary = commands.add_parser("inspect", help="Summarize a saved result bundle")
    summary.add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = {"pentaceratops": __version__}
        required = ("numpy", "scipy", "pandas", "astropy", "pytransit", "matplotlib",
                    "mechanicalsoup", "beautifulsoup4", "setuptools")
        missing = []
        for name in required + ("astroquery", "lightkurve"):
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
