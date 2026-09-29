#!/usr/bin/env python3
"""Offline wheel smoke test; never install into or modify the active environment.

Run after building a wheel: python tools/verify_distribution.py path/to/*.whl
Artifacts are unpacked to a temporary directory, not to the research checkout.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--tests", type=Path, help="Optionally run this test directory against the wheel")
    parser.add_argument("--reference-root", type=Path, help="Preserved engine for opt-in test comparisons")
    parser.add_argument("--junitxml", type=Path, help="Optional pytest report destination")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pentaceratops-wheel-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(args.wheel) as wheel:
            names = wheel.namelist()
            assert "pentaceratops/data/ldc_tess.csv" in names
            assert "pentaceratops/data/ldc_kepler.csv" in names
            assert any(name.startswith("pentaceratops/companions/data/") for name in names)
            assert not any(name.startswith(("research/", "triceratops/")) for name in names)
            wheel.extractall(root)
        environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
        script = """
from pathlib import Path
import sys
import pentaceratops
assert Path(pentaceratops.__file__).is_relative_to(Path.cwd())
from pentaceratops import Target, RunResult, run_evidence, scenario_probabilities
from pentaceratops.evidence import real, eclipses, fourier, fourier_eclipses
from pentaceratops.experimental import covariance, sampler_mode
from pentaceratops.companions.load import load_isochrones
load_isochrones(5.0)
assert 'triceratops' not in sys.modules
assert 'lightkurve' not in sys.modules
assert not any(name.startswith('astroquery') for name in sys.modules)
print('Wheel imports and bundled tables OK; no research-package or catalog imports.')
"""
        subprocess.run([sys.executable, "-c", script], cwd=root, env=environment, check=True)
        subprocess.run([sys.executable, "-m", "pentaceratops", "doctor"],
                       cwd=root, env=environment, check=True)
        if args.tests is not None:
            tests = args.tests.resolve()
            command = [sys.executable, "-m", "pytest", str(tests),
                       "-c", str(tests.parent / "pyproject.toml"), "-q", "-p", "no:cacheprovider"]
            if args.reference_root is not None:
                command.extend(["--reference-root", str(args.reference_root.resolve())])
            if args.junitxml is not None:
                command.extend(["--junitxml", str(args.junitxml.resolve())])
            subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
