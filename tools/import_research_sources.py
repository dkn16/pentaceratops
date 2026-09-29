#!/usr/bin/env python3
"""One-time mechanical extraction; refuses to overwrite any imported source.

This is migration tooling, not a runtime dependency or an update command.
Only package imports, data paths and misleading top-level docstrings change.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ENGINE = {
    "triceratops_new.py": "_target.py",
    "funcs.py": "stellar.py",
    "priors.py": "priors.py",
    "likelihoods.py": "likelihoods/real.py",
    "likelihoods_fourier.py": "likelihoods/fourier.py",
    "marginal_likelihoods_new.py": "evidence/real.py",
    "marginal_likelihoods_eb.py": "evidence/eclipses.py",
    "marginal_likelihoods_fourier.py": "evidence/fourier.py",
    "marginal_likelihoods_eb_fourier.py": "evidence/fourier_eclipses.py",
    "calc_probs_fourier.py": "fourier.py",
    "persistent.py": "sampling/persistent.py",
    "scenario_sampling.py": "sampling/policy.py",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relocate(text, destination):
    depth = ".." if "/" in destination else "."
    substitutions = {
        "from .persistent import": f"from {depth}sampling.persistent import",
        "from .likelihoods import": f"from {depth}likelihoods.real import",
        "from .likelihoods_fourier import": f"from {depth}likelihoods.fourier import",
        "from .priors import": f"from {depth}priors import",
        "from .funcs import": f"from {depth}stellar import",
        "from .scenario_sampling import": "from .sampling.policy import",
        "from .marginal_likelihoods_new import": f"from {depth}evidence.real import",
        "from .marginal_likelihoods_eb import": f"from {depth}evidence.eclipses import",
        "from .marginal_likelihoods_fourier import": f"from {depth}evidence.fourier import",
        "from . import marginal_likelihoods_new as": "from .evidence import real as",
        "from . import marginal_likelihoods_eb as": "from .evidence import eclipses as",
        "from . import marginal_likelihoods_fourier as": "from .evidence import fourier as",
        "from . import marginal_likelihoods_eb_fourier as": "from .evidence import fourier_eclipses as",
        "from .stecomp.": "from .companions.",
    }
    for old, new in substitutions.items():
        text = text.replace(old, new)
    if destination in ("evidence/real.py", "evidence/eclipses.py"):
        tree = ast.parse(text)
        if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            end = tree.body[0].end_lineno
            text = ('"""Scenario evidences computed with persistent sampling.\n\n'
                    'Extracted from the corrected research engine; numerical logic is unchanged.\n'
                    'N is the active particle count and steps is the MCMC effort.\n'
                    'See docs/architecture.md for migration limitations.\n"""\n'
                    + "\n".join(text.splitlines()[end:]) + "\n")
        text = text.replace("os.path.dirname(__file__), 'data/", "os.path.dirname(__file__), '../data/")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    home, project = args.home.resolve(), args.project.resolve()
    root = home / "triceratops"
    pairs = [(root / "triceratops" / old, "src/pentaceratops/" + new)
             for old, new in ENGINE.items()]
    for subdir in ("data", "stecomp"):
        for path in sorted((root / "triceratops" / subdir).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".csv", ".npy", ".txt"):
                name = str(path.relative_to(root / "triceratops"))
                pairs.append((path, "src/pentaceratops/" + name.replace("stecomp/", "companions/")))
    for source, destination in (
        ("tess_test/flux_frame.py", "preprocessing/flux.py"),
        ("tess_test/fgp_detrend.py", "preprocessing/fgp.py"),
        ("fgp_uncertainty/fourier_posterior.py", "preprocessing/posterior.py"),
        ("fgp_uncertainty/v2_matched_windows/correlated_likelihood.py", "experimental/covariance.py"),
    ):
        pairs.append((home / source, "src/pentaceratops/" + destination))
    for name in ("batched_priors", "physical_batch", "sparse_kernels", "v2_adapter", "sampler_mode"):
        pairs.append((home / "fgp_uncertainty/v2_batched_cpu" / f"{name}.py",
                      f"src/pentaceratops/experimental/{name}.py"))
    pairs.append((root / "LICENSE", "LICENSE"))
    for source, destination in pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        if (project / destination).exists():
            raise FileExistsError(f"Will not overwrite {project / destination}")
    rows = []
    for source, destination in pairs:
        original = source.read_bytes()
        data = original
        if source.suffix == ".py":
            relative = destination.removeprefix("src/pentaceratops/")
            text = relocate(original.decode(), relative) if source.name in ENGINE else original.decode()
            if relative.startswith("experimental/"):
                text = text.replace('REPO = Path(os.environ.get("PENTACERATOPS_ROOT", Path(__file__).resolve().parents[2] / "triceratops"))\nsys.path.insert(0, str(REPO))\n', '')
                for old, new in {
                    "from triceratops import likelihoods as": "from ..likelihoods import real as",
                    "from triceratops.triceratops_new import": "from .._target import",
                    "from triceratops import marginal_likelihoods_new as": "from ..evidence import real as",
                    "from triceratops import marginal_likelihoods_eb as": "from ..evidence import eclipses as",
                    "from correlated_likelihood import": "from .covariance import",
                    "from physical_batch import": "from .physical_batch import",
                    "from sparse_kernels import": "from .sparse_kernels import",
                    "from batched_priors import": "from .batched_priors import",
                    "from v2_adapter import": "from .v2_adapter import",
                }.items():
                    text = text.replace(old, new)
            ast.parse(text)
            data = text.encode()
        output = project / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        rows.append(dict(source=str(source), destination=destination,
                         source_sha256=sha(original), imported_sha256=sha(data),
                         bytes=len(data)))
    manifest = dict(
        source_git_head=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        source_git_status=subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True),
        note="Imported working-tree files, including uncommitted corrections; old checkout untouched.",
        files=rows)
    (project / "docs").mkdir(exist_ok=True)
    (project / "docs/source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Imported {len(rows)} files into {project}; recorded hashes and source Git state.")


if __name__ == "__main__":
    main()
