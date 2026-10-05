"""One-call offline evidence from a JSON configuration and prepared local files."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import hz
from .preprocessing.folded import FoldedFourierData, load_folded_real
from .results import RunResult


COMMON_DEFAULTS = dict(N=500, steps=50, seed=42, posterior_samples=2000,
                       parity="profile", eb_eta=1., missing_host_policy="error")
COMMON_OPTIONS = set(COMMON_DEFAULTS) | {"nsamples", "scenarios", "filt"}
REAL_OPTIONS = {"primary_only", "include_gp", "timing_policy"}


def _keys(mapping, allowed, label):
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a JSON object")
    unknown = set(mapping)-set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}")


def _path(value, base):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("File paths must be nonempty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base/path).resolve()


def _fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def _field(path, mission, target_id):
    stars = pd.read_csv(path)
    required = {"ID", "mass", "rad", "Teff", "plx", "Tmag", "Jmag", "Hmag", "Kmag",
                "tdepth", "fluxratio"}
    if not required.issubset(stars):
        raise ValueError(f"Stellar CSV is missing columns: {sorted(required-set(stars))}")
    if stars.empty or stars.ID.duplicated().any():
        raise ValueError("Stellar CSV must contain unique hosts, with the target first")
    if (isinstance(target_id, bool) or not isinstance(target_id, int) or target_id <= 0
            or not np.all(np.isfinite(stars.ID) & (stars.ID > 0) & (stars.ID == np.floor(stars.ID)))):
        raise ValueError("target_id and stellar IDs must be positive integers")
    if int(stars.iloc[0].ID) != target_id or not stars.iloc[0].tdepth > 0:
        raise ValueError("First stellar row must be the eligible configured target")
    eligible = stars[stars.tdepth > 0]
    for key in ("plx", "Tmag", "Jmag", "Hmag", "Kmag", "fluxratio"):
        if not np.isfinite(eligible[key]).all():
            raise ValueError(f"Require finite {key} for every eligible host")
    if not ((eligible.fluxratio > 0) & (eligible.fluxratio <= 1)).all():
        raise ValueError("Eligible host fluxratio must be in (0, 1]")
    return SimpleNamespace(mission=mission, stars=stars, ID=target_id)


def run_prepared(config_path, *, likelihood=None, output_path=None, **overrides):
    """Run all eligible evidence scenarios from a local JSON configuration.

    Paths in the config are relative to its directory. An explicit output_path
    override is relative to the caller's working directory. Other overrides
    replace config settings, which replace the recorded HZ API defaults.
    No catalogs, PSD fits, or source-population downloads are performed.
    """
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text())
    _keys(config, {"schema_version", "mission", "target_id", "stars", "trilegal",
                   "molusc", "input", "likelihood", "settings", "output"}, "configuration")
    if config.get("schema_version") != 1:
        raise ValueError("Require evidence configuration schema_version=1")
    missing = {"mission", "target_id", "stars", "trilegal", "input"}-set(config)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    mission = config["mission"]
    if mission not in ("TESS", "Kepler", "K2"):
        raise ValueError("mission must be TESS, Kepler, or K2")
    domain = config.get("likelihood", "real") if likelihood is None else likelihood
    if domain not in ("real", "fourier"):
        raise ValueError("likelihood must be real or fourier")
    allowed = COMMON_OPTIONS | (REAL_OPTIONS if domain == "real" else set())
    settings = config.get("settings", {})
    _keys(settings, allowed, "settings")
    _keys(overrides, allowed, "overrides")
    options = dict(COMMON_DEFAULTS, nsamples=20 if domain == "real" else 7)
    if domain == "real":
        options.update(primary_only=False, include_gp=True, timing_policy="observed")
    options.update(settings)
    options.update(overrides)
    for key in ("N", "steps", "nsamples", "posterior_samples"):
        hz._positive_integer(options[key], key)
    if (isinstance(options["seed"], bool) or not isinstance(options["seed"], int)
            or not 0 <= options["seed"] < 2**32):
        raise ValueError("seed must be a nonnegative 32-bit integer")
    if not np.isfinite(options["eb_eta"]) or options["eb_eta"] < 0:
        raise ValueError("eb_eta must be finite and nonnegative")
    if options["parity"] not in ("profile", "marginalize"):
        raise ValueError("parity must be profile or marginalize")
    if options["missing_host_policy"] not in ("error", "solar"):
        raise ValueError("missing_host_policy must be error or solar")
    if domain == "real":
        for key in ("primary_only", "include_gp"):
            if not isinstance(options[key], bool):
                raise ValueError(f"{key} must be a boolean")
        if options["timing_policy"] not in ("observed", "legacy"):
            raise ValueError("timing_policy must be observed or legacy")
    selected = options.get("scenarios")
    if selected is not None and (not isinstance(selected, (list, tuple)) or not selected
            or any(not isinstance(s, str) for s in selected)
            or len(set(selected)) != len(selected)
            or set(selected)-set((*hz.FAMILIES, "NTP", "NEB", "NEBx2P"))):
        raise ValueError("scenarios must contain unique supported labels")
    base = config_path.parent
    if output_path is None and "output" not in config:
        raise ValueError("Supply output in the config or --output on the command line")
    destination = _path(config["output"], base) if output_path is None else _path(output_path, Path.cwd())
    RunResult.check_destination(destination)
    inputs = config["input"]
    _keys(inputs, {"format", "path", "prepared", "posterior", "directory"}, "input")
    kind = inputs.get("format")
    required = {"candidate": {"path"}, "folded_real": {"prepared", "posterior"},
                "folded_fourier": {"directory"}}.get(kind)
    if required is None or set(inputs) != {"format"} | required:
        raise ValueError("input requires format=candidate/path, folded_real/prepared/posterior, "
                         "or folded_fourier/directory")
    if kind != "candidate" and kind != "folded_"+domain:
        raise ValueError("Input format does not match the requested likelihood")
    if kind == "candidate" and options.get("primary_only"):
        raise ValueError("Candidate bundles contain secondary data; use explicit primary-only folded_real inputs")
    paths = {name: _path(inputs[name], base) for name in required}
    star_path, population = (_path(config[key], base) for key in ("stars", "trilegal"))
    molusc = None if config.get("molusc") is None else _path(config["molusc"], base)
    files = [config_path, star_path, population] + ([] if molusc is None else [molusc])
    if kind == "folded_fourier":
        directory = paths["directory"]
        files += [directory/"fold.json", directory/"fold_map.npz", directory/"block/block.json"]
        files += [directory/"block"/(key+".npy")
                  for key in ("time", "flux", "precision", "projected_residual", "sigma")]
    else:
        files += list(paths.values())
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Required evidence input missing: {path}")
    target = _field(star_path, mission, config["target_id"])
    for index, star in target.stars.iterrows():
        if star.tdepth > 0:
            for key in ("mass", "rad", "Teff"):
                if (not np.isfinite(star[key]) or star[key] <= 0) and (
                        index == 0 or options["missing_host_policy"] == "error"):
                    raise ValueError(f"Invalid {key} for host {star.ID}; supply stellar inputs")
    provenance = [_fingerprint(path) for path in files]
    if kind == "candidate":
        from .preprocessing.evidence_inputs import candidate_folded_real, candidate_folded_fourier
        saved = RunResult.load(paths["path"])
        candidate = saved.metadata.get("settings", {}).get("candidate", {})
        if candidate.get("mission") != mission or candidate.get("host_id") != config["target_id"]:
            raise ValueError("Prepared candidate and configured target/mission do not match")
        prepare = candidate_folded_real if domain == "real" else candidate_folded_fourier
        data = prepare(saved)
    elif kind == "folded_real":
        data = load_folded_real(paths["prepared"], paths["posterior"],
                                primary_only=options["primary_only"])
    else:
        data = FoldedFourierData.load(paths["directory"])
    function = hz.calc_probs_folded_real if domain == "real" else hz.calc_probs_folded_fourier
    result = function(target, data, trilegal_fname=str(population),
                       molusc_file=None if molusc is None else str(molusc), **options)
    result.metadata["prepared_run"] = dict(schema_version=1, likelihood=domain, input_format=kind,
        configuration=config, resolved_settings=options, files=provenance, output_path=str(destination),
        preparation="fixed saved PSD; no detrending refit" if kind == "candidate" else "loaded prepared likelihood inputs")
    result.save(destination)
    return result
