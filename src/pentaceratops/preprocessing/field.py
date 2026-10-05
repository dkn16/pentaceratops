"""Prepare and reload a TESS stellar field without running evidence."""
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path
import shutil
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from ..results import RunResult
from .aperture import geometry_from_fits, dilution_depths
from .flux import require_aperture_frame


STELLAR_COLUMNS = ("ID", "Tmag", "Jmag", "Hmag", "Kmag", "ra", "dec",
                   "mass", "rad", "Teff", "plx")
POPULATION_COLUMNS = ("Mact", "logg", "logTe", "[M/H]", "TESS", "J", "H", "Ks")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _query_stars(host_id, search_radius):
    try:
        from astroquery.mast import Catalogs
        import astropy.units as u
    except ImportError as exc:
        raise ImportError("Stellar catalog queries require pentaceratops[catalogs]") from exc
    catalog = Catalogs.query_object(f"TIC {host_id}", catalog="TIC",
                                   radius=search_radius * 20.25 * u.arcsec)
    return catalog.to_pandas()


def _ordered_stars(stars, host_id):
    stars = stars.copy(deep=True) if isinstance(stars, pd.DataFrame) else pd.read_csv(stars)
    missing = set(STELLAR_COLUMNS) - set(stars.columns)
    if missing:
        raise ValueError(f"Stellar catalog is missing columns: {sorted(missing)}")
    # Retain scientific columns and known catalog quality flags, not arbitrary
    # object-valued catalog columns that cannot be written portably to CSV.
    columns = [*STELLAR_COLUMNS, *[c for c in ("disposition", "duplicate_id") if c in stars]]
    stars = stars[columns].copy()
    for column in STELLAR_COLUMNS:
        stars[column] = pd.to_numeric(stars[column], errors="raise")
    ids = stars.ID.to_numpy()
    if (not np.isfinite(ids).all() or np.any(ids <= 0) or np.any(ids != np.floor(ids))
            or stars.ID.duplicated().any()):
        raise ValueError("Stellar IDs must be unique positive integers")
    stars["ID"] = stars.ID.astype("int64")
    host = stars.ID == host_id
    if host.sum() != 1:
        raise ValueError("Catalog must contain exactly one row for the target")
    stars = pd.concat([stars[host], stars[~host]], ignore_index=True)
    if not np.isfinite(stars[["ra", "dec", "Tmag"]].to_numpy()).all():
        raise ValueError("All sources require finite ra, dec and Tmag for aperture dilution")
    return stars


def _population_body(table):
    missing = set(POPULATION_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"TRILEGAL population is missing columns: {sorted(missing)}; "
                         "supply a TESS/2MASS population")
    numeric = table[list(POPULATION_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if numeric.empty or not np.isfinite(numeric.to_numpy()).all() or (numeric.Mact <= 0).any():
        raise ValueError("TRILEGAL population must contain finite stellar rows and positive masses")
    return numeric


def _validate_population(path):
    # Validate exactly the rows consumed by the existing reader ([:-2]). Some
    # adopted service CSVs have just one footer, so preserve their bytes and
    # historical row selection rather than rejecting or rewriting them here.
    table = pd.read_csv(path)
    if len(table) < 3:
        raise ValueError("TRILEGAL CSV has no stellar rows under the existing reader convention")
    _population_body(table.iloc[:-2])
    if not table.iloc[-1].astype(str).str.startswith("#TRILEGAL").any():
        raise ValueError("TRILEGAL CSV must retain its completion record")


def _download_population(ra, dec, destination, *, mag_lim, timeout, poll_interval):
    from ..stellar import query_TRILEGAL
    url = query_TRILEGAL(ra, dec, mag_lim=str(mag_lim))
    if url is None:
        raise RuntimeError("TRILEGAL is unavailable; retry or supply trilegal_fname")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("TRILEGAL population did not finish; retry or supply trilegal_fname")
        try:
            with urlopen(url, timeout=min(60., remaining)) as response:
                raw = response.read()
            text = raw.decode("utf-8")
            if text.rstrip().endswith("#TRILEGAL normally terminated"):
                break
        except (URLError, TimeoutError):
            pass
        time.sleep(min(poll_interval, max(0., deadline - time.monotonic())))
    table = pd.read_csv(io.StringIO(text), sep=r"\s+")
    # Strip only trailing non-stellar service records, not missing-value stars.
    while len(table) and str(table.iloc[-1, 0]).startswith("#"):
        table = table.iloc[:-1]
    numeric = _population_body(table)
    table = table.copy()
    for column in POPULATION_COLUMNS:
        table[column] = numeric[column]
    table["population_record"] = "star"
    footer = pd.DataFrame({"population_record": ["#TRILEGAL", "#TRILEGAL normally terminated"]})
    pd.concat([table, footer], ignore_index=True).to_csv(destination, index=False)
    _validate_population(destination)
    return dict(source="TRILEGAL", url=url, raw_sha256=hashlib.sha256(raw).hexdigest(),
                ra=ra, dec=dec, mag_lim=mag_lim)


def prepare_target(prepared, output_dir, *, stars=None, trilegal_fname=None,
                   transit_depth=None, search_radius=10, mag_lim=21.,
                   population_timeout=900., poll_interval=10.):
    """Create stars.csv and trilegal.csv and return an evidence-ready target.

    Accept a TESS candidate RunResult or saved NPZ from prepare_candidate.
    Query TIC unless stars is a DataFrame/CSV; calculate dilution using the
    original science FITS apertures. transit_depth optionally supplies a
    dimensionless aperture depth when absent from the saved bundle.
    Download TRILEGAL unless trilegal_fname supplies an existing population.
    Missing stellar inputs fail explicitly; no solar substitutions are made.

    output_dir must not exist. It receives stars.csv, trilegal.csv, and
    field.json (identity, input provenance, per-sector dilution and hashes).
    Use load_target(output_dir) for subsequent offline calls. This function
    does not change photometry, run evidence, or overwrite any existing field.
    population_timeout bounds result polling after the TRILEGAL form query.
    """
    destination = Path(output_dir).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Field already exists: {destination}; use load_target or a new directory")
    destination = destination.resolve()
    if isinstance(prepared, (str, Path)):
        prepared = RunResult.load(Path(prepared).expanduser())
    if not isinstance(prepared, RunResult) or prepared.metadata.get("artifact") != "candidate_preparation":
        raise TypeError("prepared must be a candidate RunResult or its NPZ path")
    settings = prepared.metadata["settings"]
    candidate = settings["candidate"]
    if candidate["mission"] != "TESS":
        raise ValueError("New stellar-field preparation currently supports TESS science FITS only")
    host_id = candidate["host_id"]
    if isinstance(host_id, bool) or not isinstance(host_id, Integral) or host_id <= 0:
        raise ValueError("Prepared TIC host ID must be a positive integer")
    for name, value in (("search_radius", search_radius), ("mag_lim", mag_lim),
                        ("population_timeout", population_timeout), ("poll_interval", poll_interval)):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    windows = prepared.output["windows"]
    require_aperture_frame(windows)
    depth = windows.get("transit_depth") if transit_depth is None else transit_depth
    if depth is None or not np.isfinite(depth) or not 0 < depth < 1:
        raise ValueError("Supply transit_depth as a dimensionless depth in the aperture flux frame")
    products = settings.get("native_products", [])
    if not products or len({p["sector"] for p in products}) != len(products):
        raise ValueError("Require one recorded science FITS product per selected sector")
    inputs = []
    for product in products:
        path = Path(product["source_file"]).expanduser().resolve()
        digest = _sha256(path)
        if product.get("fits_sha256", digest) != digest:
            raise ValueError(f"Science FITS changed since photometry preparation: {path}")
        inputs.append(dict(sector=int(product["sector"]), path=str(path), sha256=digest))
    population = None if trilegal_fname is None else Path(trilegal_fname).expanduser().resolve()
    if population is not None:
        _validate_population(population)
    source = "TIC query" if stars is None else "supplied table"
    catalog = _query_stars(host_id, search_radius) if stars is None else stars
    field = _ordered_stars(catalog, host_id)
    geometries = [geometry_from_fits(p["path"], field[["ra", "dec"]].to_numpy(),
                  tic=host_id, sector=p["sector"]) for p in inputs]
    fraction, host_depth, per_sector = dilution_depths(field.Tmag.to_numpy(), geometries, float(depth))
    field["fluxratio"], field["tdepth"] = fraction, host_depth
    from ..prepared import _field, _stellar_parameters
    host = _field(field, "TESS", int(host_id))
    _stellar_parameters(host, "error")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pentaceratops-field-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        field.to_csv(staging/"stars.csv", index=False)
        if population is None:
            population_info = _download_population(float(field.iloc[0].ra), float(field.iloc[0].dec),
                staging/"trilegal.csv", mag_lim=float(mag_lim), timeout=float(population_timeout),
                poll_interval=float(poll_interval))
        else:
            shutil.copyfile(population, staging/"trilegal.csv")
            population_info = dict(source="supplied file", path=str(population), sha256=_sha256(population))
        from .. import __version__
        metadata = dict(schema_version=1, package_version=__version__, mission="TESS", target_id=int(host_id),
            candidate=candidate, stellar_source=source, search_radius_pixels=float(search_radius),
            transit_depth=float(depth), depth_source="prepared" if transit_depth is None else "explicit",
            products=inputs, fluxratio_per_sector=per_sector.tolist(), population=population_info,
            sha256={name: _sha256(staging/name) for name in ("stars.csv", "trilegal.csv")})
        (staging/"field.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
        # Exclusive mkdir prevents concurrent preparations overwriting a field.
        # Publish the completion manifest last; load_target rejects incomplete directories.
        destination.mkdir()
        published = []
        try:
            for name in ("stars.csv", "trilegal.csv", "field.json"):
                (staging/name).rename(destination/name)
                published.append(destination/name)
        except BaseException:
            for path in published:
                path.unlink()
            destination.rmdir()
            raise
    return load_target(destination)


def load_target(directory):
    """Reload a prepare_target field, verifying saved file hashes, without network access."""
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory/"field.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("mission") != "TESS":
        raise ValueError("Unsupported stellar-field manifest")
    for name in ("stars.csv", "trilegal.csv"):
        if _sha256(directory/name) != metadata["sha256"][name]:
            raise ValueError(f"Saved {name} changed; prepare a new field with the intended inputs")
    from ..prepared import _field, _stellar_parameters
    target = _field(directory/"stars.csv", metadata["mission"], metadata["target_id"])
    _stellar_parameters(target, "error")
    _validate_population(directory/"trilegal.csv")
    target.trilegal_fname = str(directory/"trilegal.csv")
    target.stars_path = str(directory/"stars.csv")
    target.field_metadata = metadata
    return target
