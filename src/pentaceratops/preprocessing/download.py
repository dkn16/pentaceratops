"""Select and cache one official PDCSAP product per TESS sector/Kepler quarter."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .flux import lightcurve_flux_frame, restore_aperture_flux


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def product_selection(table, mission, segments=None, cadence_seconds=None):
    """Choose before downloading, so duplicates/unused sectors are not fetched."""
    choices = {}
    for index, row in enumerate(table):
        match = re.search(r"(?:Sector|Quarter)\s+(\d+)", str(row["mission"]), re.I)
        if match is None:
            continue
        segment = int(match.group(1))
        if segments is not None and segment not in segments:
            continue
        author = str(row["author"])
        allowed = ("SPOC", "TESS-SPOC") if mission == "TESS" else ("Kepler",)
        if author not in allowed:
            continue
        exposure = float(row["exptime"])
        if not np.isfinite(exposure) or exposure <= 0:
            continue
        if cadence_seconds is not None and not np.isclose(exposure, cadence_seconds, rtol=.05):
            continue
        if cadence_seconds is None and exposure < (60 if mission == "TESS" else 1000):
            continue  # Avoid TESS 20-s and Kepler monthly short-cadence duplicates.
        preferred = 120. if mission == "TESS" else 1800.
        rank = (allowed.index(author), abs(exposure - preferred), str(row["productFilename"]))
        if segment not in choices or rank < choices[segment][0]:
            choices[segment] = (rank, index)
    if segments is not None and set(segments) - set(choices):
        raise ValueError(f"No matching products for requested segments {sorted(set(segments)-set(choices))}")
    if not choices:
        raise ValueError("No supported official light curves match this target/cadence selection")
    return [(segment, choices[segment][1]) for segment in sorted(choices)]


def normalize_product(lc, candidate, segment, exposure_seconds):
    """Read full BJD explicitly and restore dilution exactly once, per product."""
    mission = candidate.mission
    meta_id = lc.meta.get("TICID" if mission == "TESS" else "KEPLERID")
    if meta_id is None or int(meta_id) != candidate.host_id:
        raise ValueError("Downloaded FITS target ID does not match requested host")
    meta_segment = lc.meta.get("SECTOR" if mission == "TESS" else "QUARTER")
    if meta_segment is None or int(meta_segment) != segment:
        raise ValueError("Downloaded FITS sector/quarter does not match selected product")
    time = np.asarray(lc.time.tdb.jd, float) - candidate.time_zero
    flux = np.asarray(lc.flux.value, float)
    error = np.asarray(lc.flux_err.value, float)
    good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    time, flux, error = time[good], flux[good], error[good]
    if len(time) < 2:
        raise ValueError("No usable photometry after quality/finite-error filtering")
    order = np.argsort(time, kind="stable")
    time, flux, error = time[order], flux[order], error[order]
    if np.any(np.diff(time) <= 0):
        raise ValueError("Duplicate timestamps within a selected product")
    baseline = float(np.median(flux))
    if baseline <= 0 or not np.isfinite(baseline):
        raise ValueError("Invalid product flux baseline")
    frame = lightcurve_flux_frame(lc)
    flux, error = restore_aperture_flux(flux/baseline, error/baseline,
                                      frame["crowding_restore_factor"])
    return dict(time=time, flux=flux, error=error, segment=int(segment),
                exposure_seconds=float(exposure_seconds)), dict(
                    segment=int(segment), sector=int(segment), **frame,
                    normalization=baseline, n_points=len(time), time_scale="TDB",
                    time_zero_bjd=candidate.time_zero, exposure_seconds=float(exposure_seconds))


def download_products(candidate, cache_dir, *, segments=None, cadence_seconds=None, offline=False):
    """Download official products, or reuse a pinned numeric bundle without MAST."""
    from ..results import RunResult
    settings = dict(schema=1, mission=candidate.mission, host_id=candidate.host_id,
                    segments=None if segments is None else sorted(set(map(int, segments))),
                    cadence_seconds=cadence_seconds, quality="default", flux="pdcsap_flux")
    key = fingerprint(settings)
    path = Path(cache_dir) / "native" / f"{candidate.mission}_{candidate.host_id}_{key[:16]}.npz"
    if path.exists():
        saved = RunResult.load(path)
        if saved.metadata["selection"] != settings:
            raise ValueError("Native cache selection mismatch")
        return saved.output, saved.metadata, path
    if offline:
        raise FileNotFoundError(f"Offline native-photometry cache missing: {path}")
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise ImportError("Download examples require pip install -e '.[catalogs]'") from exc
    prefix = "TIC" if candidate.mission == "TESS" else "KIC"
    kwargs = dict(mission=candidate.mission,
                  author=["SPOC", "TESS-SPOC"] if candidate.mission == "TESS" else "Kepler")
    if segments is not None:
        kwargs["sector" if candidate.mission == "TESS" else "quarter"] = list(segments)
    if cadence_seconds is not None:
        kwargs["exptime"] = cadence_seconds
    elif candidate.mission == "Kepler":
        kwargs["exptime"] = "long"
    search = lk.search_lightcurve(f"{prefix} {candidate.host_id}", **kwargs)
    selected = product_selection(search.table, candidate.mission, segments, cadence_seconds)
    blocks, products = [], []
    download_dir = Path(cache_dir) / "lightkurve"
    download_dir.mkdir(parents=True, exist_ok=True)
    for segment, index in selected:
        print(f"Downloading {prefix} {candidate.host_id}, segment {segment}", flush=True)
        row = search.table[index]
        lc = search[index].download(download_dir=str(download_dir), quality_bitmask="default",
                                    flux_column="pdcsap_flux")
        if lc is None:
            raise RuntimeError(f"Failed to download segment {segment}; no partial native cache published")
        block, product = normalize_product(lc, candidate, segment, float(row["exptime"]))
        product.update(author=str(row["author"]), product_filename=str(row["productFilename"]))
        filename = Path(product["source_file"])
        if filename.is_file():
            digest = hashlib.sha256()
            with filename.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024**2), b""):
                    digest.update(chunk)
            product["fits_sha256"] = digest.hexdigest()
        blocks.append(block)
        products.append(product)
    metadata = dict(selection=settings, products=products)
    try:
        RunResult(blocks, metadata=metadata).save(path)
    except FileExistsError:
        saved = RunResult.load(path)
        if saved.metadata["selection"] != settings:
            raise ValueError("Native cache selection mismatch")
        return saved.output, saved.metadata, path
    return blocks, metadata, path
