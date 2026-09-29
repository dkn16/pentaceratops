"""Cached TOI/KOI ephemerides: NASA default, optional Bayesian TESS catalogue.

No network access at import. TIC/KIC identify hosts, TOI/KOI identify signals;
ambiguous hosts never silently select their first catalog row.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
BAYESIAN_TESS_URL = "https://bayesianexoplanets.github.io/tois.csv"
SCHEMAS = {
    "TESS": dict(table="toi", host="tid", signal="toi", period="pl_orbper",
                 epoch="pl_tranmid", duration="pl_trandurh", depth="pl_trandep",
                 disposition="tfopwg_disp", radius="st_rad", teff="st_teff"),
    "Kepler": dict(table="cumulative", host="kepid", signal="kepoi_name",
                   period="koi_period", epoch="koi_time0bk", duration="koi_duration",
                   depth="koi_depth", disposition="koi_disposition",
                   radius="koi_srad", teff="koi_steff"),
}


@dataclass(frozen=True)
class Identifier:
    mission: str
    kind: str
    value: str

    @property
    def label(self):
        return f"{self.kind} {self.value}"


def parse_identifier(text, mission=None):
    value = str(text).strip().upper()
    if value.isdigit() and mission in SCHEMAS:
        value = ("TIC " if mission == "TESS" else "KIC ") + value
    match = re.fullmatch(r"(TIC|KIC|TOI|KOI)[ -]*([0-9]+(?:\.[0-9]{1,2})?)", value)
    if match is None:
        # Archive spelling of KOI names.
        match = re.fullmatch(r"(K)([0-9]+\.[0-9]{1,2})", value)
    if match is None:
        raise ValueError("Use TIC 123, KIC 123, TOI-123.01 or KOI-123.01; arbitrary names are not resolved")
    kind, number = match.groups()
    kind = "KOI" if kind == "K" else kind
    found_mission = "TESS" if kind in ("TIC", "TOI") else "Kepler"
    if mission is not None and found_mission != mission:
        raise ValueError(f"{kind} is not a {mission} identifier")
    if kind in ("TIC", "KIC"):
        if not number.isdigit() or int(number) <= 0:
            raise ValueError("Host IDs must be positive integers")
        number = str(int(number))
    else:
        decimal = Decimal(number)
        if "." not in number or decimal <= 0 or decimal == int(decimal):
            raise ValueError("Include the candidate suffix, e.g. TOI-700.02")
        number = f"{decimal:.2f}"
    return Identifier(found_mission, kind, number)


def finite(value):
    try:
        answer = float(value)
    except (TypeError, ValueError):
        return None
    return answer if np.isfinite(answer) else None


def archive_query(query, cache_dir, *, offline=False):
    """Pin each successful query in a pickle-free cache; never refresh implicitly."""
    from ..results import RunResult
    key = hashlib.sha256(query.encode()).hexdigest()
    path = Path(cache_dir) / "catalog" / f"{key}.npz"
    if path.exists():
        saved = RunResult.load(path)
        if saved.metadata.get("query") != query:
            raise ValueError("Catalog cache/query mismatch")
        return saved.output, saved.metadata
    if offline:
        raise FileNotFoundError(f"Offline catalog cache missing: {path}")
    url = TAP_URL + "?" + urlencode({"query": query, "format": "json"})
    with urlopen(url, timeout=60) as response:
        payload = response.read(8 * 1024**2 + 1)
    if len(payload) > 8 * 1024**2:
        raise ValueError("Unexpectedly large target catalog response")
    rows = json.loads(payload)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Archive did not return a table; no cache was written")
    metadata = dict(query=query, url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    response_sha256=hashlib.sha256(payload).hexdigest())
    try:
        RunResult(rows, metadata=metadata).save(path)
    except FileExistsError:
        # Concurrent workers may have completed the same query. Use the winner.
        saved = RunResult.load(path)
        if saved.metadata.get("query") != query:
            raise ValueError("Catalog cache/query mismatch")
        return saved.output, saved.metadata
    return rows, metadata


def bayesian_rows(cache_dir, *, offline=False, catalog_file=None):
    """Read the site's published Epoch (BTJD) and Duration (days), not Phase/Tau."""
    from ..results import RunResult
    path = Path(cache_dir) / "catalog" / "bayesian_tess_v1.npz"
    if catalog_file is None and path.exists():
        saved = RunResult.load(path)
        return saved.output, saved.metadata
    if catalog_file is not None:
        source = Path(catalog_file).expanduser().resolve()
        if source.stat().st_size > 64 * 1024**2:
            raise ValueError("Unexpectedly large Bayesian catalogue")
        payload = source.read_bytes()
        origin = str(source)
    elif offline:
        raise FileNotFoundError(f"Offline Bayesian catalogue cache missing: {path}")
    else:
        with urlopen(BAYESIAN_TESS_URL, timeout=60) as response:
            payload = response.read(64 * 1024**2 + 1)
        origin = BAYESIAN_TESS_URL
    if len(payload) > 64 * 1024**2:
        raise ValueError("Unexpectedly large Bayesian catalogue")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {"TIC", "TOI", "Period", "Epoch", "Duration"}
    if not required.issubset(reader.fieldnames or ()):
        raise ValueError("Bayesian catalogue requires TIC, TOI, Period, Epoch (BTJD), Duration (days). "
                         "Phase and Tau alone are not an absolute ephemeris.")
    rows = list(reader)
    provenance = dict(source="bayesian", url=origin, response_sha256=hashlib.sha256(payload).hexdigest(),
                      retrieved_utc=datetime.now(timezone.utc).isoformat(),
                      epoch_convention="Epoch = BJD - 2457000", duration_convention="Duration in days")
    if catalog_file is None:
        try:
            RunResult(rows, metadata=provenance).save(path)
        except FileExistsError:
            saved = RunResult.load(path)
            return saved.output, saved.metadata
    else:
        # A local snapshot is identified by its content, not the time it is read.
        provenance.pop("retrieved_utc")
    return rows, provenance


def catalog_rows(identifier, cache_dir, *, offline=False, source="nasa", catalog_file=None):
    if source == "bayesian":
        if identifier.mission != "TESS":
            raise ValueError("The optional Bayesian catalogue is currently TESS-only")
        rows, provenance = bayesian_rows(cache_dir, offline=offline, catalog_file=catalog_file)
        selected = []
        for row in rows:
            tic, toi = finite(row.get("TIC")), finite(row.get("TOI"))
            if tic is None or toi is None:
                continue
            match = int(tic) == int(identifier.value) if identifier.kind == "TIC" else f"{toi:.2f}" == identifier.value
            if not match:
                continue
            epoch, duration = finite(row.get("Epoch")), finite(row.get("Duration"))
            if epoch is not None and not 0 < epoch < 100000:
                raise ValueError("Bayesian Epoch must be BTJD; refusing to guess another time convention")
            selected.append(dict(tid=int(tic), toi=f"{toi:.2f}", pl_orbper=finite(row.get("Period")),
                pl_tranmid=None if epoch is None else epoch+2457000.,
                pl_trandurh=None if duration is None else duration*24., pl_trandep=None,
                _mask_companion=True, _bayesian_row=row))
        return selected, provenance
    if source != "nasa":
        raise ValueError(f"Unknown catalogue source: {source}")
    if catalog_file is not None:
        raise ValueError("--catalog-file applies only to the optional Bayesian catalogue")
    schema = SCHEMAS[identifier.mission]
    columns = list(dict.fromkeys([*schema.values()][1:] + ["ra", "dec"]))
    # All SQL values originate in the strictly validated identifier above.
    if identifier.kind in ("TIC", "KIC"):
        condition = f"{schema['host']}={int(identifier.value)}"
    elif identifier.kind == "TOI":
        condition = f"toi='{identifier.value}'"
    else:
        archive_name = f"K{int(Decimal(identifier.value)):05d}.{identifier.value.split('.')[1]}"
        condition = f"kepoi_name='{archive_name}'"
    query = f"select {','.join(columns)} from {schema['table']} where {condition}"
    rows, provenance = archive_query(query, cache_dir, offline=offline)
    return rows, provenance


def signal_identifier(row, mission):
    schema = SCHEMAS[mission]
    value = str(row[schema["signal"]])
    return parse_identifier(("TOI " if mission == "TESS" else "") + value, mission)


@dataclass(frozen=True)
class Candidate:
    mission: str
    host_id: int
    identifier: str
    period_days: float
    epoch_bjd: float
    duration_hours: float
    depth_ppm: float | None = None

    @property
    def tag(self):
        host = "TIC" if self.mission == "TESS" else "KIC"
        return f"{host}{self.host_id}_{self.identifier.replace(' ', '').replace('-', '')}"

    @property
    def time_zero(self):
        return 2457000.0 if self.mission == "TESS" else 2454833.0

    @property
    def epoch_relative(self):
        return self.epoch_bjd - self.time_zero

    @property
    def duration_days(self):
        return self.duration_hours / 24.0


def candidate_from_row(row, mission, host_id, overrides=None, identifier=None):
    schema = SCHEMAS[mission]
    overrides = {} if overrides is None else overrides
    unknown = set(overrides) - {"period_days", "epoch_bjd", "duration_hours", "depth_ppm"}
    if unknown:
        raise ValueError(f"Unknown ephemeris overrides: {sorted(unknown)}")
    epoch = finite(row.get(schema["epoch"]))
    if epoch is not None and mission == "Kepler":
        epoch += 2454833.0  # KOI time0bk is BKJD, not BTJD or full BJD.
    values = dict(period_days=finite(row.get(schema["period"])), epoch_bjd=epoch,
                  duration_hours=finite(row.get(schema["duration"])),
                  depth_ppm=finite(row.get(schema["depth"])))
    values.update({key: finite(value) for key, value in overrides.items() if value is not None})
    missing = [key for key in ("period_days", "epoch_bjd", "duration_hours") if values[key] is None]
    if missing:
        raise ValueError(f"Missing ephemeris {missing}; supply explicit overrides (epoch in full BJD)")
    if values["period_days"] <= 0 or not 0 < values["duration_hours"] / 24 < values["period_days"] / 2:
        raise ValueError("Require positive period and a duration shorter than half the period")
    if not 2400000 < values["epoch_bjd"] < 2600000:
        raise ValueError("epoch_bjd must be full BJD, not BTJD/BKJD")
    if values["depth_ppm"] is not None and not 0 < values["depth_ppm"] < 1e6:
        raise ValueError("depth_ppm must be in (0, 1000000)")
    label = identifier or signal_identifier(row, mission).label
    return Candidate(mission, int(host_id), label, **values)


def resolve_candidate(target, cache_dir, *, mission=None, candidate=None, overrides=None,
                      offline=False, mask_other_planets=True, source="nasa", catalog_file=None):
    """Return candidate, companion ephemerides, and complete catalog provenance."""
    ident = parse_identifier(target, mission)
    schema = SCHEMAS[ident.mission]
    rows, initial_query = catalog_rows(ident, cache_dir, offline=offline, source=source, catalog_file=catalog_file)
    queries = [initial_query]
    if ident.kind in ("TOI", "KOI"):
        if len(rows) != 1:
            raise ValueError(f"Expected one archive row for {ident.label}; found {len(rows)}")
        host_id = int(rows[0][schema["host"]])
        host = parse_identifier(("TIC " if ident.mission == "TESS" else "KIC ") + str(host_id))
        all_rows, provenance = catalog_rows(host, cache_dir, offline=offline, source=source, catalog_file=catalog_file)
        queries.append(provenance)
        if candidate is not None and parse_identifier(candidate, ident.mission) != ident:
            raise ValueError("--candidate conflicts with the target signal ID")
    else:
        host_id = int(ident.value)
        all_rows = rows
        if candidate is not None:
            selection = parse_identifier(candidate, ident.mission)
            if selection.kind in ("TIC", "KIC"):
                raise ValueError("--candidate must be a TOI or KOI signal ID")
            rows = [row for row in all_rows if signal_identifier(row, ident.mission) == selection]
            if len(rows) != 1:
                raise ValueError(f"{selection.label} is not uniquely associated with {ident.label}")
    if len(rows) > 1:
        choices = ", ".join(signal_identifier(row, ident.mission).label for row in rows)
        raise ValueError(f"{ident.label} has multiple candidates: {choices}. Pass --candidate.")
    row = rows[0] if rows else {}
    selected = candidate_from_row(row, ident.mission, host_id, overrides,
                                  identifier=None if row else f"{ident.kind} {ident.value} custom")
    companions = []
    if mask_other_planets:
        allowed = {"CP", "KP", "PC", "APC"} if ident.mission == "TESS" else {"CONFIRMED", "CANDIDATE"}
        for other in all_rows:
            if signal_identifier(other, ident.mission).label == selected.identifier:
                continue
            if other.get("_mask_companion") or str(other.get(schema["disposition"], "")).upper() in allowed:
                companions.append(candidate_from_row(other, ident.mission, host_id))
    if source == "bayesian" and row and str(row["_bayesian_row"].get("passed_all_tests", "")).lower() == "false":
        import warnings
        warnings.warn(f"{selected.identifier} is flagged as failing tests in the Bayesian catalogue; "
                      "ephemeris selection is not a validation/FA classification")
    return selected, companions, dict(source=source, queries=queries, selected_row=row, host_rows=all_rows,
                                     overrides=overrides or {},
                                     companions=[asdict(item) for item in companions])
