"""Explicit scenario weighting and portable, pickle-free result bundles."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .sampling.policy import SCENARIOS

_PLANET_ON_TARGET = {"TP", "PTP", "DTP"}
_NEARBY = {"NTP", "NEB", "NEBx2P"}
_ECLIPSES = {name for name in SCENARIOS if "EB" in name}


def scenario_probabilities(table, *, evidence_column, eb_eta=1.0):
    """Normalize comparable log evidences with one explicit EB odds multiplier.

    The caller MUST choose a comparable evidence column. This function cannot
    repair likelihood-frame/data-set mismatches. ``eb_eta`` is an absolute
    multiplier applied to the supplied *unweighted* column, never to old ``prob``
    or ``eta`` values. All rows (including different nearby stars) remain distinct.
    """
    if not np.isfinite(eb_eta) or eb_eta < 0:
        raise ValueError("eb_eta must be finite and nonnegative")
    result = pd.DataFrame(table).copy(deep=True)
    if result.empty:
        raise ValueError("No scenarios supplied")
    labels = result["scenario"]
    unknown = set(labels) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenario labels: {sorted(unknown, key=str)}")
    values = result[evidence_column].to_numpy(dtype=float)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Evidence must be finite or -inf; NaN/+inf cannot be normalized")
    if not np.isfinite(values).any():
        raise ValueError("All scenarios have zero evidence; FPP is undefined")
    # Remove a large common offset before adding small prior log odds.
    values = values - np.max(values[np.isfinite(values)])
    is_eb = labels.isin(_ECLIPSES).to_numpy()
    log_odds = np.zeros(len(result))
    log_odds[is_eb] = np.log(eb_eta) if eb_eta > 0 else -np.inf
    log_weights = values + log_odds
    if not np.isfinite(log_weights).any():
        raise ValueError("All scenarios have zero evidence or prior weight; FPP is undefined")
    result["prob"] = np.exp(log_weights - logsumexp(log_weights))
    result["eta"] = np.where(is_eb, eb_eta, 1.0)
    result.attrs = {
        "FPP": float(result.loc[~labels.isin(_PLANET_ON_TARGET), "prob"].sum()),
        "FPP_EB": float(result.loc[is_eb, "prob"].sum()),
        "NFPP": float(result.loc[labels.isin(_NEARBY), "prob"].sum()),
        "eb_eta": float(eb_eta), "evidence_column": evidence_column,
    }
    return result


def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not allowed in result bundles")
        key = f"array_{len(arrays):06d}"
        arrays[key] = value
        return {"type": "array", "key": key}
    if isinstance(value, np.generic):
        return _pack(value.item(), arrays)
    if isinstance(value, pd.DataFrame):
        if isinstance(value.index, pd.MultiIndex):
            raise TypeError("MultiIndex tables are not supported in result bundles")
        return {"type": "dataframe", "columns": list(value.columns),
                "data": _pack(value.to_dict(orient="list"), arrays),
                "index": _pack(value.index.tolist(), arrays),
                "index_name": _pack(value.index.name, arrays),
                "attrs": _pack(value.attrs, arrays)}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Result dictionary keys must be strings")
        return {"type": "dict", "items": {key: _pack(item, arrays)
                                            for key, item in value.items()}}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_pack(item, arrays) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_pack(item, arrays) for item in value]}
    if isinstance(value, float) and not np.isfinite(value):
        return {"type": "nonfinite", "value": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}; supply plain metadata")


def _unpack(value, arrays):
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "array":
        return arrays[value["key"]].copy()
    if kind == "nonfinite":
        return float(value["value"])
    if kind == "list":
        return [_unpack(item, arrays) for item in value["items"]]
    if kind == "tuple":
        return tuple(_unpack(item, arrays) for item in value["items"])
    if kind == "dict":
        return {key: _unpack(item, arrays) for key, item in value["items"].items()}
    if kind == "dataframe":
        index = _unpack(value["index"], arrays) if "index" in value else None
        frame = pd.DataFrame(_unpack(value["data"], arrays), columns=value["columns"], index=index)
        frame.index.name = _unpack(value.get("index_name"), arrays)
        frame.attrs = _unpack(value["attrs"], arrays)
        return frame
    raise ValueError(f"Unknown bundle value type: {kind}")


@dataclass
class RunResult:
    """Numerical output and complete sampler pools from one recorded invocation."""

    output: object
    sampling: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def check_destination(path):
        """Reject collisions/invalid suffix before starting expensive sampling."""
        path = Path(path).expanduser()
        if path.suffix != ".npz":
            raise ValueError("Result bundle path must end in .npz")
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        return path

    def save(self, path):
        """Atomically publish ONE compressed NPZ bundle; never overwrite.

        Use a scratch path for large posteriors. No default output directory or
        machine-specific path is inferred. A bundle contains arrays and UTF-8
        JSON metadata, and is loadable with ``allow_pickle=False``.
        """
        path = self.check_destination(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        payload = {"schema_version": 1, "output": self.output,
                   "sampling": self.sampling, "metadata": self.metadata}
        manifest = json.dumps(_pack(payload, arrays), allow_nan=False)
        arrays["manifest"] = np.frombuffer(manifest.encode("utf-8"), dtype=np.uint8)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".penta-", suffix=".npz",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            # Atomic no-clobber publication; temp and final are on the same FS.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            manifest = json.loads(arrays["manifest"].tobytes().decode("utf-8"))
            payload = _unpack(manifest, arrays)
        if payload.pop("schema_version") != 1:
            raise ValueError("Unsupported result bundle schema")
        return cls(**payload)
