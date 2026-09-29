"""Opt-in observation of sampler calls, without draws or numerical changes.

The physical sampler coordinates and importance weights are the authoritative
posterior representation. A log-likelihood value is NOT a normalized posterior
density. Evidence routines can include scenario-prior terms in their log target.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from functools import partial
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

_active = ContextVar("pentaceratops_sampling_record", default=None)
_event = ContextVar("pentaceratops_evidence_event", default=None)


def snapshot(value):
    """Copy numerical outputs so later engine mutations cannot change records."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    return value


def callable_name(function):
    """Identity for functions, partials, and callable instances without repr()."""
    if isinstance(function, partial):
        return f"functools.partial({callable_name(function.func)})"
    cls = type(function)
    return (f"{getattr(function, '__module__', cls.__module__)}."
            f"{getattr(function, '__qualname__', cls.__qualname__)}")


def describe_input(value):
    """Retain numeric inputs; describe (never pickle) callback functions."""
    if callable(value):
        return {"callable": callable_name(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: describe_input(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe_input(item) for item in value]
    return snapshot(value)


@contextmanager
def capture_sampling():
    """Capture complete weighted pools and per-function physical results.

    The list is populated as calls complete, even if a subsequent call fails.
    Context-local recording does not make the legacy numerical engine thread-safe.
    """
    records = []
    token = _active.set(records)
    event_token = _event.set(None)
    try:
        yield records
    finally:
        _event.reset(event_token)
        _active.reset(token)


def record_sampler(function):
    signature = inspect.signature(function)

    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        records = _active.get()
        if records is not None:
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            values = arguments.arguments
            evidence, weights, positions, log_values = result
            record = {
                "kind": "sampler",
                "event": _event.get(),
                "log_evidence": float(evidence),
                "samples": np.array(positions, copy=True),
                "weights": np.array(weights, copy=True),
                "log_target": np.array(log_values, copy=True),
                "log_target_definition": "Value passed to persistent sampling; not log posterior",
                "n_active": int(values["n_active"]),
                "ndim": int(values["ndim"]),
                "mcmc_steps": int(values["mcmc_steps"]),
                "target_ess": values["target_ess"],
                "log_target_function": callable_name(values["log_likelihood_function"]),
                "prior_transform_function": callable_name(values["prior_transform"]),
                "finite_result": bool(np.isfinite(evidence)),
            }
            records.append(record)
        return result

    return wrapper


def record_evidence(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapper(*args, **kwargs):
        records = _active.get()
        if records is None:
            return function(*args, **kwargs)
        event = len(records)
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        entry = {"kind": "evidence", "event": event,
                 "function": f"{function.__module__}.{function.__name__}",
                 "inputs": describe_input(dict(arguments.arguments)),
                 "status": "running"}
        records.append(entry)
        token = _event.set(event)
        try:
            result = function(*args, **kwargs)
            entry.update(status="complete", result=snapshot(result))
            return result
        except Exception as exc:
            entry.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            _event.reset(token)

    return wrapper


def instrument_evidence(namespace):
    """Observe this module's public evidence functions, not imported aliases."""
    for name, function in tuple(namespace.items()):
        if (name.startswith("lnZ_") and inspect.isfunction(function)
                and function.__module__ == namespace["__name__"]):
            namespace[name] = record_evidence(function)
