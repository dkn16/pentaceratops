"""Shared, provisional sampling policy for Pentaceratops and v3.

Defaults are based on the two-target, three-seed v3 sweep of 2026-09-15.
They are computational choices, not guarantees of evidence precision.
Explicit N/steps independently override the corresponding policy field.
"""
from numbers import Integral

POLICY_VERSION = "scenario-defaults-2026-09-15"
SCENARIOS = (
    "TP", "EB", "EBx2P", "PTP", "PEB", "PEBx2P", "STP", "SEB",
    "SEBx2P", "DTP", "DEB", "DEBx2P", "BTP", "BEB", "BEBx2P",
    "NTP", "NEB", "NEBx2P",
)
_RECOMMENDED = {
    "TP": (100, 20), "PTP": (100, 20), "DTP": (100, 20),
    "NTP": (100, 20), "SEBx2P": (100, 20),
    "SEB": (200, 50), "BEBx2P": (200, 50), "NEB": (200, 30),
    "BEB": (500, 50),
}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return int(value)


def sampling_config(scenario, N=None, steps=None):
    """Return a fresh, JSON-safe configuration for a scenario label."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown sampling scenario: {scenario!r}")
    if scenario in _RECOMMENDED:
        n_default, steps_default = _RECOMMENDED[scenario]
        basis = "shared_sweep_choice"
    elif scenario == "STP":
        n_default, steps_default = 500, 50
        basis = "user_STP_fallback_unvalidated"
    else:
        n_default, steps_default = 200, 30
        basis = "user_fallback_unvalidated"
    return {
        "N": n_default if N is None else _positive_integer(N, "N"),
        "steps": steps_default if steps is None else _positive_integer(steps, "steps"),
        "basis": basis,
        "N_overridden": N is not None,
        "steps_overridden": steps is not None,
    }


def sampling_policy(N=None, steps=None):
    """Resolve all families, including nearby-source scenarios."""
    return {name: sampling_config(name, N, steps) for name in SCENARIOS}
