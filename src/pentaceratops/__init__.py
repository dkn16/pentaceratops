"""Pentaceratops: real-space and Fourier-space transit validation.

Importing the package performs no catalogue queries or downloads.
"""

__version__ = "0.1.0.dev8"
__all__ = ["Target", "calc_probs_fourier", "run_evidence", "RunResult",
           "scenario_probabilities", "calc_probs_folded_real", "calc_probs_folded_fourier",
           "load_folded_real", "prepare_folded_fourier", "FoldedFourierData",
           "run_folded_baseline", "__version__"]


def __getattr__(name):
    if name == "run_folded_baseline":
        from .evidence.folded_baseline import run_folded_baseline
        return run_folded_baseline
    if name in ("calc_probs_folded_real", "calc_probs_folded_fourier"):
        from . import hz
        return getattr(hz, name)
    if name in ("load_folded_real", "prepare_folded_fourier", "FoldedFourierData"):
        from .preprocessing import folded
        return getattr(folded, name)
    if name == "run_evidence":
        from .api import run_evidence
        return run_evidence
    if name == "RunResult":
        from .results import RunResult
        return RunResult
    if name == "Target":
        from .api import Target
        return Target
    if name == "calc_probs_fourier":
        from .api import calc_probs_fourier
        return calc_probs_fourier
    if name == "scenario_probabilities":
        from .results import scenario_probabilities
        return scenario_probabilities
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
