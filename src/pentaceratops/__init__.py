"""Pentaceratops: real-space and Fourier-space transit validation.

Importing the package performs no catalogue queries or downloads.
"""

__version__ = "0.1.0.dev3"
__all__ = ["Target", "calc_probs_fourier", "run_evidence", "RunResult",
           "scenario_probabilities", "__version__"]


def __getattr__(name):
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
