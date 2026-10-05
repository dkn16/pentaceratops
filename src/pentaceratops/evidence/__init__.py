"""Direct prepared-data evidence and the individual scenario integrators.

Use ``from pentaceratops.evidence import evidence`` for the public Python call.
The real/Fourier submodules remain available for explicit low-level recipes.
"""


def __getattr__(name):
    if name == "evidence":
        from ..prepared import evidence
        globals()[name] = evidence
        return evidence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
