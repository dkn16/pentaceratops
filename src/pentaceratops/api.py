"""Recorded compatibility API for the extracted research engine.

This layer deliberately does not change likelihood-frame conventions, masks,
PSDs or dilution. Benchmark-specific orchestration is not yet a turnkey API.
"""

from contextlib import contextmanager
from functools import wraps
from functools import lru_cache
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import warnings

import numpy as np

from . import __version__
from .results import RunResult
from .sampling.records import callable_name, capture_sampling, snapshot


@lru_cache(maxsize=1)
def _source_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def _seeded(seed):
    if seed is None:
        yield
        return
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def run_evidence(function, *args, output_path=None, seed=None, metadata=None, **kwargs):
    """Call an engine function and retain all weighted sampler pools by default.

    ``output_path`` explicitly selects a scratch NPZ file; omitted means memory
    only. This does not increase equal-weight resampling counts (or consume extra
    random numbers). ``seed`` restores the caller's RNG state after the run.
    Parallel jobs must use separate processes, not threads.
    """
    if output_path is not None:
        RunResult.check_destination(output_path)
    details = dict(metadata or {})
    details.update(package_version=__version__, python=platform.python_version(),
                   numpy=np.__version__, seed=seed,
                   scipy=version("scipy"), source_sha256=_source_fingerprint(),
                   pytransit=version("pytransit"), meepmeep=version("meepmeep"),
                   numba=version("numba"),
                   function=callable_name(function))
    with _seeded(seed), capture_sampling() as records:
        output = function(*args, **kwargs)
    result = RunResult(snapshot(output), records, details)
    if output_path is not None:
        result.save(output_path)
    return result


def calc_probs_fourier(*args, output_path=None, seed=None, **kwargs):
    """Return a RunResult containing the Fourier scenario table and full pools.

    All numerical arguments follow ``pentaceratops.fourier.calc_probs_fourier``.
    Its returned table has explicit null-referenced ``lnBF`` and FPP attributes.
    """
    from .fourier import calc_probs_fourier as calculate
    return run_evidence(calculate, *args, output_path=output_path, seed=seed, **kwargs)


def _target_class():
    from ._target import target

    class Target(target):
        """Catalog-backed compatibility target; construction can query services.

        This is not the fully corrected benchmark runner. Real-space frame/null
        corrections and covariance orchestration still live in the research
        workflows. Use the original numerical signature; results and sampler
        pools are retained in ``last_run`` without changing resampling counts.
        """

        @wraps(target.calc_probs)
        def calc_probs(self, *args, output_path=None, seed=None, **kwargs):
            from .evidence import real, eclipses
            if output_path is not None:
                RunResult.check_destination(output_path)
            warnings.warn(
                "Compatibility engine: benchmark-specific frame/null corrections and "
                "FGP covariance orchestration are not automatically applied. "
                "See docs/architecture.md before interpreting FPP.",
                UserWarning, stacklevel=2,
            )
            counts = (real.POSTERIOR_NSAMPLES, eclipses.POSTERIOR_NSAMPLES)

            def calculate():
                try:
                    super(Target, self).calc_probs(*args, **kwargs)
                    table = self.probs.copy(deep=True)
                    table.attrs.update(FPP=float(self.FPP), NFPP=float(self.NFPP),
                                       engine="research_compatibility")
                    return table
                finally:
                    real.POSTERIOR_NSAMPLES, eclipses.POSTERIOR_NSAMPLES = counts

            self.last_run = run_evidence(
                calculate, seed=seed,
                metadata={"target_id": int(self.ID), "likelihood": "real_compatibility"},
            )
            # Named best-fit columns not present in the historical scenario table.
            self.last_run.metadata["best_fit_extras"] = {
                name: np.array(getattr(self, name), copy=True)
                for name in ("star_num", "u1", "u2", "fluxratio_EB", "fluxratio_comp")
            }
            if output_path is not None:
                self.last_run.save(output_path)
            return self.last_run

    Target.__module__ = __name__
    Target.__qualname__ = "Target"
    return Target


def __getattr__(name):
    if name == "Target":
        cls = _target_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)
