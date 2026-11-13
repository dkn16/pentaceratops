"""Opt-in process-local sampler hooks, restored on context exit."""
from contextlib import contextmanager
from types import SimpleNamespace
import time
from .batched_priors import make_prior
from .v2_adapter import fatal_batch


@contextmanager
def mode_context(module, adapter, mode, validate=None):
    if mode not in ('scalar', 'prior_only', 'batch_prior'): raise ValueError(mode)
    original = module._run_persistent_evidence
    stats = dict(sampler_seconds=0., prior_seconds=0., likelihood_seconds=0.,
                 prior_batches=0, likelihood_batches=0, sampler_invocations=0)
    def run(loglike, prior_transform, ndim, n_active=100, target_ess=None, mcmc_steps=20, **unused):
        if unused: raise ValueError(f'Unexpected sampler options: {unused}')
        prior_batch = make_prior(prior_transform)
        loglike_batch = lambda theta: adapter.likelihood_batch(loglike, theta)
        sigma = getattr(adapter, 'epoch_sigma', None)
        if sigma is not None:
            # Free epoch: one trailing coordinate, offset ~ N(0, sigma) in days. The scenario's own
            # prior and likelihood see the original coordinates only.
            import numpy as np
            from scipy.special import ndtri
            if mode != 'batch_prior' or validate is not None or not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('Free epoch needs batch_prior mode and a positive finite width')
            base_prior, base_batch, base_like = prior_transform, prior_batch, loglike
            draw = lambda u: sigma*ndtri(np.clip(u, 1e-9, 1-1e-9))
            def prior_transform(u):
                u = np.asarray(u, float)
                return np.append(base_prior(u[:-1].copy()), draw(u[-1]))
            def prior_batch(u):
                u = np.asarray(u, float)
                return np.column_stack([base_batch(np.ascontiguousarray(u[:, :-1])), draw(u[:, -1])])
            def loglike(theta):
                adapter.scalar_epoch_offset = float(theta[-1])
                try: return base_like(theta[:-1])
                finally: adapter.scalar_epoch_offset = 0.
            loglike_batch = lambda theta: adapter.likelihood_batch(base_like, theta)
            ndim = ndim+1
        if validate is not None:
            validate(loglike, prior_transform, prior_batch, loglike_batch, ndim)
        stats['sampler_invocations'] += 1
        def timed(fn, label):
            def call(value):
                start = time.perf_counter()
                try: return fn(value)
                finally: stats[label+'_seconds'] += time.perf_counter()-start
            return call
        def batched(fn, label):
            measured = timed(fn, label)
            def call(value):
                stats[label+'_batches'] += 1
                return measured(value)
            return fatal_batch(call, label)
        start = time.perf_counter()
        try:
            z, weights, positions, values = module.persistent_sampling(
                log_likelihood_function=timed(loglike, 'likelihood'),
                prior_transform=timed(prior_transform, 'prior'), n_active=n_active, ndim=ndim,
                target_ess=2*n_active if target_ess is None else target_ess, mcmc_steps=mcmc_steps,
                prior_transform_batched=None if mode == 'scalar' else batched(prior_batch, 'prior'),
                log_likelihood_function_batched=batched(loglike_batch, 'likelihood') if mode == 'batch_prior' else None)
        finally:
            stats['sampler_seconds'] += time.perf_counter()-start
        import numpy as np
        if sigma is not None:
            positions = np.asarray(positions)
            adapter.epoch_samples = positions[:, -1].copy()
            positions = positions[:, :-1]
        return z, SimpleNamespace(samples=positions, weights=weights, logz=np.array([z]), log_likelihoods=values)
    module._run_persistent_evidence = run
    try: yield stats
    finally: module._run_persistent_evidence = original
