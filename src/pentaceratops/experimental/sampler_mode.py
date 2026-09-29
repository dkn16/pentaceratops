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
        return z, SimpleNamespace(samples=positions, weights=weights, logz=np.array([z]), log_likelihoods=values)
    module._run_persistent_evidence = run
    try: yield stats
    finally: module._run_persistent_evidence = original
