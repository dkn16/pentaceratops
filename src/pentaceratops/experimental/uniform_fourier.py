"""Shared-covariance uniform Fourier inference using the existing scenario priors."""
from dataclasses import replace
import importlib
from types import SimpleNamespace

import numpy as np

from .joint_fourier import JointFourierAdapter
from .v2_adapter import fatal_batch
from ..likelihoods import real as lk
from ..sampling.persistent import persistent_sampling


class UniformFourierAdapter(JointFourierAdapter):
    def __init__(self, metric, period, phase, *, nsamples=7, mission='TESS', filt=None,
                 timing_policy='observed', max_shift=None, backend='optimized'):
        super().__init__(metric, period, phase, nsamples=nsamples, mission=mission, filt=filt)
        if timing_policy not in ('observed', 'legacy') or backend not in ('optimized', 'scalar'):
            raise ValueError('Unknown timing policy or Fourier backend')
        if timing_policy == 'observed' and max_shift is not None:
            raise ValueError("An explicit timing cut requires timing_policy='legacy'")
        if max_shift is not None and (not np.isfinite(max_shift) or max_shift <= 0):
            raise ValueError('max_shift must be a positive finite duration')
        self.timing_policy, self.max_shift, self.backend = timing_policy, max_shift, backend
        self.window_half = float(np.max(abs(metric.blocks[0].time-phase)))
        self.parity_index = np.repeat(np.arange(len(metric.blocks)), metric.length)
        self.primary_time = (self.native_time+period/4) % period-period/4
        self.primary_half = self.primary_time < period/4
        # Use exactly the same phase grid in both parities, avoiding roundoff
        # from adding/subtracting a period during repeated-model compression.
        self.parity_time = np.tile(metric.blocks[0].time-phase, len(metric.blocks))

    def allowed(self, kind, parameters):
        period = np.asarray(parameters['P_orb'])
        if kind == 'planet' or self.timing_policy == 'observed':
            return np.ones(period.shape, dtype=bool)
        delta = (lk.mean_anomaly_difference(parameters['ecc'],
                    np.deg2rad(parameters['argp']))-.5)*period
        good = np.ones(period.shape, dtype=bool)
        if kind == 'x2p':
            good &= abs(delta)/2 <= self.window_half
        if self.max_shift is not None:
            good &= abs(delta) <= self.max_shift
        return good

    def native_model(self, kind, p, *, reverse=False):
        if self.timing_policy == 'observed':
            result = super().native_model(kind, p, reverse=reverse)
        else:
            self._check(kind, p)
            for name in ('_tm_cache', '_tm_sec_cache'):
                getattr(lk, name)['id_time'] = None
            if kind == 'planet':
                primary = lk.simulate_TP_transit(self.primary_time, **p)
                result = np.where(self.primary_half, primary, 1.)
            elif kind == 'binary':
                primary, secondary = lk.simulate_EB_transit_secondary(
                    self.primary_time, self.primary_time-self.period/2, **p)
                result = np.where(self.primary_half, primary, secondary)
            else:
                primary, secondary = lk.simulate_EB_transit_evenodd(
                    self.parity_time, self.parity_time, **p)
                result = np.where(self.parity_index == int(reverse), primary, secondary)
            result = 1+self.aperture_fraction*(result-1)
        if kind != 'x2p' and len(self.metric.blocks) == 2:
            result[self.metric.length:] = result[:self.metric.length]
        return result

    def _mask_projected_component(self, kind, values, secondary, reverse):
        if self.timing_policy == 'legacy':
            if kind == 'x2p':
                keep = self.parity_index == (int(reverse) if not secondary else 1-int(reverse))
            else:
                keep = ~self.primary_half if secondary else self.primary_half
            values[:, ~keep] = 0.
        return values

    def _project(self, kind, p, *, reverse=False):
        model = super()._project(kind, p, reverse=reverse)
        if kind != 'x2p' and len(self.metric.blocks) == 2:
            model[:, self.metric.length:] = model[:, :self.metric.length]
        return model

    def _scalar(self, kind, p):
        if not self.allowed(kind, p):
            return np.inf
        return super()._scalar(kind, p)

    def photometric_columns(self, kind, parameters):
        valid = self.allowed(kind, parameters)
        result = np.full(len(valid), -np.inf)
        if not valid.any():
            return result
        p = {key: value[valid] for key, value in parameters.items()}
        model = self._project(kind, p)
        if kind == 'x2p':
            first = self.metric.gains(model)
            second = self.metric.gains(self._project(kind, p, reverse=True))
            result[valid] = np.maximum(first, second)
        else:
            result[valid] = self.metric.common_gains(model[:, :self.metric.length])
        self.profile['model_batches'] += 1
        self.calls[kind] += int(valid.sum())
        return result

    def call(self, function, *args, **kwargs):
        """Replace only photometry, preserving recipe priors and output packing.

        Both scalar and optimized paths use the same common aperture-frame
        null normalization. Process isolation is required, as for other package
        adapters. Sampler hooks are restored even when a recipe fails.
        """
        module = importlib.import_module(function.__module__)
        original_sampler = module._run_persistent_evidence

        def sample(original, prior, ndim, n_active=100, target_ess=None, mcmc_steps=20):
            if original.domain != 'fourier':
                raise TypeError('Expected a Fourier scenario recipe')
            # physical_batch reuses the same physical model and prior weights;
            # its real-domain marker identifies the adapter callback contract.
            def cost(*unused, **parameters):
                return self._scalar(original.kind, parameters)
            observation = replace(original, domain='real',
                normalization=self.metric.null_loglike, residual_cost=cost, cost_options={})
            if self.backend == 'scalar':
                return original_sampler(observation, prior, ndim, n_active=n_active,
                                        target_ess=target_ess, mcmc_steps=mcmc_steps)
            z, weights, positions, values = persistent_sampling(
                log_likelihood_function=observation, prior_transform=prior, ndim=ndim,
                n_active=n_active, target_ess=2*n_active if target_ess is None else target_ess,
                mcmc_steps=mcmc_steps, prior_transform_batched=fatal_batch(prior.batch, 'prior'),
                log_likelihood_function_batched=fatal_batch(
                    lambda theta: self.likelihood_batch(observation, theta), 'likelihood'))
            return z, SimpleNamespace(samples=positions, weights=weights,
                                      logz=np.array([z]), log_likelihoods=values)
        module._run_persistent_evidence = sample
        try:
            return function(*args, **kwargs)
        finally:
            module._run_persistent_evidence = original_sampler
