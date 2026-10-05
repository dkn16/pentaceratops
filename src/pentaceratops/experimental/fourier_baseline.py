"""Change only observations/covariance for frozen Fourier scenario callbacks.

Original prior transforms, geometry, physical renderers, timing selections,
posterior packaging and scalar sampler are reused without modification.
The original full-period primary/secondary and even/odd conventions describe
model coordinates only: no observed cadence is discarded and there is one
joint covariance across every retained orbital phase and epoch.
"""
from dataclasses import replace
import numpy as np


class FourierBaselineObservation:
    def __init__(self, original, adapter):
        from ..evidence.scenario import ScenarioLikelihood
        if not isinstance(original, ScenarioLikelihood) or original.domain != 'fourier':
            raise TypeError('Require the original Fourier ScenarioLikelihood')
        self.original, self.adapter = original, adapter
        self.kind = original.kind
        self.max_shift = original.cost_options.get('max_shift')
        expected = {'planet':'lnL_TP_fourier', 'binary':'lnL_EB_second_fourier',
                    'x2p':'lnL_EB_evenodd_fourier'}[self.kind]
        if original.residual_cost.__name__ != expected:
            raise ValueError('Unrecognized baseline residual function')
        if original.exptime != adapter.exptime or original.nsamples != adapter.nsamples:
            raise ValueError('Exposure integration differs from baseline')
        self.window_half = max(float(np.max(abs(original.time))),
            float(np.max(abs(original.secondary[0])))) if self.kind=='x2p' else None
        self.observation = replace(original, normalization=0., residual_cost=self.cost)
        self.calls = 0
        p = adapter.period
        relative = adapter.metric.time-adapter.epoch
        self.primary_time = (relative+p/4)%p-p/4
        self.secondary_time = self.primary_time-p/2
        self.primary_half = self.primary_time < p/4
        # Match the original full-P even/odd arrays, starting at archive time 0.
        # Small grid rounding differences are data coordinates, not new physics.
        self.parity_time = adapter.metric.time % p-adapter.epoch
        self.parity = np.floor(adapter.metric.time/p).astype(np.int64)%2
        self.reference = False

    def timing_allowed(self, parameters):
        if self.kind == 'planet':return True
        delta = (self.adapter.lk.mean_anomaly_difference(parameters['ecc'],
            np.deg2rad(parameters['argp']))-.5)*parameters['P_orb']
        if self.kind=='x2p' and abs(delta)/2 > self.window_half:return False
        if self.max_shift is not None:
            if not np.isfinite(self.max_shift) or self.max_shift<=0 or abs(delta)>self.max_shift:
                return False
        return True

    def render(self, parameters):
        lk = self.adapter.lk
        # Reset the original identity-only data cache. It must not confuse
        # arrays recycled by independent model checks with the fixed fit grid.
        for name in ('_tm_cache','_tm_sec_cache'):
            getattr(lk,name)['id_time']=None
        if self.kind=='planet':
            p=lk.simulate_TP_transit(self.primary_time, **parameters)
            model=np.where(self.primary_half,p,1.)
            models=[model]
        elif self.kind=='binary':
            p,s=lk.simulate_EB_transit_secondary(self.primary_time,self.secondary_time,**parameters)
            models=[np.where(self.primary_half,p,s)]
        else:
            p,s=lk.simulate_EB_transit_evenodd(self.parity_time,self.parity_time,**parameters)
            models=[np.where(self.parity==0,p,s),np.where(self.parity==0,s,p)]
        return [1+self.adapter.aperture_fraction*(model-1) for model in models]

    def cost(self, *unused_arrays, **parameters):
        self.calls += 1
        supplied_shift = parameters.pop('max_shift', None)
        if supplied_shift != self.max_shift:raise ValueError('Baseline timing constraint changed')
        if not self.timing_allowed(parameters):return np.inf
        models=self.render(parameters)
        gains=[self.adapter.metric.gain(model) for model in models]
        chosen=int(np.argmax(gains));gain=gains[chosen]
        if self.reference:
            # Direct full residual expression independently checks sparse gain.
            direct=0.
            for block,lo,hi in zip(self.adapter.metric.blocks,self.adapter.metric.offsets[:-1],self.adapter.metric.offsets[1:]):
                residual=block.flux-models[chosen][lo:hi]
                nullres=block.flux-1
                direct+=.5*(nullres@block.projected_residual-residual@block.precision@residual)
            np.testing.assert_allclose(gain,direct,atol=2e-5,rtol=2e-12)
        if self.adapter.record:
            self.adapter.snapshot=dict(kind=self.kind,parameters={k:np.asarray(v).item() for k,v in parameters.items()},
                model=models[chosen].copy(),photometric_loglike_ratio=float(gain),
                aperture_fraction=float(self.adapter.aperture_fraction),
                original_max_shift=self.max_shift,reversed_parity=bool(chosen),
                parity_loglike_ratios=[float(g) for g in gains])
        return -float(gain)

    def __call__(self, theta):return self.observation(theta)
