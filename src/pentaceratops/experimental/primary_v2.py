"""Opt-in optimized v2 likelihood for observed even/odd primary windows only.

The conditional FGP covariance is retained in full. Missing secondary data
contribute no likelihood term or synthetic non-detection constraint. The
inherited 2P recipe still fits alternating eclipse depths in even/odd windows.
"""
from pathlib import Path
import numpy as np
from .v2_adapter import OptimizedAdapter


def load_primary_inputs(row, folded_root=''):
    """Load primary-only products with the established per-panel noise model."""
    path = Path(row['prepared_path'])
    folded_path = Path(row.get('folded_path') or Path(folded_root)/row['group']/row['tag']/'folded_posterior.npz')
    result = {}
    with np.load(path, allow_pickle=False) as prepared, np.load(folded_path, allow_pickle=False) as folded:
        if str(prepared['v2_likelihood_mode']) != 'primary_only':
            raise ValueError('Require explicitly prepared primary-only v2 inputs')
        if len(prepared['time_secondary']):
            raise ValueError('Primary-only loader refuses to discard observed secondary data')
        factors = []
        for panel in ('even','odd','secondary'):
            for key, source in (('time','time'),('map_flux','map_flux'),('mean_flux','conditional_mean'),('input_error','input_error')):
                result[f'{panel}_{key}'] = np.asarray(folded[f'{panel}_{source}'],float)
            for key, source in (('time','time'),('map_flux','flux'),('input_error','err')):
                np.testing.assert_allclose(result[f'{panel}_{key}'],prepared[f'{source}_{panel}'],atol=1e-10,rtol=0)
            factor = np.asarray(folded[f'{panel}_covariance_factor'],float)
            factors.append(factor)
            result[f'{panel}_sigma'] = (np.full(len(factor),np.median(result[f'{panel}_input_error']))
                                       if len(factor) else np.empty(0))
        if len({a.shape[1] for a in factors}) != 1:
            raise ValueError('Panel coefficient columns must have identical ordering')
        result['factor'] = np.vstack(factors)
        result.update(exptime=float(prepared['exptime_days']),period=float(prepared['period']),
                      primary_time=prepared['time'].copy(),primary_flux=folded['primary_conditional_mean'].copy())
    for field, suffix in (('flux','mean_flux'),('map_flux','map_flux'),('sigma','sigma')):
        result[field] = np.concatenate([result[p+'_'+suffix] for p in ('even','odd','secondary')])
    return result


class PrimaryOnlyAdapter(OptimizedAdapter):
    """Same optimized v2 physical recipes, scored on even/odd observations.

    Empty secondary arrays are structural placeholders, never observations.
    This explicit adapter does not change the default three-window adapter.
    """
    def __init__(self, data, include_gp=True, parity='profile', chunk_size=64, *,
                 nsamples=20, timing_policy='legacy'):
        if len(data['secondary_time']) or len(data['secondary_mean_flux']) or len(data['secondary_sigma']):
            raise ValueError('Primary-only mode requires no secondary observations')
        if not len(data['even_time']) or not len(data['odd_time']):
            raise ValueError('Primary-only v2 requires observed even and odd primary windows')
        if len(data['flux']) != len(data['even_time']) + len(data['odd_time']):
            raise ValueError('Primary-only data vector must contain only even and odd panels')
        super().__init__(data, include_gp=include_gp, parity=parity, chunk_size=chunk_size,
                         nsamples=nsamples, timing_policy=timing_policy)

    def binary(self, time, flux, sigma, time_secondary, flux_secondary, sigma_secondary,
               R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp,
               companion_fluxratio=0., companion_is_host=False, exptime=.00139, nsamples=None):
        nsamples = self.integration_samples(nsamples)
        self.calls['binary'] += 1
        parameters = dict(R_EB=R_EB,EB_fluxratio=EB_fluxratio,P_orb=P_orb,inc=inc,a=a,R_s=R_s,
                          u1=u1,u2=u2,ecc=ecc,argp=argp,companion_fluxratio=companion_fluxratio,
                          companion_is_host=companion_is_host,exptime=exptime,nsamples=nsamples)
        primary, _ = self.lk.simulate_EB_transit(self.sorted_eo_time, **parameters)
        # simulate_EB_transit also returns a depth probe. It is deliberately
        # unused: no observed secondary means no constraint on that depth.
        model = self.aperture(primary)[self.inverse_order]
        gain = self.metric.gain(model)
        self.save_snapshot('binary',parameters,model,gain,secondary_omitted=True,phantom_loglike_ratio=0.)
        return -gain

    def photometric_columns(self, kind, p, *, nsamples=None):
        if kind != 'binary':
            return super().photometric_columns(kind,p,nsamples=nsamples)
        nsamples = self.integration_samples(nsamples)
        n = len(p['P_orb'])
        primary = self.dilute(self.undiluted(p,'primary',np.zeros(n),nsamples=nsamples),p)
        value = self.score(primary[:,self.inverse_order])
        self.calls['binary'] += n
        self.profile['model_batches'] += 1
        return value
