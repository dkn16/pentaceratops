"""Opt-in strict CPU batching of the ORIGINAL matched-window v2 likelihood.

No raw-flux/full-phase v3 model is substituted. The inherited scalar callbacks
remain the reference and best-fit replay path. Only batched evaluations use
the vector physics and exposure kernels copied from the validated v3 test.
"""
import time
import numpy as np
from .covariance import ModelAdapter
from .physical_batch import PhysicalBatch
from .sparse_kernels import orbit_setup, project_strict, quadratic_strict


class BatchFailure(BaseException):
    """Cannot be swallowed by a sampler's Exception-based fallback."""


def fatal_batch(fn, label):
    def wrapped(*args):
        try: return fn(*args)
        except Exception as exc: raise BatchFailure(f'{label}: {exc}') from exc
    return wrapped


class OptimizedAdapter(ModelAdapter):
    """Fixed-P/2P batched models with configurable within-exposure integration.

    ``nsamples`` defaults to the historical 20. An evidence function's explicit
    integration setting takes precedence in ``likelihood_batch``; direct model
    calls can supply a keyword override. Exposure duration remains data['exptime'].
    """

    def __init__(self, data, include_gp=True, parity='profile', chunk_size=64, *,
                 nsamples=20, timing_policy='legacy'):
        super().__init__(data, include_gp=include_gp, parity=parity, nsamples=nsamples,
                         timing_policy=timing_policy)
        self.chunk_size = chunk_size
        self.physical = {}
        self.grids = {}
        self.metric_arrays = {}
        for name, grid in dict(primary=self.sorted_eo_time, **self.times).items():
            times = np.ascontiguousarray(grid)
            for mult in (1, 2):
                phase = times % (mult*data['period'])
                order = np.argsort(phase, kind='stable')
                self.grids[name,mult] = (times,np.ascontiguousarray(phase[order]),
                    np.ascontiguousarray(order),np.arange(len(times),dtype=np.int64),np.ones(len(times)))
        for name,metric in [('joint',self.metric),('secondary',self.secondary_metric)]:
            n = len(metric.flux)
            self.metric_arrays[name] = (np.ascontiguousarray(metric.precision.ravel()),
                np.ascontiguousarray(metric.projected_residual),np.array([0,n]),np.array([0,n*n]))
        self.reset_profile()

    def reset_profile(self):
        self.profile = dict(physical_setup_seconds=0.,physics_seconds=0.,exposure_seconds=0.,
                            covariance_seconds=0.,native_evaluations=0,model_batches=0)

    def make_physics(self, scalar):
        if scalar not in self.physical:
            start=time.perf_counter()
            self.physical[scalar]=PhysicalBatch(scalar)
            self.profile['physical_setup_seconds']+=time.perf_counter()-start
        return self.physical[scalar]

    def undiluted(self, p, grid, t0, secondary=False, planet=False, x2p=False, *, nsamples=None):
        nsamples = self.integration_samples(nsamples)
        start=time.perf_counter()
        n=len(p['P_orb']);pars=np.zeros((n,10))
        k=p['R_p']*self.lk.Rearth/(p['R_s']*self.lk.Rsun) if planet else p['R_EB']/p['R_s']
        if not planet:k=np.where(abs(k-1)<1e-6,k*.999,k)
        pars[:,0]=1/k if secondary else k
        pars[:,1]=t0;pars[:,2]=p['P_orb']
        pars[:,3]=p['a']/((k*p['R_s'] if secondary else p['R_s'])*self.lk.Rsun)
        pars[:,4]=p['inc']*(np.pi/180.)
        pars[:,5]=p['ecc']
        pars[:,6]=(90-p['argp']+(180 if secondary else 0))*(np.pi/180.)
        pars[:,7]=p['u1'];pars[:,8]=p['u2'];pars[:,9]=1.
        if not np.isfinite(pars).all():raise ValueError('Nonfinite model parameters')
        expected=self.data['period']*(2 if x2p else 1)
        if not np.allclose(p['P_orb'],expected,atol=0,rtol=1e-12):raise ValueError('Expected fixed P or 2P')
        coefficients,windows=orbit_setup(pars)
        times,phase,order,index,weight=self.grids[grid,2 if x2p else 1]
        signal,count=project_strict(pars,coefficients,windows,times,phase,order,index,weight,len(times),self.data['exptime'],nsamples)
        self.profile['exposure_seconds']+=time.perf_counter()-start
        self.profile['native_evaluations']+=int(count.sum())
        return 1+signal

    def dilute(self, model, p, planet=False, secondary=False):
        # Preserve the original sequential dilution arithmetic and aperture frame.
        comp=p['companion_fluxratio']/(1-p['companion_fluxratio'])
        host=np.where(p['companion_is_host'],comp,1.)
        if planet:
            dilution=np.divide(1.,comp,out=np.full(len(comp),np.inf),where=comp!=0)
            dilution=np.where(p['companion_is_host'],dilution,comp)
            result=(model+dilution[:,None])/(1+dilution[:,None])
        else:
            eb=p['EB_fluxratio']/(1-p['EB_fluxratio'])
            ratio=host/eb if secondary else eb/host
            result=(model+ratio[:,None])/(1+ratio[:,None])
            dilution=np.where(p['companion_is_host'],1/(comp+eb),comp/(1+eb))
            result=(result+dilution[:,None])/(1+dilution[:,None])
        return 1+self.aperture_fraction*(result-1)

    def score(self, model, metric='joint'):
        start=time.perf_counter()
        signal=np.ascontiguousarray(model-1)
        finite=np.isfinite(signal).all(axis=1)
        result=np.full(len(model),-np.inf)
        result[finite]=quadratic_strict(signal[finite],*self.metric_arrays[metric])
        result[~np.isfinite(result)]=-np.inf
        self.profile['covariance_seconds']+=time.perf_counter()-start
        return result

    def photometric_columns(self, kind, p, *, nsamples=None):
        nsamples = self.integration_samples(nsamples)
        n=len(p['P_orb']);out=np.full(n,-np.inf)
        offset=(self.lk.mean_anomaly_difference(p['ecc'],p['argp']*(np.pi/180.))-.5)*p['P_orb']
        if kind=='x2p':
            if self.timing_policy == 'legacy':
                half=max(np.max(abs(self.times['even'])),np.max(abs(self.times['odd'])))
                allowed=abs(offset)/2<=half
            else:
                from .window_support import alternating_overlap
                allowed=alternating_overlap(self.times['even'],self.times['odd'],p,self.data['exptime'])
            ids=np.flatnonzero(allowed)
            if not len(ids):return out
            p={k:v[ids] for k,v in p.items()};offset=offset[ids]
            primary={};secondary={}
            for panel in ('even','odd'):
                primary[panel]=self.dilute(self.undiluted(p,panel,-.5*offset,x2p=True,nsamples=nsamples),p)
                secondary[panel]=self.dilute(self.undiluted(p,panel,.5*offset,secondary=True,x2p=True,nsamples=nsamples),p,secondary=True)
            flat=np.ones((len(ids),len(self.times['secondary'])))
            forward=np.concatenate([primary['even'],secondary['odd'],flat],axis=1)
            reverse=np.concatenate([secondary['even'],primary['odd'],flat],axis=1)
            a,b=self.score(forward),self.score(reverse)
            out[ids]=np.maximum(a,b) if self.parity=='profile' else np.logaddexp(a,b)-np.log(2)
        else:
            primary=self.dilute(self.undiluted(p,'primary',np.zeros(n),planet=kind=='planet',nsamples=nsamples),p,planet=kind=='planet')
            sec=np.ones((n,len(self.times['secondary'])))
            phantom=np.zeros(n)
            if kind=='binary':
                ts=self.times['secondary'];outside=(offset<ts.min())|(offset>ts.max())
                # Preserve the existing centered-secondary non-detection
                # constraint under both timing policies.
                proposal=self.dilute(self.undiluted(p,'secondary',np.where(outside,0.,offset),secondary=True,nsamples=nsamples),p,secondary=True)
                sec[~outside]=proposal[~outside]
                phantom[outside]=self.score(proposal[outside],metric='secondary')
            model=np.concatenate([primary[:,self.inverse_order],sec],axis=1)
            out=self.score(model)+phantom
        self.calls[kind]+=n
        self.profile['model_batches']+=1
        return out

    def likelihood_batch(self, scalar, theta):
        if self.record:raise RuntimeError('Best-fit replay must use the original scalar callback')
        physics=self.make_physics(scalar)
        # Use the SAME integration count as the scalar evidence callback, even
        # when the caller overrides the adapter's default via scenario kwargs.
        nsamples = self.integration_samples(physics.nsamples)
        start=time.perf_counter();out,ids,columns=physics.evaluate(theta)
        self.profile['physics_seconds']+=time.perf_counter()-start
        for lo in range(0,len(ids),self.chunk_size):
            part=slice(lo,lo+self.chunk_size)
            out[ids[part]]+=self.photometric_columns(physics.kind,{k:v[part] for k,v in columns.items()},nsamples=nsamples)
        out[~np.isfinite(out)]=-np.inf
        return out
