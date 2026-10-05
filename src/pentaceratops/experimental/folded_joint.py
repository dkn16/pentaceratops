"""Full-orbit phase folding of a PSD Gaussian, with exact folded model means.

This is ordinary equal-weight compression, not a claim of sufficient statistics.
All native observed samples enter once; empty phase bins are absent.
"""
import time
import numpy as np
from scipy.sparse import csr_matrix
from pytransit import QuadraticModel
from .joint_fourier import JointFourierAdapter
from .sparse_kernels import orbit_setup, project_strict
from ..likelihoods import real as lk


def full_orbit_fold(time, period, epoch, bin_days):
    """Return A, phase centers, native-to-retained-bin ids and equal weights.

    Use 2P with an even number of bins, preserving the two eclipse parities.
    The bin width is at most bin_days. Each observed native sample enters once.
    """
    time=np.asarray(time,float)
    if (time.ndim!=1 or not len(time) or not np.isfinite(time).all()
            or not np.isfinite([period,epoch,bin_days]).all()
            or period<=0 or bin_days<=0 or bin_days>period/2):
        raise ValueError('Invalid full-orbit folding inputs')
    n=2*int(np.ceil(period/bin_days));width=2*period/n
    phase=np.remainder(time-epoch,2*period)
    slot=np.floor(phase/width+.5).astype(np.int64)%n
    retained,inverse,counts=np.unique(slot,return_inverse=True,return_counts=True)
    weights=1./counts[inverse]
    operator=csr_matrix((weights,(inverse,np.arange(len(time)))),shape=(len(retained),len(time)))
    return operator,epoch+retained*width,np.ascontiguousarray(inverse),np.ascontiguousarray(weights)


class FoldedJointFourierAdapter(JointFourierAdapter):
    """Score A m(theta) against A y with A C A.T, throughout the complete 2P orbit.

    The folded Gaussian block and folding map must be built from the same native
    observation ordering. The scalar reference applies A to dense PyTransit
    predictions; the optimized path accumulates native exposures into A's bins.
    """
    def __init__(self, block, native_time, bin_index, weights, period, epoch, *,
                 parity='profile', nsamples=7, chunk_size=64, mission='TESS', filt=None):
        super().__init__([block],period,epoch,parity=parity,nsamples=nsamples,
                         chunk_size=chunk_size,mission=mission,filt=filt)
        time=np.asarray(native_time,float);ids=np.asarray(bin_index);weights=np.asarray(weights,float)
        m=len(block.time)
        if (time.ndim!=1 or not len(time) or not np.isfinite(time).all()
                or ids.shape!=time.shape or ids.dtype.kind not in 'iu'
                or weights.shape!=time.shape or not np.isfinite(weights).all()
                or np.any(weights<=0) or np.any(ids<0) or np.any(ids>=m)):
            raise ValueError('Invalid native folding map')
        if not np.allclose(np.bincount(ids,weights=weights,minlength=m),1.,atol=1e-12,rtol=0):
            raise ValueError('Fold weights must sum to one in every retained bin')
        self.fold_operator=csr_matrix((weights,(ids,np.arange(len(time)))),shape=(m,len(time)))
        self.native_time=np.ascontiguousarray(time-epoch)
        self.identity=np.ascontiguousarray(ids,dtype=np.int64)
        self.weights=np.ascontiguousarray(weights)
        self.fold_size=m
        self.primary_model=QuadraticModel(interpolate=False)
        self.secondary_model=QuadraticModel(interpolate=False)
        for model in (self.primary_model,self.secondary_model):
            model.set_data(self.native_time,exptimes=self.exptime,nsamples=self.nsamples)
        for mult in (1,2):
            phase=self.native_time%(mult*self.period);order=np.argsort(phase,kind='stable')
            self.grids[mult]=(np.ascontiguousarray(phase[order]),np.ascontiguousarray(order))

    def native_model(self,kind,p,*,reverse=False):
        # Fold the signal, preserving exactly unit baseline despite roundoff.
        native=super().native_model(kind,p,reverse=reverse)
        return 1+np.asarray(self.fold_operator@(native-1)).ravel()

    def _project(self,kind,p,*,reverse=False):
        self._check(kind,p);start=time.perf_counter();n=len(p['P_orb'])
        k,tpri,tsec,apri,asec=self._components(kind,p,reverse)
        signal=np.zeros((n,self.fold_size))
        phase,order=self.grids[2 if kind=='x2p' else 1]
        for secondary in (False,True) if kind!='planet' else (False,):
            pars=np.empty((n,10))
            pars[:,0]=1/k if secondary else k
            pars[:,1]=tsec if secondary else tpri
            pars[:,2]=p['P_orb']
            pars[:,3]=p['a']/((k if secondary else 1)*p['R_s']*lk.Rsun)
            pars[:,4]=np.deg2rad(p['inc']);pars[:,5]=p['ecc']
            pars[:,6]=np.deg2rad((270 if secondary else 90)-p['argp'])
            pars[:,7],pars[:,8]=p['u1'],p['u2']
            pars[:,9]=self.aperture_fraction*(asec if secondary else apri)
            coeff,windows=orbit_setup(pars)
            value,counts=project_strict(pars,coeff,windows,self.native_time,phase,order,
                self.identity,self.weights,self.fold_size,self.exptime,self.nsamples)
            signal+=value;self.profile['native_evaluations']+=int(counts.sum())
        self.profile['exposure_seconds']+=time.perf_counter()-start
        return 1+signal

    def scenario_kwargs(self,function,star,*,trilegal,molusc=None,N=500,steps=50):
        values=super().scenario_kwargs(function,star,trilegal=trilegal,molusc=molusc,N=N,steps=steps)
        for key in ('time','time_even'):
            if key in values:values[key]=self.metric.time-self.epoch
        return values
