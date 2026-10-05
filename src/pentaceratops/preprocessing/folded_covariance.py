"""Retain full-orbit phase folding and propagate its actual sampling covariance."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.fft import rfft, irfft
from ..likelihoods.joint_fourier import GaussianBlock, observed_covariance


def folding_operator(length, period_bins, events, observed, *, alternating=False):
    """Original rounded-cycle folding, with contributions from selected events.

    P folds use round(k*P/dt); 2P folds use round(k*2P/dt), as in the archived
    exporter. Equal-weight observed contributors are averaged in each phase
    bin. Empty bins are absent observations, never synthetic filled data.
    """
    observed=np.asarray(observed)
    if observed.dtype!=bool or observed.shape!=(length,) or not np.isfinite(period_bins) or period_bins<2:
        raise ValueError('Invalid native-grid mask or period')
    events=np.asarray(events)
    if events.ndim!=1 or events.dtype.kind not in 'iu' or len(np.unique(events))!=len(events) or np.any(events<1):
        raise ValueError('Require unique positive event numbers')
    n=int(period_bins);size=2*n if alternating else n
    rows=[];columns=[]
    for event in events:
        cycle=int(event)-1
        start=int(.5+(cycle-cycle%2)*period_bins)+cycle%2*n if alternating else int(.5+cycle*period_bins)
        bins=np.arange(n)+(cycle%2*n if alternating else 0)
        ids=start+np.arange(n);valid=(ids>=0)&(ids<length)
        bins,ids=bins[valid],ids[valid]
        keep=observed[ids]
        rows.extend(bins[keep]);columns.extend(ids[keep])
    rows=np.asarray(rows,int);columns=np.asarray(columns,int)
    counts=np.bincount(rows,minlength=size)
    retained=np.flatnonzero(counts)
    if not len(retained):raise ValueError('No observed folded bins')
    operator=coo_matrix((1./counts[rows],(rows,columns)),shape=(size,length)).tocsr()[retained]
    return operator,retained,counts


def propagate_fourier_covariance(power, length, operator, *, chunk_columns=128, workers=1):
    """Compute A C A.T by FFT covariance products, without making native C."""
    observed_covariance(power,length,np.array([0],dtype=int))
    if operator.shape[1]!=length or not operator.shape[0] or chunk_columns<1:
        raise ValueError('Invalid folding operator')
    m=operator.shape[0];covariance=np.empty((m,m))
    for lo in range(0,m,chunk_columns):
        hi=min(m,lo+chunk_columns)
        rhs=operator[lo:hi].T.toarray()
        modes=rfft(rhs,axis=0,workers=workers)
        modes *= (np.asarray(power)/length)[:,None]
        product=irfft(modes,n=length,axis=0,workers=workers)
        covariance[:,lo:hi]=operator@product
    return .5*(covariance+covariance.T)


def dropped_fourier_modes(length, *, roll=0):
    """Original per-half DC and (when real) Nyquist nuisance modes.

    These reproduce the original full-P half split or the two full-P parity
    folds; they impose no local eclipse window and discard no observations.
    """
    if length%2:raise ValueError('Require the original equal-length fold halves')
    half=length//2;order=np.roll(np.arange(length),roll);columns=[]
    for ids in (order[:half],order[half:]):
        mode=np.zeros(length);mode[ids]=1.;columns.append(mode)
        if half%2==0:
            mode=np.zeros(length);mode[ids]=(-1.)**np.arange(half);columns.append(mode)
    return np.column_stack(columns)


def marginalized_mode_block(time, flux, covariance, exptime, modes):
    """Normalized Gaussian on the subspace retaining the original FFT modes."""
    block=GaussianBlock.from_covariance(time,flux,covariance,exptime)
    modes=np.asarray(modes,float)
    if modes.ndim!=2 or modes.shape[0]!=len(flux) or not np.isfinite(modes).all():
        raise ValueError('Invalid ignored-mode basis')
    u,s,_=np.linalg.svd(modes,full_matrices=False)
    rank=int(np.sum(s>max(modes.shape)*np.finfo(float).eps*s[0])) if len(s) else 0
    if rank>=len(flux):raise ValueError('No retained likelihood modes')
    u=u[:,:rank]
    if rank:
        projected=block.precision@u
        gram=u.T@projected
        sign,logdet_gram=np.linalg.slogdet(gram)
        if sign!=1:raise ValueError('Invalid nuisance-mode metric')
        precision=block.precision-projected@np.linalg.solve(gram,projected.T)
        block.precision=np.ascontiguousarray(.5*(precision+precision.T))
        block.projected_residual=block.precision@(block.flux-1)
        block.logdet+=float(logdet_gram)
        r=block.flux-1
        block.null_loglike=float(-.5*(r@block.projected_residual+block.logdet+(len(r)-rank)*np.log(2*np.pi)))
    else:block.precision=np.ascontiguousarray(block.precision)
    return block,rank
