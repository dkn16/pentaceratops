import numpy as np
import pytest
from pentaceratops.experimental.folded_joint import full_orbit_fold,FoldedJointFourierAdapter
from pentaceratops.experimental.joint_fourier import JointFourierAdapter
from pentaceratops.likelihoods.joint_fourier import GaussianBlock,observed_covariance


def example():
    n=360;grid=np.arange(n)*.025;mask=(np.arange(n)%9!=0)&~((grid>3.1)&(grid<3.5))
    ids=np.flatnonzero(mask);t=grid[ids]
    power=n*.002**2*(1+3/(1+np.arange(n//2+1)**2))
    cov=observed_covariance(power,n,ids);flux=1+.0003*np.sin(t)
    native=GaussianBlock.from_covariance(t,flux,cov,.025)
    a,ft,bins,weights=full_orbit_fold(t,2.,.13,.10)
    folded=GaussianBlock.from_covariance(ft,1+a@(flux-1),np.asarray(a@cov@a.T),.025)
    return native,folded,a,bins,weights


def test_covariance_after_full_orbit_fold_matches_simulated_draws():
    native,folded,a,bins,weights=example()
    rng=np.random.default_rng(701)
    c=np.linalg.inv(native.precision)
    draws=rng.multivariate_normal(np.zeros(len(c)),c,size=12000)
    sample=np.cov((a@draws.T),rowvar=True)
    actual=np.linalg.inv(folded.precision)
    np.testing.assert_allclose(sample,actual,rtol=.10,atol=.045*np.max(np.diag(actual)))
    np.testing.assert_allclose(np.asarray(a.sum(axis=1)).ravel(),1,atol=1e-14)
    assert a.nnz==len(native.time)


def test_gaps_and_parity_preserved_without_synthetic_bins():
    t=np.r_[np.arange(0,1,.05),np.arange(2,3,.05),np.arange(4,5,.05)]
    a,ft,bins,w=full_orbit_fold(t,2.,0.,.10)
    assert len(ft)<40
    f=1+np.where((t%4)<1,-.02,-.004)
    y=a@f
    np.testing.assert_allclose(y[ft<1],.98)
    np.testing.assert_allclose(y[(ft>=2)&(ft<3)],.996)
    assert not np.any((ft>1.1)&(ft<1.9))


@pytest.mark.parametrize('kind',['planet','binary','x2p'])
@pytest.mark.parametrize('parity',['profile','marginalize'])
def test_exact_model_projection_and_scalar_batch_scores(kind,parity):
    native,folded,a,bins,weights=example()
    original=JointFourierAdapter([native],2.,.13,parity=parity,nsamples=7)
    adapter=FoldedJointFourierAdapter(folded,native.time,bins,weights,2.,.13,parity=parity,nsamples=7)
    original.aperture_fraction=adapter.aperture_fraction=.07
    count=3
    p=dict(P_orb=np.full(count,4. if kind=='x2p' else 2.),inc=np.full(count,89.8),
        ecc=np.array([0.,.1,.2]),argp=np.array([90.,110.,270.]),a=np.full(count,5e11),
        R_s=np.ones(count),u1=np.full(count,.3),u2=np.full(count,.2),R_p=np.array([2.,5.,10.]),
        R_EB=np.array([.2,.5,1.2]),EB_fluxratio=np.full(count,.2),companion_fluxratio=np.full(count,.15),
        companion_is_host=np.array([False,True,False]))
    exact=[];scores=[]
    for i in range(count):
        q={k:v[i] for k,v in p.items()}
        for reverse in (False,True) if kind=='x2p' else (False,):
            expected=1+a@(original.native_model(kind,q,reverse=reverse)-1)
            actual=adapter.native_model(kind,q,reverse=reverse)
            np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=0)
        exact.append(adapter.native_model(kind,q))
        q.pop('R_p' if kind!='planet' else 'R_EB')
        if kind=='planet':q.pop('EB_fluxratio')
        q.update(exptime=.025,nsamples=7)
        scores.append(-adapter._scalar(kind,q))
    np.testing.assert_allclose(adapter._project(kind,p),exact,atol=3e-9,rtol=1e-12)
    np.testing.assert_allclose(adapter.photometric_columns(kind,p),scores,atol=2e-5,rtol=2e-12)


def test_no_compression_preserves_likelihood():
    native,folded,a,bins,weights=example()
    t=native.time[:40];c=np.linalg.inv(native.precision)[:40,:40];f=native.flux[:40]
    a,ft,bins,w=full_orbit_fold(t,20.,0.,.001)
    assert a.shape[0]==len(t)
    b=GaussianBlock.from_covariance(ft,a@f,np.asarray(a@c@a.T),.025)
    n=GaussianBlock.from_covariance(t,f,c,.025)
    signal=np.sin(t)*.001
    g=lambda block,s: block.projected_residual@s-.5*s@block.precision@s
    np.testing.assert_allclose(g(b,a@signal),g(n,signal),atol=1e-10)
