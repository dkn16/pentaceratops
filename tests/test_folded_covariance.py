import numpy as np
import pytest
from scipy.linalg import null_space
from scipy.stats import multivariate_normal
from pentaceratops.preprocessing.folded_covariance import (
    folding_operator,propagate_fourier_covariance,dropped_fourier_modes,marginalized_mode_block)
from pentaceratops.likelihoods.joint_fourier import observed_covariance,JointFourierMetric


@pytest.mark.parametrize('alternating',[False,True])
def test_fold_and_covariance_use_same_gapped_selected_contributors(alternating):
    n=91;period=12.3;events=np.array([1,3,4,6]);mask=np.ones(n,bool)
    mask[3:7]=False;mask[40:44]=False
    a,ids,counts=folding_operator(n,period,events,mask,alternating=alternating)
    flux=np.sin(np.arange(n));flux[~mask]=12345.
    values=a@flux;manual=[]
    for row in ids:
        contributors=[]
        for event in events:
            k=event-1
            if alternating and row//int(period)!=k%2:continue
            start=int(.5+(k-k%2)*period)+k%2*int(period) if alternating else int(.5+k*period)
            index=start+row%int(period)
            if index<n and mask[index]:contributors.append(flux[index])
        manual.append(np.mean(contributors))
        assert counts[row]==len(contributors)
    np.testing.assert_allclose(values,manual,atol=1e-15)
    power=n*(.1+1/(1+np.arange(n//2+1))**2)
    dense=observed_covariance(power,n,np.arange(n))
    expected=a@dense@a.T
    actual=propagate_fourier_covariance(power,n,a,chunk_columns=3)
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=1e-15)
    # Independent draws validate covariance, including off-diagonal entries.
    rng=np.random.default_rng(35);white=rng.normal(size=(12000,n))
    draws=np.fft.irfft(np.fft.rfft(white,axis=1)*np.sqrt(power/n),n=n,axis=1)
    folded=(a@draws.T).T
    np.testing.assert_allclose(np.cov(folded,rowvar=False),actual,
        atol=.04*np.max(np.diag(actual)),rtol=.06)


def test_empty_phase_bins_are_removed_not_filled():
    mask=np.ones(30,bool);mask[[2,12,22]]=False
    a,ids,counts=folding_operator(30,10.,np.array([1,2,3]),mask)
    assert counts[2]==0 and 2 not in ids and a.shape==(9,30)
    np.testing.assert_allclose(a@np.ones(30),1.)


def test_dropped_modes_equal_explicit_reduced_gaussian_with_missing_bins():
    rng=np.random.default_rng(6);n=24;keep=np.arange(n)!=5
    modes=dropped_fourier_modes(n,roll=3)[keep]
    x=rng.normal(size=(keep.sum(),keep.sum()));cov=x@x.T+np.eye(keep.sum())
    flux=1+rng.normal(size=keep.sum());model=1+rng.normal(size=keep.sum())
    b,rank=marginalized_mode_block(np.arange(keep.sum()),flux,cov,1.,modes)
    basis=null_space(modes.T)
    expected=multivariate_normal.logpdf(basis.T@(flux-model),cov=basis.T@cov@basis)
    assert rank==4 and b.precision.flags.c_contiguous
    assert JointFourierMetric([b]).loglike(model)==pytest.approx(expected,abs=1e-10)
    np.testing.assert_allclose(b.precision@modes,0.,atol=1e-12)


def test_complete_fold_preserves_original_dropped_dc_nyquist_cost():
    rng=np.random.default_rng(8);n=32;variance=.01
    flux=1+rng.normal(size=n)*.1;model=1+rng.normal(size=n)*.03
    roll=5;modes=dropped_fourier_modes(n,roll=roll)
    b,_=marginalized_mode_block(np.arange(n),flux,np.eye(n)*variance,1.,modes)
    d=np.roll(flux,roll);m=np.roll(model,roll);expected=0.
    for lo in (0,n//2):
        df=np.fft.rfft(d[lo:lo+n//2])[1:-1]
        mf=np.fft.rfft(m[lo:lo+n//2])[1:-1]
        expected+=np.sum((abs(df)**2-abs(df-mf)**2)/(n//2*variance))
    assert JointFourierMetric([b]).gain(model)==pytest.approx(expected,abs=1e-11)
