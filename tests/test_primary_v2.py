"""Primary-only covariance, observation selection and physical-model checks."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal
from test_exposure import synthetic_data, physical_columns, scalar_scores, assert_agreement
from pentaceratops.experimental.primary_v2 import PrimaryOnlyAdapter, load_primary_inputs
from pentaceratops.preprocessing.primary_v2 import (
    fold_operator, likelihood_masks, select_primary_windows, prepare_primary_v2,
)


def primary_data(exptime=10/1440):
    full=synthetic_data(exptime);data=dict(full)
    n=len(full['even_time'])+len(full['odd_time'])
    for key in ('time','mean_flux','sigma'):data['secondary_'+key]=np.empty(0)
    for key in ('flux','map_flux','sigma'):data[key]=full[key][:n].copy()
    data['factor']=full['factor'][:n].copy()
    return data,full


@pytest.mark.parametrize('kind',['planet','binary','x2p'])
@pytest.mark.parametrize('parity',['profile','marginalize'])
@pytest.mark.parametrize('exptime',[2/1440,30/1440])
def test_scalar_batch_agree_without_secondary(kind,parity,exptime):
    data,_=primary_data(exptime);adapter=PrimaryOnlyAdapter(data,parity=parity,chunk_size=3)
    columns=physical_columns(kind)
    for fraction in (1.,.07):
        adapter.aperture_fraction=fraction
        assert_agreement(scalar_scores(adapter,kind,columns),adapter.photometric_columns(kind,columns))


def test_primary_metric_is_marginal_not_flat_secondary_condition():
    data,full=primary_data()
    # Strong correlated trend makes incorrectly conditioning on absent
    # secondary values distinguishable from retaining the observed marginal.
    factor=np.ones((len(full['flux']),2))*.004
    full['factor']=factor;data['factor']=factor[:len(data['flux'])]
    adapter=PrimaryOnlyAdapter(data)
    whole=np.diag(full['sigma']**2)+factor@factor.T
    covariance=whole[:len(data['flux']),:len(data['flux'])]
    model=1-.002*np.exp(-np.arange(len(data['flux']))/10)
    expected=multivariate_normal.logpdf(data['flux'],mean=model,cov=covariance)
    null=multivariate_normal.logpdf(data['flux'],mean=np.ones(len(model)),cov=covariance)
    assert adapter.metric.gain(model)==pytest.approx(expected-null,abs=1e-10)
    assert np.max(abs(adapter.metric.precision-np.linalg.inv(whole)[:len(model),:len(model)]))>1
    assert np.any(abs(covariance[:29,29:])>0)  # Retain cross-parity correlations.


def test_no_secondary_depth_veto_or_phantom_likelihood(monkeypatch):
    data,_=primary_data();a=PrimaryOnlyAdapter(data)
    p={k:v[:1] for k,v in physical_columns('binary').items()}
    a.record=True
    before=scalar_scores(a,'binary',p)
    original=a.lk.simulate_EB_transit
    def different_unobserved_depth(*args,**kwargs):
        primary,_=original(*args,**kwargs)
        return primary,1e9
    monkeypatch.setattr(a.lk,'simulate_EB_transit',different_unobserved_depth)
    np.testing.assert_array_equal(before,scalar_scores(a,'binary',p))
    assert a.snapshot['phantom_loglike_ratio']==0
    assert a.snapshot['secondary_omitted']


def test_nonempty_secondary_and_missing_parity_rejected():
    data,full=primary_data()
    with pytest.raises(ValueError,match='no secondary'):PrimaryOnlyAdapter(full)
    data['odd_time']=np.empty(0)
    with pytest.raises(ValueError,match='even and odd'):PrimaryOnlyAdapter(data)


def test_gapped_fold_does_not_reassign_absent_bins():
    t=np.array([-.22,-.18,-.02,.02,.18,.22]);e=np.array([1.,2.,1.,1.,1.,2.])
    op,count=fold_operator(t,e,np.array([-.2,.2]),.1)
    np.testing.assert_array_equal(count,[2,2])
    np.testing.assert_allclose(op.toarray(),[[.8,.2,0,0,0,0],[0,0,0,0,.8,.2]])


def raw_inputs():
    dt=.04
    t=np.r_[np.arange(-5.,5.+dt/2,dt),np.arange(15.,25.+dt/2,dt)]
    sector=np.repeat([1,2],len(t)//2)
    rel=t-np.rint(t/20)*20
    rng=np.random.default_rng(784)
    raw=1+.001*np.sin(t)-.003*np.exp(-.5*(rel/.045)**2)+rng.normal(0,.0002,len(t))
    err=np.full(len(t),.0003);grid=np.arange(-10,11)*dt
    d=dict(time_native=t,flux_raw_native=raw,err_native=err,sector_native=sector,
           other_planet_mask_native=np.zeros(len(t),bool),time_all=t,flux_raw_all=raw,
           flux_all=raw.copy(),err_all=err.copy(),fgp_model_all=np.zeros(len(t)),flux_cleaned_all=raw.copy(),
           sector_all=sector,bin_days=np.array(dt),period=np.array(20.),epoch_btjd=np.array(0.),
           duration_days=np.array(.16),exptime_days=np.array(dt),transit_depth=np.array(.003),
           time_secondary=np.empty(0),flux_secondary=np.empty(0),err_secondary=np.empty(0))
    for suffix in ('','_even','_odd'):
        d['time'+suffix]=grid.copy();d['flux'+suffix]=np.ones(len(grid));d['err'+suffix]=np.full(len(grid),.0003)
    return d,dict(preprocessing=dict(fgp_padding_fraction=.1,fgp_iterations=3,fgp_cutoff_per_day=.3))


def test_primary_preprocessing_preserves_holdout_and_covariance(tmp_path):
    data,metadata=raw_inputs();before=data['flux_all'].copy()
    prepared,folded,report=prepare_primary_v2(data,metadata,sector_output=tmp_path,draws=2048)
    np.testing.assert_array_equal(data['flux_all'],before)
    assert len(prepared['time_secondary'])==0 and folded['secondary_covariance_factor'].shape[0]==0
    for p in tmp_path.glob('sector_*_posterior.npz'):
        with np.load(p) as a:
            assert not np.any(a['protected']&a['fit_mask'])
            assert a['fit_mask'].sum()>=140
    assert report['window_audit']['panels'][1]['missing_eclipse_points']==0
    pp=tmp_path/'prepared.npz';fp=tmp_path/'folded.npz'
    np.savez_compressed(pp,**prepared);np.savez_compressed(fp,**folded)
    loaded=load_primary_inputs(dict(prepared_path=str(pp),folded_path=str(fp)))
    assert loaded['factor'].shape[0]==len(loaded['flux'])
    adapter=PrimaryOnlyAdapter(loaded)
    assert np.isfinite(adapter.metric.null_loglike)
    for panel in ('even','odd'):
        m=folded[panel+'_draws']-folded[panel+'_conditional_mean'][:,None]
        expected=folded[panel+'_covariance_factor']@folded[panel+'_covariance_factor'].T
        np.testing.assert_allclose(np.cov(m),expected,rtol=.2,atol=.08*expected.diagonal().max())
