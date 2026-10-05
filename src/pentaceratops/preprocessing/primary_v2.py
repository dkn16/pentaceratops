"""Conditional-FGP preprocessing for observed even/odd primary windows.

Uses the existing package FGP and exact coefficient posterior. Selection and
training protection depend on times and errors, never flux/FPP. No missing
secondary observations are synthesized. Independent sectors share zero
cross-sector covariance; the folded even/odd cross-covariance is preserved.
"""
import numpy as np
from scipy.sparse import csr_matrix
from . import fgp
from .posterior import conditional_posterior

PANEL_KEYS = {'primary': ('time','flux','err'),
              'even': ('time_even','flux_even','err_even'),
              'odd': ('time_odd','flux_odd','err_odd')}

def fold_operator(relative_time, error, centers, dt, choose=None):
    """Match the original regular-bin fold, even when retained bins have gaps.

    Do not construct edges directly from a compressed, nonuniform center
    list: that would let an absent bin's observations leak into its neighbor.
    """
    relative_time, error, centers = map(np.asarray, (relative_time, error, centers))
    if len(centers) == 0:
        return csr_matrix((0, len(relative_time))), np.zeros(0, int)
    if dt <= 0 or np.any(np.diff(centers) <= 0):
        raise ValueError("Require positive cadence and increasing bin centers")
    ids = np.rint(centers / dt).astype(int)
    np.testing.assert_allclose(centers, ids * dt, atol=1e-10, rtol=0)
    full = np.arange(ids[0], ids[-1] + 1) * dt
    edges = np.concatenate(([full[0] - dt / 2], full + dt / 2))
    slots = np.digitize(relative_time, edges) - 1
    lookup = np.full(len(full), -1, int)
    lookup[ids - ids[0]] = np.arange(len(ids))
    valid = np.isfinite(relative_time) & np.isfinite(error) & (error > 0)
    valid &= (slots >= 0) & (slots < len(full))
    if choose is not None:
        valid &= np.asarray(choose, bool)
    indices = np.flatnonzero(valid)
    rows = lookup[slots[indices]]
    retained = rows >= 0
    rows, indices = rows[retained], indices[retained]
    weights = 1 / error[indices]**2
    denominator = np.bincount(rows, weights=weights, minlength=len(centers))
    counts = np.bincount(rows, minlength=len(centers))
    matrix = csr_matrix((weights / denominator[rows], (rows, indices)),
                         shape=(len(centers), len(relative_time)))
    return matrix, counts



def phase_selections(data):
    t = np.asarray(data['time_all'],float)
    epoch,period = float(data['epoch_btjd']),float(data['period'])
    cycle = np.rint((t-epoch)/period).astype(int)
    relative = t-(epoch+cycle*period)
    return {p:(relative,np.ones(len(t),bool) if p=='primary' else cycle%2==int(p=='odd')) for p in PANEL_KEYS}


def likelihood_masks(data,time,observed):
    """Protect exactly the observed samples entering even/odd fold operators."""
    time,observed = np.asarray(time,float),np.asarray(observed,bool)
    if time.ndim!=1 or observed.shape!=time.shape or not np.isfinite(time).all():
        raise ValueError('Require a finite regular grid and matching observed mask')
    if len(data['time_secondary']):
        raise ValueError('Primary-only selection refuses to discard secondary observations')
    cycles=np.rint((time-float(data['epoch_btjd']))/float(data['period'])).astype(int)
    relative=time-(float(data['epoch_btjd'])+cycles*float(data['period']))
    dt=float(data.get('fold_bin_days',data['bin_days']))
    masks={}
    for panel in ('even','odd'):
        grid=np.asarray(data['time_'+panel],float)
        if not len(grid):raise ValueError('Missing observed primary parity: '+panel)
        operator,_=fold_operator(relative,np.ones(len(time)),grid,dt,observed&(cycles%2==int(panel=='odd')))
        masks[panel]=np.asarray(operator.getnnz(axis=0)).ravel()>0
    if np.any(masks['even']&masks['odd']):raise ValueError('Primary parity windows overlap')
    return masks,masks['even']|masks['odd']


def refold(relative,flux,error,grid,dt,choose):
    """Preserve the historical inverse-variance mean and empirical error floor."""
    op,count=fold_operator(relative,error,grid,dt,choose)
    if np.any(count==0):raise ValueError('Retained phase grid contains an unobserved bin')
    mean=np.asarray(op@flux).ravel();sigma=np.empty(len(grid))
    for i in range(len(grid)):
        ids=op.indices[op.indptr[i]:op.indptr[i+1]]
        formal=np.sqrt(1/np.sum(1/np.asarray(error)[ids]**2))
        empirical=np.std(np.asarray(flux)[ids],ddof=1)/np.sqrt(len(ids)) if len(ids)>1 else 0.
        sigma[i]=max(formal,empirical)
    return mean,sigma,count


def audit_windows(data, minimum_training=140):
    dt,tau=float(data.get('fold_bin_days',data['bin_days'])),float(data['duration_days'])
    rows=[]
    for panel,(relative,choose) in phase_selections(data).items():
        grid=np.asarray(data[PANEL_KEYS[panel][0]])
        if not len(grid):raise ValueError(panel+': empty window')
        op,count=fold_operator(relative,data['err_all'],grid,dt,choose)
        support=np.asarray(op.getnnz(axis=0)).ravel()>0
        central=choose&(relative>=-tau/2)&(relative<tau/2)
        if not central.any() or np.any(central&~support) or np.any(count==0):
            raise ValueError(panel+': incomplete observed eclipse or empty bins')
        if grid[0]-dt/2>-tau/2+1e-12 or grid[-1]+dt/2<tau/2-1e-12:
            raise ValueError(panel+': grid does not enclose nominal eclipse')
        rows.append(dict(panel=panel,n_bins=len(grid),n_points=int(support.sum()),
                         in_eclipse_points=int(central.sum()),missing_eclipse_points=0,empty_bins=0))
    _,protected=likelihood_masks(data,data['time_all'],np.ones(len(data['time_all']),bool))
    sectors=[]
    for sec in np.unique(data['sector_all']):
        use=data['sector_all']==sec;n=int(use.sum());held=int((protected&use).sum())
        if n-held<minimum_training:raise ValueError(f'Sector {sec}: {n-held} training bins < {minimum_training}')
        sectors.append(dict(sector=int(sec),n_observed=n,n_protected=held,n_training=n-held,n_training_overlap=0))
    return dict(panels=rows,sectors=sectors)


def select_primary_windows(data):
    """Retain original primary grids, shortening only for exact GP holdout."""
    if len(data['time_secondary']):raise ValueError('Secondary observations exist; use the three-panel mode')
    dt,tau=float(data['bin_days']),float(data['duration_days'])
    original={p:np.array(data[k[0]],copy=True) for p,k in PANEL_KEYS.items()}
    if any(not len(g) for g in original.values()):raise ValueError('Missing primary/even/odd grid')
    selections=phase_selections(data);initial={}
    for p,g in original.items():
        rel,choose=selections[p]
        full=np.arange(round(g[0]/dt),round(g[-1]/dt)+1)*dt
        _,count=fold_operator(rel,data['err_all'],full,dt,choose)
        initial[p]=full[count>0]
    limits=[None]+sorted({float(abs(x)+dt/2) for g in initial.values() for x in g
                         if tau-1e-12<=abs(x)+dt/2<max(abs(g))+dt/2-1e-12},reverse=True)
    last='no candidate window'
    for half in limits:
        grids={p:g.copy() if half is None else g[abs(g)<=half-dt/2+1e-12] for p,g in initial.items()}
        if half is not None and any(not len(g) or min(-g[0],g[-1])+dt/2<tau-1e-12 for g in grids.values()):continue
        trial=dict(data,fold_bin_days=np.array(dt))
        for p,g in grids.items():trial[PANEL_KEYS[p][0]]=g
        try:audit=audit_windows(trial)
        except ValueError as error:last=str(error);continue
        for p,g in grids.items():
            tk,fk,ek=PANEL_KEYS[p];rel,choose=selections[p]
            data[tk]=g;data[fk],data[ek],_=refold(rel,data['flux_all'],data['err_all'],g,dt,choose)
        actual=min(min(-g[0],g[-1])+dt/2 for g in grids.values())
        data.update(fold_bin_days=np.array(dt),window_half_days=np.array(actual),v2_likelihood_mode=np.array('primary_only'))
        return dict(policy='original_primary_grids_if_feasible',trimmed_for_training=half is not None,
                    half_window_days=float(actual),minimum_training_per_sector=140,
                    restored_missing_bins={p:len(initial[p])-len(original[p]) for p in initial},**audit)
    raise ValueError('Cannot retain observed primary eclipses and sufficient training: '+last)


def binned_sector_series(time,flux,error,dt):
    order=np.argsort(time);time,flux,error=[np.asarray(a)[order] for a in (time,flux,error)]
    origin=np.floor(time[0]/dt)*dt;index=np.rint((time-origin)/dt).astype(int)
    n=int(index.max())+1;t=origin+np.arange(n)*dt
    f=np.full(n,np.nan);e=np.full(n,np.nan)
    for j in np.unique(index):
        use=(index==j)&np.isfinite(flux)&np.isfinite(error)&(error>0)
        if not use.any():continue
        w=1/error[use]**2;f[j]=np.sum(w*flux[use])/w.sum();e[j]=np.sqrt(1/w.sum())
    return t,f,e


def prepare_primary_v2(data,metadata,*,sector_output=None,draws=2048,seed=697901):
    """Return primary-only prepared arrays, folded posterior, and diagnostics.

    All input data are copied. Original files and ephemerides are unchanged.
    This is a cut conditional-FGP likelihood, with the same scalar median
    observation error per folded panel as the established v2 comparison.
    """
    data={k:np.array(v,copy=True) for k,v in data.items()}
    selection=select_primary_windows(data);prep=metadata['preprocessing'];fits=[]
    dt=float(data['bin_days'])
    for sec in np.unique(data['sector_all']):
        keep=(data['sector_native']==sec)&~data['other_planet_mask_native'].astype(bool)
        t,raw,error=binned_sector_series(data['time_native'][keep],data['flux_raw_native'][keep],data['err_native'][keep],dt)
        observed=np.isfinite(raw)&np.isfinite(error)&(error>0);reference=data['sector_all']==sec
        np.testing.assert_allclose(t[observed],data['time_all'][reference],atol=1e-8,rtol=0)
        np.testing.assert_allclose(raw[observed],data['flux_raw_all'][reference],atol=1e-12,rtol=1e-10)
        masks,protected=likelihood_masks(data,t,observed)
        if np.sum(observed&~protected)<140:raise ValueError('Insufficient training after exact window protection')
        padding=int(np.ceil(prep['fgp_padding_fraction']*len(t)))
        t=np.r_[t,t[-1]+dt*np.arange(1,padding+1)]
        raw=np.pad(raw,(0,padding),constant_values=np.nan);error=np.pad(error,(0,padding),constant_values=np.nan)
        protected=np.pad(protected,(0,padding),constant_values=False)
        result=fgp.detrend_regular_segment(t,raw,error,protected,dt,iterations=prep['fgp_iterations'],cutoff_per_day=prep['fgp_cutoff_per_day'])
        obs=result['observed'];training=result['fit_mask']
        np.testing.assert_array_equal(training,obs&~protected)
        np.testing.assert_allclose(result['cleaned_flux'][protected],raw[protected]/result['center'],atol=1e-12,rtol=0)
        posterior=conditional_posterior(result['fit_values'],training,result['stellar_psd'],result['white_variance'],dt,prep['fgp_cutoff_per_day'])
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(sec)]))
        coefficients,logp=posterior.sample(draws,rng)
        scale=result['scale']/result['center'];design=posterior.design[obs]*scale
        factor=posterior.coefficient_factor;mean=posterior.coefficient_mean
        for new,old in [('cleaned_flux','flux_cleaned_all'),('model_flux','fgp_model_all'),('residual_flux','flux_all'),('relative_error','err_all')]:
            data[old][reference]=result[new][obs]
        diagnostic=dict(sector=int(sec),n_observed=int(obs.sum()),n_training=int(training.sum()),
                        n_protected=int(protected.sum()),n_training_overlap=0,n_modes=len(mean)//2,
                        secondary_observations=0,protection_mode='primary_even_odd_windows')
        fits.append(dict(reference=reference,design=design,coefficient_mean=mean,coefficient_factor=factor,
                         coefficient_draws=coefficients,fit_mask=training[obs],diagnostics=diagnostic))
        if sector_output is not None:
            from pathlib import Path
            out=Path(sector_output);out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out/f'sector_{int(sec)}_posterior.npz',time=t,observed=obs,protected=protected,
                fit_mask=training,training_values=result['fit_values'],stellar_psd=result['stellar_psd'],
                white_variance=result['white_variance'],coefficient_mean=mean,coefficient_covariance_factor=factor,
                coefficient_draws=coefficients,log_posterior=logp,center=result['center'],scale=result['scale'])
    selections=phase_selections(data);saved={};training=np.zeros(len(data['time_all']),bool)
    for fit in fits:training[fit['reference']]=fit['fit_mask']
    for panel,(tk,fk,ek) in PANEL_KEYS.items():
        rel,choose=selections[panel];grid=data[tk]
        old,error,count=refold(rel,data['flux_all'],data['err_all'],grid,dt,choose)
        op,counts=fold_operator(rel,data['err_all'],grid,dt,choose)
        np.testing.assert_array_equal(count,counts)
        if panel in ('even','odd') and op[:,training].nnz:raise ValueError('FGP reused a likelihood observation')
        factors=[];mean_trend=np.zeros(len(grid));perturbations=np.zeros((len(grid),draws))
        for fit in fits:
            basis=op[:,fit['reference']]@fit['design']
            mean_trend+=basis@fit['coefficient_mean']
            factors.append(basis@fit['coefficient_factor'])
            perturbations+=basis@(fit['coefficient_draws']-fit['coefficient_mean'][:,None])
        factor=np.concatenate(factors,axis=1);mean=np.asarray(op@data['flux_cleaned_all']).ravel()-mean_trend
        # Independent posterior draws verify propagation to the folded panels.
        sd=np.linalg.norm(factor,axis=1);positive=sd>0
        discrepancy=np.median(abs(perturbations.std(axis=1,ddof=1)[positive]/sd[positive]-1)) if positive.any() else 0.
        if discrepancy>.12:raise ValueError('Folded draws disagree with analytic covariance')
        data[fk],data[ek]=old,error
        for key,value in [('time',grid),('map_flux',old),('input_error',error),('conditional_mean',mean),
                          ('covariance_factor',factor),('draws',mean[:,None]-perturbations)]:saved[panel+'_'+key]=value
    ncoef=saved['even_covariance_factor'].shape[1]
    for key in ('time','map_flux','input_error','conditional_mean'):saved['secondary_'+key]=np.empty(0)
    saved['secondary_covariance_factor']=np.empty((0,ncoef));saved['secondary_draws']=np.empty((0,draws))
    data['fgp_protect_secondary']=np.array(False);data['fgp_protection_mode']=np.array('primary_even_odd_windows')
    audit=audit_windows(data)
    return data,saved,dict(window_selection=selection,window_audit=audit,sectors=[f['diagnostics'] for f in fits],
                           likelihood_mode='primary_only',fgp_protection_mode='primary_even_odd_windows',
                           n_coefficient_columns=ncoef,draws_per_sector=draws)
