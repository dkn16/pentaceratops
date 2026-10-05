"""Scenario sampling for the joint Fourier model; isolated workers only.

Population and geometric recipes are reused from the package. The photometric
callback uses every observed sample, so optional legacy window arguments never
supply extra data or a synthetic secondary constraint.
"""
import importlib

import numpy as np
from scipy.special import logsumexp

from ..experimental.covariance import patched_engine, scenario_functions
from ..experimental.sampler_mode import mode_context

FAMILIES=("TP","EB","EBx2P","PTP","PEB","PEBx2P","STP","SEB","SEBx2P",          "DTP","DEB","DEBx2P","BTP","BEB","BEBx2P")


def sample_joint_scenario(adapter, scenario, star, *, trilegal, molusc=None,
                          N=500, steps=50, seed=42, posterior_samples=2000):
    """Retain complete weighted pools, resamples, and an independently replayed model."""
    name=scenario[1:] if scenario.startswith("N") else scenario
    if name not in FAMILIES:
        raise ValueError("Unknown scenario")
    if int(posterior_samples)!=posterior_samples or posterior_samples<1:
        raise ValueError("posterior_samples must be positive")
    fn=scenario_functions()[name]
    module=importlib.import_module(fn.__module__)
    from . import real,eclipses
    kwargs=adapter.scenario_kwargs(fn,star,trilegal=trilegal,molusc=molusc,N=N,steps=steps)
    old_counts=real.POSTERIOR_NSAMPLES,eclipses.POSTERIOR_NSAMPLES
    real.POSTERIOR_NSAMPLES=eclipses.POSTERIOR_NSAMPLES=int(posterior_samples)
    state=np.random.get_state();np.random.seed(seed)
    result={}
    adapter.reset_profile();adapter.calls=dict(planet=0,binary=0,x2p=0)
    try:
        with mode_context(module,adapter,"batch_prior") as stats,patched_engine(adapter):
            original=module._run_persistent_evidence
            def capture(loglike,prior_transform,ndim,**options):
                z,sampler=original(loglike,prior_transform,ndim,**options)
                positions=np.asarray(sampler.samples);weights=np.asarray(sampler.weights)
                targets=np.asarray(sampler.log_likelihoods)
                finite=np.isfinite(targets)&np.isfinite(weights)&(weights>0)
                result.update(lnBF=float(z),physical_parameters=positions.copy(),
                    weights=weights.copy(),log_target=targets.copy())
                if np.isfinite(z) and finite.any():
                    ids=np.flatnonzero(finite);prob=weights[finite]/weights[finite].sum()
                    chosen=np.random.default_rng(seed+9281).choice(ids,size=posterior_samples,p=prob)
                    result.update(posterior_parameters=positions[chosen].copy(),
                        posterior_log_target=targets[chosen].copy(),posterior_source_index=chosen)
                    best=int(np.argmax(np.where(finite,targets,-np.inf)))
                    adapter.record=True;adapter.snapshot=None
                    try: replay=loglike(positions[best])
                    finally: adapter.record=False
                    if not np.isclose(replay,targets[best],atol=2e-5,rtol=2e-12):
                        raise RuntimeError("Scalar best-fit replay differs from retained log target")
                    if adapter.snapshot is None:raise RuntimeError("No physical best-fit model")
                    result["bestfit"]=dict(adapter.snapshot,theta=positions[best].copy(),
                        log_target=float(targets[best]),scalar_replay=float(replay))
                return z,sampler
            module._run_persistent_evidence=capture
            try: physical=fn(**kwargs)
            finally: module._run_persistent_evidence=original
        if not result:
            if physical.get("lnZ")!=-np.inf:raise RuntimeError("Unexpected sampler bypass")
            result.update(lnBF=-np.inf,physical_parameters=np.empty((0,0)),weights=np.empty(0),
                          log_target=np.empty(0),empty_support=True)
        elif not stats["prior_batches"] or not stats["likelihood_batches"]:
            raise RuntimeError("Optimized callbacks were not used")
        result.update(lnZ=result["lnBF"]+adapter.metric.null_loglike,
            null_loglike=adapter.metric.null_loglike,scenario=scenario,seed=int(seed),N=int(N),steps=int(steps),
            backend=getattr(adapter,"backend","joint_fourier_observed_strict_batch"),physical_result=physical,
            sampling_stats=dict(stats),profile=dict(adapter.profile),parity=adapter.parity)
        return result
    finally:
        real.POSTERIOR_NSAMPLES,eclipses.POSTERIOR_NSAMPLES=old_counts
        np.random.set_state(state)


def calc_probs_joint_fourier(target,P_orb,blocks,epoch,*,trilegal_fname=None,molusc_file=None,
                            N=500,steps=50,nsamples=7,seed=42,posterior_samples=2000,
                            parity="profile",eb_eta=1.,filt=None):
    """Return all scenario Bayes factors/FPP and posterior records for one target.

    blocks contain observed flux and PSD covariance, constructed with
    GaussianBlock.from_spectrum or preprocessing.joint_fourier.prepare_sector.
    This opt-in API neither interpolates gaps nor requires a secondary window.
    Threads are unsupported; parallel campaigns should use isolated processes.
    """
    import pandas as pd
    from ..experimental.joint_fourier import JointFourierAdapter
    if not np.isfinite(eb_eta) or eb_eta<=0:raise ValueError("eb_eta must be positive")
    stars=target.stars.loc[target.stars.tdepth>0].reset_index(drop=True)
    if stars.empty:raise ValueError("No eligible stars; run calc_depths first")
    population=trilegal_fname or getattr(target,"trilegal_fname",None)
    if population is None:raise ValueError("Supply the shared TRILEGAL population")
    adapter=JointFourierAdapter(blocks,P_orb,epoch,parity=parity,nsamples=nsamples,
        mission=getattr(target,"mission","TESS"),filt=filt)
    rows=[];records={}
    for i,star in stars.iterrows():
        star=star.copy()
        if i:
            for key,value in (("mass",1.),("rad",1.),("Teff",5777.)):
                if not np.isfinite(star[key]):star[key]=value
        adapter.aperture_fraction=float(star.fluxratio)
        for scenario in FAMILIES if i==0 else ("NTP","NEB","NEBx2P"):
            result=sample_joint_scenario(adapter,scenario,star,trilegal=population,
                molusc=molusc_file if i==0 else None,N=N,steps=steps,seed=seed+1009*len(rows),
                posterior_samples=posterior_samples)
            records[(int(star.ID),scenario)]=result
            rows.append(dict(ID=int(star.ID),scenario=scenario,lnBF=result["lnBF"],lnZ=result["lnZ"]))
    table=pd.DataFrame(rows);binary=table.scenario.str.contains("EB").to_numpy()
    values=table.lnBF.to_numpy()+np.where(binary,np.log(eb_eta),0.)
    if not np.isfinite(logsumexp(values)):raise RuntimeError("No finite scenario evidence")
    table["probability"]=np.exp(values-logsumexp(values))
    accepted=table.scenario.isin(("TP","PTP","DTP"))
    table.attrs.update(FPP=float(table.loc[~accepted,"probability"].sum()),
        FPP_EB=float(table.loc[binary,"probability"].sum()),eb_eta=eb_eta,
        posterior_records=records,backend="joint_fourier_observed_strict_batch")
    return table
