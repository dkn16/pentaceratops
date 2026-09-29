"""Strict vector physics for all 15 target families (neighbors reuse T).

This reproduces the pinned scalar engine, including its flux conventions,
base-10 background prior and scenario-dependent collision rules. It does not
change the scientific model. Discrete background lookups are cached once.
"""
import inspect
import numpy as np


def closure(fn):
    return inspect.getclosurevars(fn).nonlocals


class PhysicalBatch:
    def __init__(self, scalar):
        self.env = e = closure(scalar)
        self.g = g = scalar.__globals__
        self.owner = scalar.__qualname__.split('.')[0]
        self.family = self.owner[4]
        supported = {f'lnZ_{p}TP' for p in 'TPSDB'} | {
            f'lnZ_{p}EB_{suffix}' for p in 'TPSDB' for suffix in ('secondary', 'evenodd')}
        if self.owner not in supported:
            raise ValueError(f'Not vectorized: {self.owner}')
        self.kind = 'x2p' if self.owner.endswith('_evenodd') else ('binary' if 'EB' in self.owner else 'planet')
        self.ndim = 5 if self.family == 'T' else 6
        if self.family in 'PS':
            self.companion = closure(e['companion_lnprior'])
        if self.family == 'S':
            name = '_comp_ldc' if self.kind == 'planet' else '_comp_props_and_ldc'
            self.ldc = ld = closure(e[name]).copy()
            if self.kind != 'planet':
                suffix = 'T' if ld['mission'] == 'TESS' else 'K'
                zs = g[f'ldc_{suffix}_Zs']
                table = g[f'ldc_{suffix}'][zs == zs[np.abs(zs-ld['Z']).argmin()]]
                ld.update(Teffs_at_Z=np.array(table.Teff, dtype=int), loggs_at_Z=np.array(table.logg, dtype=float),
                          u1s_at_Z=np.array(table.aLSM if suffix=='T' else table.a, dtype=float),
                          u2s_at_Z=np.array(table.bLSM if suffix=='T' else table.b, dtype=float))
        if self.family == 'D' or self.owner == 'lnZ_BTP':
            self.n_background = len(e['fluxratios_comp_T'])
            self.background_prior = np.array([e['lnprior_background_idx'](i) for i in range(self.n_background)])
        if self.owner == 'lnZ_BTP':
            self.background = np.array([e['_bg_props_and_ldc'](i) for i in range(self.n_background)])
        elif self.family == 'B':
            self.n_background = len(e['masses_comp'])
            z = e['Zs_comp'] if e['Zs_comp'] is not None else np.zeros(self.n_background)
            self.background_ldc = np.array([e['_ldc_for_bg'](float(l), float(t), float(v))
                for l,t,v in zip(e['loggs_comp'],e['Teffs_comp'],z)])
            self.bg_prior = closure(e['lnprior_background_combined'])
            self.background_dm = np.array([self.bg_prior['_delta_mag_primary_idx'](i) for i in range(self.n_background)])

    def flux_fraction(self, mass, target, filt=None):
        flux = self.g['flux_relation']
        f = flux(mass) if filt is None else flux(mass, filt)
        ft = flux(np.array([target])) if filt is None else flux(np.array([target]), filt)
        return f/(f+ft)

    def bound_prior(self, companion_mass, secondary_mass=None):
        c, g = self.companion, self.g
        filt = None if c['contrast_curve_file'] is None else c['filt']
        f = self.flux_fraction(companion_mass, c['M_s'], filt)
        combined = f/(1-f)
        if secondary_mass is not None:
            f2 = self.flux_fraction(secondary_mass, c['M_s'], filt)
            combined += f2/(1-f2)
        dm = 2.5*np.log10(combined)
        prior = g['lnprior_bound_TP' if self.kind=='planet' else 'lnprior_bound_EB'](
            c['M_s'], c['plx'], abs(dm), c['separations'], c['contrasts'])
        return np.where((prior > 0) | (dm > 0), -np.inf, prior)

    def background_binary_prior(self, index, mass, secondary_mass):
        c = self.bg_prior
        dm = self.background_dm[index]
        if c['separations'] is None:
            prior = np.full(len(index), min(0., np.log10((c['N_comp']/.1)*(1/3600)**2*2.2**2)))
        else:
            f = 10**(dm/2.5)/(1+10**(dm/2.5))
            bound = self.flux_fraction(mass, c['M_s'], c['filt'])
            eb = self.flux_fraction(secondary_mass, c['M_s'], c['filt'])
            eb *= np.divide(f, bound, out=np.zeros(len(f)), where=bound>0)
            combined = f/(1-f)+eb/(1-eb)
            dm_comb = 2.5*np.log10(combined)
            prior = np.minimum(0., self.g['lnprior_background'](c['N_comp'], abs(dm_comb), c['separations'], c['contrasts']))
        return np.where(dm>0, -np.inf, prior)

    def evaluate(self, theta):
        theta = np.asarray(theta, float)
        if theta.ndim != 2 or theta.shape[1] != self.ndim:
            raise ValueError('Unexpected physical parameter shape')
        base = np.full(len(theta), -np.inf)
        valid = np.isfinite(theta).all(axis=1)
        valid &= (theta[:,0]>0) & (theta[:,2]>=0) & (theta[:,2]<1) & (theta[:,4]>0)
        if self.family in 'PS': valid &= theta[:,5]>0
        if self.family in 'DB': valid &= (theta[:,5]>=0) & (theta[:,5]<self.n_background)
        ids = np.flatnonzero(valid)
        if not len(ids): return base, ids, {}
        t = theta[ids]
        P, inc, ecc, argp, size = t[:,:5].T
        e, g = self.env, self.g
        mstar = float(e.get('M_s',1.))  # BTP never needs target mass.
        n = len(ids)
        mass, radius = np.full(n,mstar), np.full(n,float(e.get('R_s',1.)))
        teff = np.full(n,float(e.get('Teff',5777.)))
        u1 = np.full(n,float(np.asarray(e.get('u1',.3)).item()))
        u2 = np.full(n,float(np.asarray(e.get('u2',.2)).item()))
        comp, prior = np.zeros(n), np.zeros(n)
        host = self.family in 'SB'
        if self.family in 'PS':
            companion_mass = t[:,5]*mstar
            comp = self.flux_fraction(companion_mass,mstar)
            if self.family == 'S':
                mass = companion_mass
                ld = self.ldc
                radius, temperature = g['stellar_relations'](mass,np.full(n,ld['R_s']),np.full(n,ld['Teff']))
                logg = np.log10(g['G']*mass*g['Msun']/(radius*g['Rsun'])**2)
                u1,u2 = g['nearest_ldc_coefficients'](temperature,logg,ld['Teffs_at_Z'],ld['loggs_at_Z'],ld['u1s_at_Z'],ld['u2s_at_Z'])
        elif self.family in 'DB':
            index = t[:,5].astype(np.int64)  # Original int(), not rounding.
            if self.family == 'D' or self.kind == 'planet':
                comp = np.asarray(e['fluxratios_comp_T'])[index]
                prior = self.background_prior[index]
            if self.owner == 'lnZ_BTP':
                mass,radius,u1,u2 = self.background[index].T
            elif self.family == 'B':
                mass = np.asarray(e['masses_comp'])[index]
                radius = np.sqrt(g['G']*mass*g['Msun']/(10**np.asarray(e['loggs_comp'])[index]))/g['Rsun']
                teff = np.asarray(e['Teffs_comp'])[index]
                u1,u2 = self.background_ldc[index].T
                comp = np.asarray(e['fr_primary_T'])[index]
        feb = np.zeros(n)
        if self.kind == 'planet':
            rp,reb = size,np.zeros(n)
            r2,total_mass = size*g['Rearth'],mass
        else:
            secondary_mass = size*mass
            reb,_ = g['stellar_relations'](secondary_mass,radius,teff)
            feb = self.flux_fraction(secondary_mass,mstar)
            if self.family == 'B':
                bound = self.flux_fraction(mass,mstar)
                feb *= np.divide(comp,bound,out=np.zeros(n),where=bound>0)
                prior = self.background_binary_prior(index,mass,secondary_mass)
            r2,total_mass = reb*g['Rsun'],mass+secondary_mass
            rp = np.zeros(n)
        if self.family in 'PS':
            prior = self.bound_prior(companion_mass,secondary_mass if self.family=='S' and self.kind!='planet' else None)
        a = ((g['G']*total_mass*g['Msun'])/(4*g['pi']**2)*(P*86400)**2)**(1/3)
        radius_sum = r2+radius*g['Rsun']
        corr = (1+ecc*np.sin(argp*g['pi']/180))/(1-ecc**2)
        ptransit = radius_sum/a*corr
        collision = 2*radius*g['Rsun'] if self.kind=='x2p' and self.family!='B' else radius_sum
        inc_min = np.degrees(np.arccos(np.minimum(1.,ptransit)))
        ok = (ptransit<=1) & (collision<=a*(1-ecc)) & (inc>=inc_min) & np.isfinite(prior)
        columns = dict(P_orb=P,inc=inc,ecc=ecc,argp=argp,a=a,R_s=radius,u1=u1,u2=u2,R_p=rp,R_EB=reb,
                       EB_fluxratio=feb,companion_fluxratio=comp,companion_is_host=np.full(n,host))
        base[ids[ok]] = -.5*g['ln2pi']-e['lnsigma']+prior[ok]
        return base,ids[ok],{k:np.asarray(v)[ok] for k,v in columns.items()}
