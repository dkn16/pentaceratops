"""Original full-period Fourier scenario means with corrected folded covariance."""
import numpy as np
from .fourier_baseline import FourierBaselineObservation


class FoldedBaselineObservation(FourierBaselineObservation):
    def render(self, parameters):
        lk=self.adapter.lk;obs=self.original
        for name in ('_tm_cache','_tm_sec_cache'):getattr(lk,name)['id_time']=None
        if self.kind=='planet':
            first=lk.simulate_TP_transit(obs.time,**parameters)
            models=[np.r_[first,np.ones(len(first))]]
        elif self.kind=='binary':
            first,second=lk.simulate_EB_transit_secondary(obs.time,obs.secondary[0],**parameters)
            models=[np.r_[first,second]]
        else:
            first,second=lk.simulate_EB_transit_evenodd(obs.time,obs.secondary[0],**parameters)
            models=[np.r_[first,second],np.r_[second,first]]
        if self.kind!='x2p':
            models=[np.roll(model,-self.adapter.fold_roll) for model in models]
        if any(len(model)!=self.adapter.fold_length for model in models):
            raise ValueError('Original physical model grid changed')
        ids=self.adapter.fold_indices
        return [1+self.adapter.aperture_fraction*(model[ids]-1) for model in models]
