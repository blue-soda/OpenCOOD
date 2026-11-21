"""Test whether an actual recent recurrent target and gate still describe a quiet cell."""
from collections import Counter
import torch
from opencood.models.sub_modules.int_feature_memory import warp_bev


class SilentGateAudit:
    def __init__(self):
        self.scenes={}
        self.reset()

    def reset(self):
        self.cache=None
        self.pose=None
        self.last=None

    def capture(self,ids,gate,candidate):
        self.last=ids,gate,candidate

    def observe(self,meta,area,old,valid,quiet,full_sites,actual,drive,teacher):
        channels=old.shape[1]
        if self.cache is None:
            aligned=old.new_zeros((1,2*channels+1,*old.shape[-2:]))
        else:
            aligned=warp_bev(self.cache,self.pose,meta['pose'],area)[0]
        ids=quiet.flatten().nonzero().flatten()[::16]
        mask=aligned[0,-1].flatten()[ids].bool()
        covered=ids[mask]
        previous=old[0].flatten(1)[:,covered]
        target,gate=aligned[0,:-1].flatten(1)[:,covered].split(channels,0)
        expected=teacher[0].flatten(1)[:,covered]
        base_error=(teacher-drive)[0].flatten(1)[:,ids].double().square()
        missing=float(base_error[:,~mask].sum())
        shift=len(covered)//2
        alternatives=dict(hold=previous,cached=previous+gate*(target-previous),
                          shuffled_gate=previous+gate.roll(shift,1)*(target-previous),
                          shuffled_target=previous+gate*(target.roll(shift,1)-previous),overwrite=target)
        c=self.scenes.setdefault(meta['scene'],Counter())
        c.update(samples=len(ids),covered=len(covered),drive=float(base_error.sum()))
        for n,value in alternatives.items():
            c[n]+=missing+float((expected-value).double().square().sum())

        # Only the cells that actually executed the recurrent convolution can update history.
        sites,current_gate,current_target=self.last
        assert torch.equal(sites,full_sites.flatten().nonzero().flatten())
        use=valid.flatten()[sites].bool()
        sites=sites[use]
        current_gate,current_target=current_gate[use].T,current_target[use].T
        prior=old[0].flatten(1)[:,sites]
        assert torch.allclose(prior+current_gate*(current_target-prior),actual[0].flatten(1)[:,sites],atol=1e-5,rtol=1e-5)
        aligned[0].flatten(1)[:channels,sites]=current_target
        aligned[0].flatten(1)[channels:-1,sites]=current_gate
        aligned[0,-1].flatten()[sites]=1
        self.cache,self.pose,self.last=aligned,meta['pose'],None
