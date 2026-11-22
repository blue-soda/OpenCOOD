"""Maintain a shallow convolution relative to an exact refresh-frame reference."""
import torch

from opencood.models.sub_modules.int_shallow_pulse import ShallowPulseConv, receptive_counts


class AnchoredObservationConv(ShallowPulseConv):
    def __init__(self, original, scales=None, signed=True):
        super().__init__(original,signed=signed)
        self.register_buffer('raw_weights',original.weight.detach().permute(2,3,1,0).contiguous())
        if scales is None:
            self.step=self.quantum
        else:
            table=original.weight.new_tensor(scales)
            if table.shape!=(128,) or not bool(((table==.125)|(table==.25)|(table==.5)).all()):
                raise ValueError('Expected 128 positive dyadic channel quanta')
            self.register_buffer('step',table.view(1,128,1,1))
            self.scaled.copy_(self.raw_weights*table.view(1,1,128,1))
        self.anchor_state = None

    def begin(self, name, reason, meta, input_shape, bounds, verify):
        self.name,self.verify = name,verify
        if reason is not None:
            self.anchor_state = None

    def accumulate(self, code, single, multi, sites, n_single, n_multi, buffer):
        ids = torch.arange(sites.numel(),device=sites.device)
        if not self.signed:
            mac=self.fma.run(code,self.scaled,ids[sites & ((n_single+n_multi)>0)],buffer)
            assert mac==int((n_single[sites]+n_multi[sites]).sum())
            return 0,mac*128
        ac = self.add.run(code.masked_fill(~single,0),self.scaled,ids[sites & (n_single>0)],buffer)
        mac = self.fma.run(code.masked_fill(~multi,0),self.scaled,ids[sites & (n_multi>0)],buffer)
        assert (ac,mac)==(int(n_single[sites].sum()),int(n_multi[sites].sum()))
        return ac*128,mac*128

    def forward(self, x):
        assert not torch.is_grad_enabled() and x.dtype==torch.float32 and x.shape[:2]==(1,128)
        h,w=x.shape[2:]
        ids=torch.arange(h*w,device=x.device)
        rec=dict(reference_ac=0,reference_mac=0,delta_ac=0,delta_mac=0,reference_sites=0,
                 delta_sites=0,clean_sites=0,floor_clamps=0,values=x.numel())
        if self.anchor_state is None:
            base=x.clone()
            codes=torch.zeros_like(x)
            absolute=x
            output=x.new_empty(h*w,128)
            rec['rebuild_mac']=128*self.static.run(x,self.raw_weights,ids,output)
            rec['rebuild_sites']=len(ids)
            ref_output=output
        else:
            base,ref_output,old_codes,last_output=self.anchor_state
            if self.name=='anchor_absolute':
                absolute=(x/self.step).round()*self.step
                codes=(absolute-base)/self.step
            else:
                codes=(x-base)/self.step
                if self.name=='anchor_relative':
                    raw=codes.round()
                    lower=(-base/self.step).ceil()
                    rec['floor_clamps']=int((raw<lower).sum())
                    codes=torch.maximum(raw,lower)
                    absolute=base+codes*self.step
                else:
                    assert self.name=='anchor_float'
                    absolute=x
            assert float(absolute.min())>=0
            change=codes-old_codes
            one,many=change.abs()==1,(change!=0)&(change.abs()!=1)
            off_one,off_many=codes.abs()==1,(codes!=0)&(codes.abs()!=1)
            nonzero,du,dm=receptive_counts(absolute,one,many)
            _,ru,rm=receptive_counts(absolute,off_one,off_many)
            dirty=(du+dm)>0
            dense_cost,ref_cost,delta_cost=2*nonzero,ru+2*rm,du+2*dm
            if not self.signed:
                ref_cost,delta_cost=2*(ru+rm),2*(du+dm)
            fresh=dirty & (dense_cost<=ref_cost) & (dense_cost<=delta_cost)
            use_ref=dirty & ~fresh & (ref_cost<=delta_cost)
            use_delta=dirty & ~fresh & ~use_ref
            output=last_output.clone()
            output[use_ref]=ref_output[use_ref]
            rec['rebuild_mac']=128*self.static.run(absolute,self.raw_weights,ids[fresh],output)
            assert rec['rebuild_mac']==128*int(nonzero[fresh].sum())
            rec['reference_ac'],rec['reference_mac']=self.accumulate(codes,off_one,off_many,use_ref,ru,rm,output)
            rec['delta_ac'],rec['delta_mac']=self.accumulate(change,one,many,use_delta,du,dm,output)
            rec.update(rebuild_sites=int(fresh.sum()),reference_sites=int(use_ref.sum()),
                       delta_sites=int(use_delta.sum()),clean_sites=int((~dirty).sum()))
        if self.verify:
            gold=torch.empty_like(output)
            self.static.run(absolute,self.raw_weights,ids,gold)
            error=(output-gold).abs()
            rec['max_error']=float(error.max())
            assert bool((error<=.0005+.0001*gold.abs()).all())
        rec['sites']=h*w
        rec['error_l1']=float((x-absolute).abs().mean())
        self.anchor_state=(base,ref_output,codes,output)
        unique={v.data_ptr():v for v in self.anchor_state}
        rec['state_bytes']=sum(v.numel()*v.element_size() for v in unique.values())
        self.record=rec
        out=output.T.reshape(1,128,h,w).contiguous()
        return out if self.bias is None else out+self.bias.view(1,-1,1,1)
