"""Analytic IF at PFN point activations before pillar max pooling.

The source uses functional ReLU after BatchNorm1d. Returning nonnegative
counts from its BN hook makes that original ReLU an identity. In identity
mode the original signed BN output is returned bitwise unchanged.
"""
import json
from pathlib import Path
import torch
from cosdh_front_stage import _InitialVoltageCount


def activate_pillar(x, scale, mode, initial_voltage=0., surrogate=False):
    assert x.ndim==3 and scale.shape==(1,x.shape[1],1)
    assert torch.isfinite(x).all() and torch.isfinite(scale).all() and (scale>0).all()
    assert initial_voltage in (0.,.5)
    if mode=='identity':
        assert initial_voltage==0.
        return x
    current=x.clamp(min=0)
    if mode=='clip':
        assert initial_voltage==0.
        return torch.minimum(current,scale)
    assert mode=='count32'
    if surrogate:return _InitialVoltageCount.apply(current,scale,32,initial_voltage)
    return torch.floor(current/(scale/32)+initial_voltage).clamp(0,32)*(scale/32)


class CoSDHPillar:
    def __init__(self,model,thresholds,mode,initial_voltage=0.,surrogate=False,encoder_paths=2,audit_gradients=False):
        assert encoder_paths in (1,2)
        self.encoder_paths=encoder_paths
        self.audit_gradients=audit_gradients
        t=json.loads(Path(thresholds).read_text())
        assert t['interface']=='pillar_vfe.pfn_layers.0.norm' and t['before_point_max'] and t['shared_agent_thresholds']
        assert len(model.pillar_vfe.pfn_layers)==1
        pfn=model.pillar_vfe.pfn_layers[0]
        assert pfn.last_vfe and pfn.use_norm and type(pfn.norm).__name__=='BatchNorm1d'
        self.scale=torch.tensor(t['range_p999'],device=next(model.parameters()).device).reshape(1,-1,1)
        assert self.scale.shape[1]==pfn.norm.num_features==64
        self.mode=mode;self.initial_voltage=initial_voltage;self.surrogate=surrogate
        self.handles=[model.pillar_vfe.register_forward_pre_hook(self.input_hook),pfn.norm.register_forward_hook(self.norm_hook)]
        self.begin()

    def input_hook(self,module,inputs):
        data=inputs[0];assert data['record_len'].tolist()==[2]
        self.agent_ids=data['voxel_coords'][:,0].long()
        assert sorted(self.agent_ids.unique().tolist())==[0,1]
        self.input_calls+=1

    def norm_hook(self,module,inputs,output):
        assert self.input_calls==1 and output.shape[0]==self.agent_ids.shape[0]
        value=activate_pillar(output,self.scale,self.mode,self.initial_voltage,self.surrogate)
        delta=(value.clamp(min=0)-output.clamp(min=0)).detach().abs()
        self.calls+=1
        self.agent_pillars=[int((self.agent_ids==i).sum()) for i in (0,1)]
        self.agent_delta=[float(delta[self.agent_ids==i].mean()) for i in (0,1)]
        if self.audit_gradients:
            assert value.requires_grad
            agent_ids=self.agent_ids.detach().clone()
            def record(grad):
                assert torch.isfinite(grad).all()
                self.gradients=[float(grad[agent_ids==i].abs().mean()) for i in (0,1)]
                return grad
            value.register_hook(record)
        return value

    def begin(self):
        self.input_calls=0;self.calls=0;self.agent_ids=None;self.agent_pillars=[];self.agent_delta=[]
        self.gradients=[]

    def gradient_summary(self):
        assert self.audit_gradients and len(self.gradients)==2 and all(x>0 for x in self.gradients)
        return dict(both_agents_nonzero=True,mean_abs=self.gradients)

    def finish(self):
        assert self.input_calls==self.calls==1 and all(n>0 for n in self.agent_pillars)
        return dict(mode=self.mode,initial_voltage_threshold_fraction=self.initial_voltage,
            calls=self.calls,agents=2,pillars_per_agent=self.agent_pillars,
            activation_abs_change_per_agent=self.agent_delta,before_point_max=True,
            BEV_paths=self.encoder_paths,one_shared_pillar_output_for_both_BEV_paths=self.encoder_paths==2)

    def close(self):
        for h in self.handles:h.remove()


def self_test():
    x=torch.arange(-64,257,dtype=torch.float32).reshape(1,1,-1).repeat(3,2,1)/128
    scale=torch.ones(1,2,1)
    assert activate_pillar(x,scale,'identity') is x
    assert torch.equal(activate_pillar(x,scale,'clip'),x.clamp(0,1))
    for v0 in (0.,.5):
        theta=scale/32;mem=torch.ones_like(x)*theta*v0;counts=torch.zeros_like(x)
        for _ in range(32):
            mem+=x.clamp(min=0)/32;spike=(mem>=theta).float();counts+=spike;mem-=spike*theta
        y=activate_pillar(x,scale,'count32',v0)
        assert torch.equal(y,counts*theta)
        # Monotone shared per-channel IF commutes with point max; it is still
        # applied before max, including padded positions used by the source.
        assert torch.equal(y.max(2,keepdim=True)[0],activate_pillar(x.max(2,keepdim=True)[0],scale,'count32',v0))
        z=x.clone().requires_grad_(True);q=activate_pillar(z,scale,'count32',v0,True)
        assert torch.equal(q.detach(),y);q.sum().backward()
        assert torch.equal(z.grad,((x>=0)&(x<=scale)).float())
    print('PFN signed identity, pre-max IF, explicit32 ticks and surrogate passed')


if __name__=='__main__':self_test()
