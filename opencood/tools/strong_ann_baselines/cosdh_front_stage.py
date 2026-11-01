"""Plain BEV stage analytic IF activations, both agents and both paths.

This is a partial encoder conversion: PillarVFE and subsequent stages stay ANN.
"""
import json
from pathlib import Path
import torch
from cosdh_readout import _CountReadout


class _InitialVoltageCount(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, steps, initial_voltage):
        ctx.save_for_backward(x,scale)
        theta=scale/steps
        return torch.floor(x/theta+initial_voltage).clamp(0,steps)*theta

    @staticmethod
    def backward(ctx, grad):
        x,scale=ctx.saved_tensors
        return grad*((x>=0)&(x<=scale)).to(grad.dtype),None,None,None


def activate(x, scale, mode, surrogate=False, initial_voltage=0.):
    if initial_voltage not in (0., .5):
        raise ValueError('Registered initial voltage fractions are zero and half threshold')
    if initial_voltage and mode not in ('count16','count32'):
        raise ValueError('Half-threshold initialization requires count activations')
    if x.ndim != 4 or x.shape[0] != 2 or scale.shape != (1,x.shape[1],1,1):
        raise ValueError('Two participants and shared channel scales required')
    if not torch.isfinite(x).all() or (x<0).any() or not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Unsigned finite activation and positive scale required')
    if mode=='identity': return x
    if mode=='clip': return torch.minimum(x,scale)
    if mode not in ('count16','count32'): raise ValueError('Unsupported activation')
    steps=int(mode[5:]);theta=scale/steps
    if initial_voltage:
        if surrogate: return _InitialVoltageCount.apply(x,scale,steps,initial_voltage)
        return torch.floor(x/theta+initial_voltage).clamp(0,steps)*theta
    if surrogate: return _CountReadout.apply(x,scale,steps)
    return torch.floor(x/theta).clamp(0,steps)*theta


class CoSDHFrontStage:
    def __init__(self, model, thresholds, mode, surrogate=False, audit_gradients=False, initial_voltage=0., encoder_paths=2):
        assert encoder_paths in (1,2)
        self.encoder_paths=encoder_paths
        t=json.loads(Path(thresholds).read_text())
        assert type(model.backbone).__name__=='BaseBEVBackbone'
        assert t['interface'] in ('backbone.blocks.0','backbone.blocks.1','backbone.blocks.2')
        self.stage=int(t['interface'].split('.')[-1])
        expected=[t['interface']+'.'+n for n,m in model.backbone.blocks[self.stage].named_children() if type(m).__name__=='ReLU']
        assert t['module_names']==expected and t['shared_agent_thresholds']
        self.sites=len(expected)
        assert self.sites==(4,6,9)[self.stage] and t['expected_calls_per_forward']==2*self.sites
        modules=dict(model.named_modules()); self.mode=mode;self.calls=[0]*self.sites;self.first={};self.delta={};self.handles=[]
        self.surrogate=surrogate;self.audit_gradients=audit_gradients;self.gradients={}
        self.initial_voltage=initial_voltage
        for phase,name in enumerate(t['module_names']):
            assert type(modules[name]).__name__=='ReLU'
            scale=torch.tensor(t['phases'][str(phase)]['range_p999'],device=next(model.parameters()).device).reshape(1,-1,1,1)
            self.handles.append(modules[name].register_forward_hook(self.hook(phase,scale)))

    def hook(self, phase, scale):
        def apply(module,inputs,output):
            value=activate(output,scale,self.mode,self.surrogate,self.initial_voltage);self.calls[phase]+=1
            if self.encoder_paths==1 or self.calls[phase]%2:
                self.first[phase]=value.detach().clone()
                self.delta[phase]=float((value-output).detach().abs().mean())
            else:
                assert torch.equal(self.first[phase],value), 'Confidence and communication paths diverged'
            if self.audit_gradients and (self.encoder_paths==1 or self.calls[phase]%2==0):
                assert value.requires_grad
                def record(grad):
                    assert torch.isfinite(grad).all()
                    self.gradients[phase]=[float(v.abs().mean()) for v in grad]
                    return grad
                value.register_hook(record)
            return value
        return apply

    def begin(self): self.calls=[0]*self.sites;self.first={};self.delta={};self.gradients={}

    def gradient_summary(self):
        assert self.audit_gradients and set(self.gradients)==set(range(self.sites))
        assert all(len(v)==2 and all(g>0 for g in v) for v in self.gradients.values())
        return dict(both_agents_nonzero=True, all_four_sites=self.stage==0,
            stage=self.stage,all_sites=True,sites=self.sites,mean_abs=self.gradients)

    def finish(self):
        assert self.calls==[self.encoder_paths]*self.sites and len(self.delta)==self.sites
        return dict(mode=self.mode,calls=self.calls[:],agents=2,encoder_paths=self.encoder_paths,
            paths_bitwise_equal=True if self.encoder_paths==2 else None,
            initial_voltage_threshold_fraction=self.initial_voltage,
            activation_abs_change=[self.delta[i] for i in range(self.sites)],stage_index=self.stage,
            first_stage_only=self.stage==0,whole_encoder_snn=False)

    def close(self):
        for h in self.handles:h.remove()


def self_test():
    x=torch.tensor([[[[0.,.25,.99,2.]]],[[[.1,.5,1.,0.]]]])
    scale=torch.ones(1,1,1,1)
    assert activate(x,scale,'identity') is x
    for steps in (16,32):assert torch.equal(activate(x,scale,'count'+str(steps)),torch.floor(x*steps).clamp(0,steps)/steps)
    assert torch.equal(activate(x,scale,'clip'),x.clamp(max=1)) and x.max()==2
    for mode in ('count16','count32'):
        z=x.clone().requires_grad_(True);value=activate(z,scale,mode,surrogate=True)
        assert torch.equal(value.detach(),activate(x,scale,mode))
        value.sum().backward();assert torch.equal(z.grad,(x<=scale).float())
    # Binary-exact inputs verify the analytic count against T explicit IF steps,
    # with one spike per tick, subtractive reset and V0 = theta/2.
    probe=torch.arange(0,257,dtype=torch.float32).reshape(1,1,1,-1).repeat(2,1,1,1)/128
    for steps in (16,32):
        theta=scale/steps;membrane=torch.ones_like(probe)*theta*.5;count=torch.zeros_like(probe)
        for _ in range(steps):
            membrane=membrane+probe/steps
            spike=(membrane>=theta).float();count+=spike;membrane-=spike*theta
        value=activate(probe,scale,'count'+str(steps),initial_voltage=.5)
        assert torch.equal(value,count*theta) and torch.count_nonzero(value[...,0])==0
        z=probe.clone().requires_grad_(True)
        train_value=activate(z,scale,'count'+str(steps),surrogate=True,initial_voltage=.5)
        assert torch.equal(train_value.detach(),value)
        train_value.sum().backward();assert torch.equal(z.grad,(probe<=scale).float())
    print('Both-agent IF grid, saturation, clipping and input preservation passed')


if __name__=='__main__':self_test()
