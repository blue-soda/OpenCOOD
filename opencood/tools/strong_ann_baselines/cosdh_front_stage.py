"""First plain BEV stage analytic IF activations, both agents and both paths.

This is a partial encoder conversion: PillarVFE and subsequent stages stay ANN.
"""
import json
from pathlib import Path
import torch


def activate(x, scale, mode):
    if x.ndim != 4 or x.shape[0] != 2 or scale.shape != (1,x.shape[1],1,1):
        raise ValueError('Two participants and shared channel scales required')
    if not torch.isfinite(x).all() or (x<0).any() or not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Unsigned finite activation and positive scale required')
    if mode=='identity': return x
    if mode=='clip': return torch.minimum(x,scale)
    if mode not in ('count16','count32'): raise ValueError('Unsupported activation')
    steps=int(mode[5:]);theta=scale/steps
    return torch.floor(x/theta).clamp(0,steps)*theta


class CoSDHFrontStage:
    def __init__(self, model, thresholds, mode):
        t=json.loads(Path(thresholds).read_text())
        assert t['interface']=='backbone.blocks.0' and t['shared_agent_thresholds'] and t['expected_calls_per_forward']==8
        assert type(model.backbone).__name__=='BaseBEVBackbone'
        modules=dict(model.named_modules()); self.mode=mode;self.calls=[0]*4;self.first={};self.delta={};self.handles=[]
        for phase,name in enumerate(t['module_names']):
            assert type(modules[name]).__name__=='ReLU'
            scale=torch.tensor(t['phases'][str(phase)]['range_p999'],device=next(model.parameters()).device).reshape(1,-1,1,1)
            self.handles.append(modules[name].register_forward_hook(self.hook(phase,scale)))

    def hook(self, phase, scale):
        def apply(module,inputs,output):
            value=activate(output,scale,self.mode);self.calls[phase]+=1
            if self.calls[phase]%2:
                self.first[phase]=value.detach().clone()
                self.delta[phase]=float((value-output).detach().abs().mean())
            else:
                assert torch.equal(self.first[phase],value), 'Confidence and communication paths diverged'
            return value
        return apply

    def begin(self): self.calls=[0]*4;self.first={};self.delta={}

    def finish(self):
        assert self.calls==[2]*4 and len(self.delta)==4
        return dict(mode=self.mode,calls=self.calls[:],agents=2,paths_bitwise_equal=True,
            activation_abs_change=[self.delta[i] for i in range(4)],first_stage_only=True,whole_encoder_snn=False)

    def close(self):
        for h in self.handles:h.remove()


def self_test():
    x=torch.tensor([[[[0.,.25,.99,2.]]],[[[.1,.5,1.,0.]]]])
    scale=torch.ones(1,1,1,1)
    assert activate(x,scale,'identity') is x
    for steps in (16,32):assert torch.equal(activate(x,scale,'count'+str(steps)),torch.floor(x*steps).clamp(0,steps)/steps)
    assert torch.equal(activate(x,scale,'clip'),x.clamp(max=1)) and x.max()==2
    print('Both-agent IF grid, saturation, clipping and input preservation passed')


if __name__=='__main__':self_test()
