"""Isolated ANN state/fast-path and exact delta execution research primitives.

These are engineering candidates, not trained detector replacements. Pose alignment
is the caller's responsibility. Delta execution currently supports one aligned grid.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class SingleGateStateUpdate(nn.Module):
    """Replace a StateUpdate: h = old + z * (candidate - old), not full GRU."""
    def __init__(self, candidate, channels, gate_width=64):
        super().__init__()
        self.candidate=copy.deepcopy(candidate)
        self.gate=nn.Sequential(nn.Conv2d(2*channels+2,gate_width,1),nn.ReLU(),nn.Conv2d(gate_width,channels,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)

    def forward(self,current,old,valid,dt=None):
        dt=getattr(self,'delta_t',.1) if dt is None else dt
        if not 0<=dt<=.25:raise ValueError('Reset state before processing a gap')
        old=torch.where(valid,old,torch.zeros_like(old))
        candidate=self.candidate(current,old,valid)
        clock=torch.full_like(valid,dtype=current.dtype,fill_value=dt)
        z=torch.sigmoid(self.gate(torch.cat((current,old,valid.to(current.dtype),clock),1)))
        return torch.where(valid,old+z*(candidate-old),candidate)


class FastReadoutUpdate(nn.Module):
    """Ordinary-frame learned correction; full refresh readout lives outside."""
    def __init__(self,inputs=384,channels=256,width=64):
        super().__init__()
        self.current=nn.Conv2d(inputs,channels,1)
        self.correction=nn.Sequential(nn.Conv2d(2*channels+1,width,1),nn.ReLU(),
            nn.Conv2d(width,width,3,padding=1,groups=width),nn.ReLU(),nn.Conv2d(width,channels,1))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)

    def forward(self,current,old,valid):
        observed=F.relu(self.current(current))
        old=torch.where(valid,old,torch.zeros_like(old))
        correction=self.correction(torch.cat((observed,old,valid.to(current.dtype)),1))
        return F.relu(torch.where(valid,old+correction,observed))


def content_request(current,aligned,valid,fraction):
    """Fixed area-budget proposal before expensive execution, plus all holes.

    Scores only select valid cells; unobserved regions are mandatory. No AP-driven
    threshold and no claim that this descriptor preserves new/moving objects.
    """
    if not 0<=fraction<=1 or current.shape[0]!=1:raise ValueError('Single stream and area fraction required')
    score=(current-aligned).abs().mean(1,keepdim=True)/(current.abs().mean(1,keepdim=True)+aligned.abs().mean(1,keepdim=True)+1e-6)
    flat=valid.flatten();indices=torch.nonzero(flat,as_tuple=False).flatten()
    k=int(indices.numel()*fraction)
    request=(~valid).clone()
    if k:
        chosen=indices[torch.argsort(score.flatten()[indices],descending=True)[:k]]
        request.flatten()[chosen]=True
    return request,score


def sparse_delta_conv(conv,delta):
    """Execute 3x3 stride-one delta Conv using active spatial vectors only.

    A position sends its full channel vector: no per-channel sparsity discount.
    Bias is omitted because it cancels in the difference. Dense temporary output,
    gathers/scatters, state reads and activation updates remain real overheads.
    """
    if delta.shape[0]!=1 or conv.kernel_size!=(3,3) or conv.stride!=(1,1) or conv.padding!=(1,1) or conv.groups!=1 or conv.dilation!=(1,1):
        raise ValueError('Only single-stream 3x3/stride1/pad1/group1 supported')
    _,cin,h,w=delta.shape
    coords=torch.nonzero(torch.any(delta!=0,dim=1)[0],as_tuple=False)
    output=delta.new_zeros(conv.out_channels,h*w);deliveries=0
    for ky in range(3):
        for kx in range(3):
            y=coords[:,0]+1-ky;x=coords[:,1]+1-kx
            keep=(y>=0)&(y<h)&(x>=0)&(x<w)
            chosen=coords[keep]
            if chosen.shape[0]==0:continue
            vectors=delta[0,:,chosen[:,0],chosen[:,1]].t()
            contribution=F.linear(vectors,conv.weight[:,:,ky,kx])
            output.index_add_(1,y[keep]*w+x[keep],contribution.t())
            deliveries+=chosen.shape[0]
    return output.view(1,conv.out_channels,h,w),dict(active_positions=coords.shape[0],
        vector_deliveries=deliveries,executed_macs=deliveries*cin*conv.out_channels,
        dense_macs=h*w*cin*conv.out_channels*9)


class DeltaReadoutProbe(nn.Module):
    """Two exact Conv/ReLU layers, retaining pre- and post-activation states."""
    def __init__(self,inputs=384,width=256,outputs=256):
        super().__init__()
        self.first=nn.Conv2d(inputs,width,3,padding=1)
        self.second=nn.Conv2d(width,outputs,3,padding=1)

    def dense(self,x):return F.relu(self.second(F.relu(self.first(x))))

    def step(self,x,state=None):
        if torch.is_grad_enabled():raise ValueError('Inference engineering only')
        if state is None:
            z1=self.first(x);a1=F.relu(z1);z2=self.second(a1);a2=F.relu(z2)
            stats=dict(refresh=True,executed_macs=x.shape[-2]*x.shape[-1]*9*(self.first.in_channels*self.first.out_channels+self.second.in_channels*self.second.out_channels))
        else:
            dz1,s1=sparse_delta_conv(self.first,x-state['input'])
            z1=state['z1']+dz1;a1=F.relu(z1)
            dz2,s2=sparse_delta_conv(self.second,a1-state['a1'])
            z2=state['z2']+dz2;a2=F.relu(z2)
            stats=dict(refresh=False,layers=[s1,s2],executed_macs=s1['executed_macs']+s2['executed_macs'])
        new=dict(input=x.clone(),z1=z1,a1=a1,z2=z2,a2=a2)
        stats['state_bytes']=sum(v.numel()*v.element_size() for v in new.values())
        stats['dense_pointwise_elements']=z1.numel()+z2.numel()
        return a2,new,stats
