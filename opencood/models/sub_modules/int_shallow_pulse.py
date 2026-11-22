"""Shallow observation membranes feeding a fixed-grid incremental 3x3 convolution."""
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionKernel, _KERNEL


def address_blocks(innovation):
    h,w = innovation.shape[2:]
    score = F.avg_pool2d(innovation.abs().sum(1,keepdim=True),4,4,ceil_mode=True)
    ordered = np.argsort(-score.flatten().cpu().numpy(),kind='stable')
    chosen = torch.zeros_like(score,dtype=torch.bool)
    chosen.flatten()[torch.as_tensor(ordered[:(len(ordered)+1)//2],device=score.device)] = True
    digest = hashlib.sha256(chosen.cpu().numpy().tobytes()).hexdigest()
    return chosen.repeat_interleave(4,2).repeat_interleave(4,3)[:,:,:h,:w], digest


def receptive_counts(levels, single, multiple):
    h,w = levels.shape[2:]
    counts = torch.cat([(levels!=0).sum(1,keepdim=True),single.sum(1,keepdim=True),
                        multiple.sum(1,keepdim=True)],1)
    padded = F.pad(counts,(1,1,1,1))
    totals = sum(padded[:,:,y:y+h,x:x+w] for y in range(3) for x in range(3))
    return totals[0].reshape(3,-1)


class ShallowPulseConv(nn.Module):
    quantum = .25

    def __init__(self, original, floating=False, signed=True):
        super().__init__()
        assert (original.in_channels,original.out_channels,original.kernel_size,original.stride,
                original.padding,original.dilation,original.groups)==(128,128,(3,3),(1,1),(1,1),(1,1),1)
        self.floating = floating
        self.signed = signed
        if floating:
            self.quantum = 1.
        self.register_buffer('scaled', (original.weight.detach()*self.quantum).permute(2,3,1,0).contiguous())
        self.bias = original.bias
        self.static = NonzeroFusionKernel()
        self.add = NonzeroFusionKernel('#define ACCUMULATE\n#define SIGN_ONLY\n'+_KERNEL) if signed else None
        self.fma = NonzeroFusionKernel('#define ACCUMULATE\n'+_KERNEL)
        self.states = {}

    def begin(self, name, reason, meta, input_shape, bounds, verify):
        self.name,self.meta,self.shape,self.bounds,self.verify = name,meta,input_shape,bounds,verify
        if reason is not None:
            self.states.pop(name,None)

    def forward(self, x):
        assert not torch.is_grad_enabled() and x.dtype==torch.float32 and x.shape[:2]==(1,128)
        target = x if self.floating else (x/self.quantum).round()
        assert float(target.min())>=0 and float(target.max())<2**20
        h,w = x.shape[2:]
        old = self.states.get(self.name)
        sizes = x.numel()
        rec = dict(values=sizes, offered=0,sent=0,units=0,transport_changes=0,
                   actual_changes=0,actual_unit_changes=0, support=None,
                   control_float=0 if self.floating else sizes, routing_int=0, rounding=0 if self.floating else sizes)
        if old is None or self.name in ('instant','float_current'):
            levels = target
        else:
            seen,received,_,pose = old
            if self.name.startswith('motion'):
                area = block_lidar_range(self.bounds,self.shape,(h,w),2)
                moved,_ = warp_bev(torch.cat((seen,received),1),pose,self.meta['pose'],area)
                seen,prior = moved.chunk(2,1)
                # warp_bev applies an explicit validity multiplication on both channel groups.
                rec['control_float'] += 2*sizes
            else:
                prior = received
            innovation = target-seen
            residual = target-prior
            if self.verify:
                assert torch.equal(innovation+(seen-prior),residual)
            allow,rec['support'] = address_blocks(innovation)
            amount = residual.sign() if self.name.endswith('unit') else residual
            release = amount.masked_fill(~allow,0)
            levels = prior+release
            if self.verify:
                assert torch.equal(residual-release,target-levels)
                assert bool((levels>=0).all()) and bool((release.masked_select(~allow)==0).all())
            rec.update(offered=int(allow.sum())*128,sent=int(release.count_nonzero()),
                       units=int(release.abs().sum()),transport_changes=int((prior!=received).sum()))
            # Two differences, one reconstruction addition, channel sums and tile pooling.
            rec['control_float'] += 3*sizes+(sizes-h*w)+h*w
        all_sites = torch.arange(h*w,device=x.device)
        unit_visits = multiple_visits = 0
        if old is None:
            buffer = x.new_empty(h*w,128)
            rebuild = self.static.run(levels,self.scaled,all_sites,buffer)
            reused = delta_sites = 0
        else:
            change = levels-old[1]
            one = change.abs()==1
            many = (change!=0) & ~one
            nz,u,m = receptive_counts(levels,one,many)
            dirty = (u+m)>0
            incremental_price = (u+2*m) if self.signed else 2*(u+m)
            increment = dirty & (incremental_price<2*nz)
            fresh = dirty & ~increment
            # Clean zero outputs are reusable as well.
            buffer = old[2].clone()
            rebuild = self.static.run(levels,self.scaled,all_sites[fresh],buffer)
            if self.signed:
                unit_visits = self.add.run(change.masked_fill(~one,0),self.scaled,all_sites[increment & (u>0)],buffer)
                multiple_visits = self.fma.run(change.masked_fill(~many,0),self.scaled,all_sites[increment & (m>0)],buffer)
                assert (unit_visits,multiple_visits)==(int(u[increment].sum()),int(m[increment].sum()))
            else:
                multiple_visits = self.fma.run(change,self.scaled,all_sites[increment],buffer)
                assert multiple_visits==int((u[increment]+m[increment]).sum())
            assert rebuild==int(nz[fresh].sum())
            rec.update(actual_changes=int((one|many).sum()), actual_unit_changes=int(one.sum()))
            rec['control_float'] += sizes
            # Channel reductions, nine-tap reductions, and the per-output cost comparison operands.
            rec['routing_int'] = 3*(sizes-h*w)+(3*9+4)*h*w
            reused,delta_sites = int((~dirty).sum()),int(increment.sum())
        rec.update(rebuild_mac=rebuild*128,delta_mac=multiple_visits*128,delta_ac=unit_visits*128,
                   reused=reused,delta_sites=delta_sites,sites=h*w,
                   error_l1=float((x-levels*self.quantum).abs().mean()))
        if self.verify:
            reference = torch.empty_like(buffer)
            self.static.run(levels,self.scaled,all_sites,reference)
            err = (buffer-reference).abs()
            rec['max_error'] = float(err.max())
            assert bool((err<=.0005+.0001*reference.abs()).all())
        stored_observation = None if self.name in ('instant','float_current') else target
        self.states[self.name] = (stored_observation,levels,buffer,self.meta['pose'])
        unique = {t.data_ptr():t for t in (stored_observation,levels,buffer) if t is not None}
        rec['state_bytes'] = sum(t.numel()*t.element_size() for t in unique.values())
        self.record = rec
        out = buffer.T.reshape(1,128,h,w).contiguous()
        return out if self.bias is None else out+self.bias.view(1,-1,1,1)
