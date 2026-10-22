"""Experimental projected-GRU gate caches with explicit recurrent ownership.

The hidden state still updates everywhere. Only selected pre-BN spatial gate
responses are recomputed; unselected responses and their input references warp
forward together. Current projections, nonlinearities and readout remain dense.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_cache_repair import tile_mask, expand_tiles
from opencood.models.sub_modules.int_selected_conv import selected_conv3x3


GATE_NAMES=('reset_gate_conv','update_gate_conv','candidate_conv')


class SelectedGate(nn.Module):
    def __init__(self, layer, bounds, threshold, execution):
        super().__init__()
        self.layer=layer;self.bounds=bounds;self.threshold=threshold;self.execution=execution
        self.context=None;self.result=None

    def forward(self, packed):
        if self.context is None:
            raise RuntimeError('Gate execution must belong to an explicit memory step')
        current=self.layer[0](packed);height,width=current.shape[-2:]
        old,old_pose,pose,refresh=self.context
        if refresh:
            response=self.layer[1](current);reference=current
            mask=torch.ones_like(current[:,:1],dtype=torch.bool);extra_macs=0
        else:
            aligned,valid=warp_bev(torch.cat((old['reference'],old['response']),1),old_pose,pose,self.bounds)
            reference,previous=aligned.split((current.shape[1],self.layer[1][1].out_channels),dim=1)
            score=((current-reference).square().mean(1,keepdim=True)/torch.maximum(
                current.square().mean(1,keepdim=True),reference.square().mean(1,keepdim=True)).clamp_min(1e-6)).sqrt()
            request=F.max_pool2d(((score>self.threshold)|~valid).float(),3,1,1)>0
            mask=expand_tiles(tile_mask(request,4),4,(height,width))
            if self.execution=='selected':
                response,extra_macs=selected_conv3x3(self.layer[1][1],current,previous,mask)
            else:
                response=torch.where(mask,self.layer[1](current),previous);extra_macs=0
            # Keep the input reference belonging to each retained response. Using
            # last-frame current inputs here would hide slowly accumulating change.
            reference=torch.where(mask,current,reference)
        self.result=dict(reference=reference.detach(),response=response.detach(),
            selected=int(mask.sum()),cells=height*width,extra_macs=extra_macs)
        return self.layer[3](self.layer[2](response))


class SelectedProjectedMemory(nn.Module):
    def __init__(self, source, threshold=.5, period=3, execution='selected'):
        super().__init__()
        if source.mode!='gru' or threshold<0 or period<1 or execution not in ('selected','masked_dense'):
            raise ValueError('Invalid selected projected-memory configuration')
        self.memory=copy.deepcopy(source);self.period=period
        self.mode=source.mode;self.channels=source.channels;self.lidar_range=source.lidar_range
        for name in GATE_NAMES:
            layer=getattr(self.memory.fusion,name)
            if not isinstance(layer[0],nn.Conv2d) or layer[0].kernel_size!=(1,1):
                raise ValueError('Expected a trained projected GRU')
            setattr(self.memory.fusion,name,SelectedGate(layer,self.lidar_range,threshold,execution))
        self.eval()

    def step(self,current,state,meta,reset=False,align=True,detach_state=True,state_observer=None):
        if self.training or torch.is_grad_enabled() or not detach_state or not align:
            raise ValueError('Aligned frozen inference only')
        old=None if state is None else state.get('gate_cache')
        refresh=reset or state is None or old is None
        if not refresh:
            elapsed=(int(meta['timestamp_us'])-state['timestamp_us'])/1e6
            refresh=(state['scene']!=meta['segment'] or elapsed>self.memory.max_gap_s or old['frames']>=self.period-1)
        for name in GATE_NAMES:
            gate=getattr(self.memory.fusion,name)
            gate.context=(None if refresh else old['gates'][name],
                None if refresh else state['pose'],meta['pose'],refresh)
        try:
            output,new_state,info=self.memory.step(current,state,meta,reset,align,detach_state,state_observer)
            gates={};extra_macs=selected=cells=0
            for name in GATE_NAMES:
                value=getattr(self.memory.fusion,name).result
                gates[name]={k:value[k] for k in ('reference','response')}
                extra_macs+=value['extra_macs'];selected+=value['selected'];cells+=value['cells']
            new_state['gate_cache']=dict(frames=0 if refresh else old['frames']+1,gates=gates)
            info.update(gate_refresh=refresh,gate_selected_fraction=selected/float(cells),
                gate_extra_linear_macs=extra_macs)
            info['state_bytes']+=sum(v.numel()*v.element_size() for g in gates.values() for v in g.values())
            return output,new_state,info
        finally:
            for name in GATE_NAMES:
                gate=getattr(self.memory.fusion,name);gate.context=gate.result=None
