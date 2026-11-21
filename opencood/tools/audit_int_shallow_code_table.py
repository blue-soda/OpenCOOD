"""Audit direct four-level event corrections across shallow DW, ReLU and pointwise mixing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_compact_middle_update import neighbors,points,supports
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.check_int_deep_spike_stream import prepare_stream,read
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_fraction_history import HistoryExperiment,numeric_mask


def route(bounds,spatial_shape,shape,stride,old_pose,pose,device):
    area=block_lidar_range(bounds,spatial_shape,shape,stride)
    ids=torch.arange(1,shape[0]*shape[1]+1,dtype=torch.float32,device=device).view(1,1,*shape)
    return warp_bev(ids,old_pose,pose,area)[0].long().flatten()-1


def linear_parts(patch,dw,bias):
    return ((patch*0.25*dw).sum(-1)+bias).relu()


def corrected_projection(now,old,weight,bias,eligible,indices,sign,table):
    pos,channel=eligible.nonzero(as_tuple=True)
    residual=(now-old).masked_fill(eligible,0)
    result=F.linear(old,weight,bias)+F.linear(residual,weight)
    if len(pos):
        vectors=table[indices[pos,channel],channel]
        result.index_add_(0,pos,vectors*sign[pos,channel,None])
    return result


class TableAudit:
    def __init__(self,exp,book):
        dw=exp.source.core.coarse_light[0]
        mix=exp.source.core.coarse_light[2]
        self.dw=dw.weight.detach().reshape(dw.out_channels,9)
        self.db=dw.bias if dw.bias is not None else self.dw.new_zeros(dw.out_channels)
        self.weight=mix.weight[:,:,0,0].detach()
        self.bias=mix.bias
        assert self.weight.shape==(256,128) and dw.groups==128 and dw.stride==(2,2)
        self.levels=self.dw.new_tensor(book['levels'])
        # Combine amplitude rank and tap into one index, then select input channel.
        base=self.levels.double()[:,None,None,None]*.25*self.dw.double().T[None,:,:,None]*self.weight.double().T[None,None]
        self.table64=base.reshape(36,128,256).contiguous()
        self.table=self.table64.float()

    def measure(self,exp,state,prev,meta,spatial_shape,mask,stats):
        levels=state['fine_levels']
        active=supports(~mask)[2][0,0]
        state['table_valid']=active
        fine_shape=levels.shape[-2:]
        middle_shape=active.shape
        args=prev['pose'],meta['pose'],levels.device
        fine=route(exp.source.bounds,spatial_shape,fine_shape,2,*args)
        middle=route(exp.source.bounds,spatial_shape,middle_shape,4,*args)
        old_values=points(prev['fine_levels'])
        changes=points(levels)-old_values[fine.clamp_min(0)].masked_fill((fine<0)[:,None],0)
        nonzero=changes[changes!=0].abs()
        assert len(nonzero)==stats['addresses'] and int(nonzero.long().sum())==stats['units']
        assert bool(torch.isin(nonzero,self.levels.int()).all())
        nout=self.weight.shape[0]
        totals=Counter()
        err32=err64=0.
        for ids in active.flatten().nonzero().flatten().split(192):
            addr,valid=neighbors(ids,middle_shape[1],fine_shape,2)
            source=middle[ids]
            old_addr,old_valid=neighbors(source.clamp_min(0),middle_shape[1],fine_shape,2)
            # All 9 taps, including padding, must correspond for event updates.
            good=(source>=0)&(fine[addr].masked_fill(~valid,-1)==old_addr.masked_fill(~old_valid,-1)).all(1)
            if 'table_valid' in prev:good &= prev['table_valid'].flatten()[source.clamp_min(0)]
            else:good.zero_()
            patch=points(levels)[addr].masked_fill(~valid[:,:,None],0).transpose(1,2)
            now=linear_parts(patch,self.dw,self.db)
            static=int(now.count_nonzero())*nout
            totals.update(sites=len(ids),good=int(good.sum()),static=static,matched=static)
            if not good.any():continue
            patch=patch[good]
            now=now[good]
            old_patch=old_values[old_addr[good]].masked_fill(~old_valid[good,:,None],0).transpose(1,2)
            old=linear_parts(old_patch,self.dw,self.db)
            now64=linear_parts(patch.double(),self.dw.double(),self.db.double())
            old64=linear_parts(old_patch.double(),self.dw.double(),self.db.double())
            delta=patch-old_patch
            changed=(now!=old)
            # Give the reference a per-output choice between a fresh sparse sum and a cached difference.
            fresh=now.count_nonzero(dim=1)
            edits=changed.sum(dim=1)
            cached=edits<=fresh
            totals['matched']-=int((fresh-torch.minimum(edits,fresh)).sum())*nout
            eligible=((delta!=0).sum(-1)==1)&(now>0)&(old>0)&(now64>0)&(old64>0)&changed&cached[:,None]
            tap=delta.abs().argmax(-1)
            code=delta.gather(-1,tap[:,:,None]).squeeze(-1)
            rank=torch.bucketize(code.abs().float(),self.levels).clamp_max(3)
            assert bool((self.levels[rank[eligible]]==code[eligible].abs()).all())
            index=rank*9+tap
            sign=code.sign()
            totals['replaced']+=int(eligible.sum())*nout
            totals['lookup_vectors']+=int(eligible.sum())
            totals['cached_sites']+=int(cached.sum())
            totals['matched_across_cached']+=int(edits[cached].sum())*nout
            # Only the cached path is reconstructed. The other sites retain the static result.
            if cached.any():
                def compare(n,o,w,b,table):
                    out=corrected_projection(n[cached],o[cached],w,b,eligible[cached],index[cached],sign[cached],table)
                    return float((out-F.linear(n[cached],w,b)).abs().max())
                err32=max(err32,compare(now,old,self.weight,self.bias,self.table))
                err64=max(err64,compare(now64,old64,self.weight.double(),None if self.bias is None else self.bias.double(),self.table64))
        assert err64<1e-8 and 0<=totals['replaced']<=totals['matched']<=totals['static']
        return dict(totals,fp32_abs=err32,fp64_abs=err64)


@torch.no_grad()
def run(a):
    source,cfg,_,_,_,_,q=prepare_stream(a.inputs,a.scans)
    book=read(a.book)
    assert book['cap']==428149 and book['levels']==[3,6,13,28]
    plan=read(a.training/'plan.json')
    trained=read(a.training/'report.json')
    assert trained['plan']==digest(a.training/'plan.json') and trained['frames']==digest(a.training/'frames.jsonl')
    assert plan['book']==digest(a.book)
    rows=[dict(r,supervised=False,label=None) for r in plan['rows']]
    assert all(digest(Path(r['pcd']))==plan['members'][r['pcd']] for r in rows)
    traces=[json.loads(x) for x in (a.training/'frames.jsonl').read_bytes().splitlines()]
    exp=HistoryExperiment(source,q,book)
    audit=TableAudit(exp,book)
    _,edge=exp.models['carry']
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for divisor in (1,2,4):
            edge.cap=book['cap']//divisor
            _,stream=loader(rows,cfg,0,303)
            for i,(sample,trace) in enumerate(zip(stream,traces)):
                meta=sample['meta']
                assert not meta['supervised'] and 'label_dict' not in sample['ego']
                if i==0 or rows[i-1]['segment']!=meta['segment']:state=None
                spatial=spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
                mask=numeric_mask(trace)
                prev=state
                _,state,stats=exp.step('carry',spatial,meta,prev,trace,mask)
                row=dict(divisor=divisor,frame=meta['frame'],age=state['since_refresh'],core=stats['core'])
                if mask is not None:
                    row.update(audit.measure(exp,state,prev,meta,spatial.shape[-2:],mask,stats))
                log.write(json.dumps(row)+'\n')
            assert i+1==211
    write_json(a.output/'report.json',dict(status='COMPLETED',scans_per_arm=211,optimizer_steps=0,labels=0,
               table_bytes=audit.table.numel()*audit.table.element_size(),train=trained['plan'],trace=trained['frames'],book=digest(a.book),
               code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','scans','book','training','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)

