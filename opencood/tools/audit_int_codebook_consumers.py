"""Measure the frozen detector and price linear readout reuse before changing it."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate, conv_costs, supports
from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.models.sub_modules.int_fine_demand import QuotaFineEdge
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.calibrate_int_middle_budget import training_rows
from opencood.tools.check_int_deep_spike_stream import build_source, prepare_stream, read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_deep_block_membrane import load_quantum
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.train_int_deep_block_adapt import trace_masks,local_request


def geometry(route,h,w,k):
    src=route.view(h,w)
    sy=src.clamp_min(0)//w
    sx=src.clamp_min(0)%w
    good=src>=0
    r=k//2
    padded=F.pad(src,(r,r,r,r),value=-1)
    for y in range(-r,r+1):
        for x in range(-r,r+1):
            yy,xx=sy+y,sx+x
            taps=(yy*w+xx).masked_fill((yy<0)|(yy>=h)|(xx<0)|(xx>=w),-1)
            good=good & (padded[r+y:r+y+h,r+x:r+x+w]==taps)
    return good


def shallow_ledger(model,state,previous,meta,spatial_shape):
    levels=state['fine_levels']
    active=supports(~model.coarse_executor.skip)[2][0,0]
    state['shallow_support']=active
    h,w=levels.shape[-2:]
    oh,ow=active.shape
    def route(shape,stride):
        height,width=shape
        bounds=block_lidar_range(model.source.bounds,spatial_shape,shape,stride)
        codes=torch.arange(1,height*width+1,dtype=torch.float32,device=levels.device).view(1,1,height,width)
        return warp_bev(codes,previous['pose'],meta['pose'],bounds)[0][0,0].long()-1
    fine=route((h,w),2)
    coarse=route((oh,ow),4)
    oy=coarse.clamp_min(0)//ow*2
    ox=coarse.clamp_min(0)%ow*2
    good=active&(coarse>=0)
    padded=F.pad(fine,(1,1,1,1),value=-1)
    for dy in range(3):
        for dx in range(3):
            yy,xx=oy+dy-1,ox+dx-1
            prior=(yy*w+xx).masked_fill((yy<0)|(yy>=h)|(xx<0)|(xx>=w),-1)
            good=good & (padded[dy:dy+2*oh:2,dx:dx+2*ow:2]==prior)
    valid=previous.get('shallow_support')
    reuse=good&valid.flatten()[coarse.clamp_min(0)] if valid is not None else torch.zeros_like(active)
    old=previous['fine_levels'].flatten(2).index_select(2,fine.clamp_min(0).flatten()).view_as(levels)
    delta=levels-old.masked_fill(fine<0,0)
    def taps(values):
        summed=values.sum(dim=1,keepdim=True,dtype=torch.float32)
        return (F.avg_pool2d(summed,3,2,1)*9).round().int()[0,0]
    static=taps(levels!=0)
    return dict(sites=int(active.sum()),reuse=int(reuse.sum()),dense=int(active.sum())*levels.shape[1]*9,
                static=int(static[active].sum()),cold=int(static[active&~reuse].sum()),
                corrections=int(taps(delta!=0)[reuse].sum()),unit_ac=int(taps(delta.abs())[reuse].sum()),
                sent_addresses=int(delta.count_nonzero()),sent_units=int(delta.abs().sum()))


class ReadoutAudit:
    def __init__(self,bounds):
        self.bounds=bounds
        self.previous={}
        self.current={}

    def start(self,prev,meta,shape,refresh):
        self.prev=prev
        self.pose=meta['pose']
        self.shape=shape
        self.fresh=refresh is not None
        self.current={}
        self.route=None
        self.good={}

    def observe(self,name,module,value):
        k=module.kernel_size[0]
        assert module.groups==1 and module.stride==(1,1) and module.padding==(k//2,k//2) and module.dilation==(1,1)
        _,c,h,w=value.shape
        # Split only the actual 384-channel concatenation, in detector order.
        groups=3 if c==384 else 1
        fanout=module.out_channels
        def count(v):
            by_site=v.reshape(groups,c//groups,h,w).sum(1,dtype=torch.float32)
            if k==1:return by_site.int()
            return (F.avg_pool2d(by_site,k,1,k//2)*k**2).round().int()
        static=count(value!=0)
        fixed=aligned=static
        eligible=torch.zeros(h,w,dtype=torch.bool,device=value.device)
        if not self.fresh:
            old=self.previous[name]
            if self.route is None:
                bounds=block_lidar_range(self.bounds,self.shape,(h,w),2)
                codes=torch.arange(1,1+h*w,device=value.device,dtype=value.dtype).view(1,1,h,w)
                self.route=warp_bev(codes,self.prev['pose'],self.pose,bounds)[0].long().flatten()-1
            if k not in self.good:self.good[k]=geometry(self.route,h,w,k)
            eligible=self.good[k]
            warped=old.flatten(2).index_select(2,self.route.clamp_min(0)).view_as(value)
            warped.masked_fill_(self.route.view(1,1,h,w)<0,0)
            fixed=count(value!=old)
            aligned=torch.where(eligible,count(value!=warped),static)
        entry=[]
        for i in range(groups):
            entry.append(dict(dense=h*w*k*k*(c//groups)*fanout,
                 static=int(static[i].sum())*fanout,fixed=int(fixed[i].sum())*fanout,
                 aligned=int(aligned[i].sum())*fanout,
                 clean_fixed=0 if self.fresh else int((fixed[i]==0).sum()),
                 clean_aligned=int(((aligned[i]==0)&eligible).sum()),sites=h*w,eligible=int(eligible.sum())))
            entry[-1]['dirty_static']=int(static[i][fixed[i]!=0].sum())*fanout
        self.current[name]=value.detach().clone()
        return entry

    def finish(self):
        # Bytes for prior input and linear preactivation per layer; excludes temporaries.
        self.previous=self.current
        self.prev=None


def compact_cost(core,counts):
    enc,upd=core.coarse_light,core.coarse_update
    res=upd.candidate.residual
    groups={'radius2':(enc[0],enc[2],upd.candidate.history,res[0]),'radius1':(res[2],),
            'output':(res[4],res[6],upd.gate[0],upd.gate[2])}
    names={id(m):n for n,m in core.named_modules()}
    return {names[id(m)]:2*m.weight.numel()*counts[g] for g,modules in groups.items() for m in modules}


@torch.no_grad()
def run(a):
    torch.manual_seed(303)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=torch.backends.cuda.matmul.allow_tf32=False
    if a.training:
        plan=read(a.inputs/'plan.json')
        rows,traces,starts=training_rows(a.inputs,a.scans,plan)
        cfg=load_yaml(str(a.inputs/'config.yaml'))
        cfg['data_augment']=[]
        source=build_source(a.inputs,cfg)
        q,_,_=load_quantum(a.inputs/'quantum.json',plan,512)
    else:
        source,cfg,rows,traces,starts,_,q=prepare_stream(a.inputs,a.scans)
    book=read(a.book)
    assert book['levels']==[3,6,13,28] and book['cap']==428149
    old=[json.loads(line) for line in a.prior.read_bytes().splitlines()]
    stream_data,stream=loader(rows,cfg,0,303)
    models={}
    for n in ('calibrated','amplitude'):
        sender=FineCodebook(source.bounds,book['cap'],book['levels']) if n=='calibrated' else QuotaFineEdge(source.bounds,n,book['cap'])
        def select(fine,active,s=sender):
            s.active=active
            return s(fine)
        model=CrossScaleCache(source,'sparse_amplitude',planner=plan_host_coalesced_refresh,
                             deep_projector=DeepSpikeProjection(source.core.deblocks[2],q,'amplitude'),
                             fine_to_coarse=lambda x,s=sender:s(x) if s.phase=='refresh' else x)
        model.coarse_executor=FineInnovationUpdate(False,receive_fine=select)
        model.coarse_interface=model.coarse_projector=MiddleDecayProjection(source.core.deblocks[1],'zero','amplitude')
        models[n]=(model,sender,ReadoutAudit(source.bounds))
    meta_layers={}
    def hook(name,m,inp,out):
        v=inp[0]
        if isinstance(m,torch.nn.ConvTranspose2d):
            total=v.numel()*(m.out_channels//m.groups)
        else:
            total=out.numel()*(m.in_channels//m.groups)
        costs[name]+=2*total*m.kernel_size[0]*m.kernel_size[1]
        if name.startswith('readout.'):
            stats[name]=audit.observe(name,m,v)
            meta_layers[name]=dict(input=list(v.shape),output=list(out.shape),kernel=list(m.kernel_size),
                                    input_bytes=v.numel()*v.element_size(),output_bytes=out.numel()*out.element_size())
    hooks=[]
    for name,m in source.core.named_modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(lambda m,x,y,n=name:hook(n,m,x,y)))
    nrows=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for sample,trace,ref in zip(stream,traces,old):
            meta=sample['meta']
            frame,reason=meta['frame'],trace['refresh']
            assert frame==trace['frame']==ref['frame']
            if frame in starts:states=dict.fromkeys(models)
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            request=local_request(trace_masks(trace,spatial.device))
            row=dict(frame=frame,scene=meta['scene'],refresh=reason,arms={})
            for n,(model,sender,audit) in models.items():
                prev=states[n]
                assert model.refresh_reason(prev,meta)==reason
                sender.begin(prev,meta,spatial.shape[-2:],reason)
                model.coarse_projector.begin(prev,reason)
                audit.start(prev,meta,spatial.shape[-2:],reason)
                costs,stats=Counter(),{}
                (pred,state,info),other=measured_step(model,spatial,prev,meta,request)
                assert sum(costs.values())==other
                if reason is None:
                    work=model.coarse_executor.counts
                    cs=compact_cost(source.core,work)
                    assert sum(cs.values())==sum(conv_costs(source.core)[key]*v for key,v in work.items())
                    costs.update(cs)
                    model.coarse_executor.inputs=None
                expected=ref['arms']['deferred']['core'] if a.training else ref['arms']['calibrated']['core']
                assert sum(costs.values())==expected
                if not a.training and n=='calibrated' and meta['supervised']:
                    gt=stream_data.post.generate_gt_bbx(dict(ego=ego))
                    scored=json.loads(json.dumps(score(stream_data.post,ego,pred,gt)))
                    assert all(v==ref['arms'][n][k] for k,v in scored.items())
                sender.commit(state,False)
                for key,part in stats.items():assert sum(z['dense'] for z in part)*2==costs[key]
                row['arms'][n]=dict(core=dict(costs),readout=stats,fine=sender.ledger)
                if a.shallow and reason is None:
                    edge=shallow_ledger(model,state,prev,meta,spatial.shape[-2:])
                    assert edge['sent_addresses']==sender.ledger['addresses'] and edge['sent_units']==sender.ledger['units']
                    row['arms'][n]['shallow']=edge
                audit.finish()
                states[n]=state
            log.write(json.dumps(row)+'\n')
            nrows+=1
    for h in hooks:h.remove()
    assert nrows==len(rows)==len(old)==(256 if a.training else 272)
    write_json(a.output/'report.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               status='COMPLETED',scans=nrows,optimizer_steps=0,book=digest(a.book),reference=digest(a.prior),
               parameters=sum(p.numel() for p in source.parameters()),layers=meta_layers))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('inputs','scans','prior','book','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--training',action='store_true')
    p.add_argument('--shallow',action='store_true')
    args=p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    run(args)

