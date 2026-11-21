"""Supervise a folded fine-history gain on matched full temporal trajectories."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

import torch

from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate,conv_costs
from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_fine_codebook import FineCodebook
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.models.sub_modules.int_nonzero_fusion import NonzeroFusionCache
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_spike_stream import prepare_stream,read
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.eval_int_fine_edge import MatchedMiddle
from opencood.tools.eval_int_fusion_cache import complete_scenes,replay_written
from opencood.tools.eval_int_middle_cap import score
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json


class HistoryExperiment:
    def __init__(self,source,quantum,book):
        self.source=source
        core=source.core
        history=core.fine_update.history
        assert history.weight.shape==(128,129,1,1) and history.bias is None
        self.history_columns=history.weight[:,:128]
        self.frozen_columns=self.history_columns.detach().clone()
        assert self.frozen_columns.abs().max()>0
        fusion=core.readout.layers[0].double_conv[0]
        self.layer=NonzeroFusionCache(fusion[0]).eval()
        fusion[0]=self.layer
        self.costs=conv_costs(core)
        self.models={}
        for name in ('continuous','carry','drop'):
            edge=None if name=='continuous' else FineCodebook(source.bounds,book['cap'],book['levels'])
            model=CrossScaleCache(source,'sparse_amplitude',planner=plan_host_coalesced_refresh,
                                  deep_projector=DeepSpikeProjection(core.deblocks[2],quantum,'amplitude'),fine_to_coarse=edge)
            model.coarse_interface=model.coarse_projector=MiddleDecayProjection(core.deblocks[1],'zero','amplitude')
            model.coarse_executor=FineInnovationUpdate(False) if edge is None else MatchedMiddle()
            self.models[name]=(model,edge)

    def set_gain(self,theta):
        self.history_columns.copy_(self.frozen_columns*math.exp(theta))

    def step(self,name,spatial,meta,prev,trace,mask=None):
        model,edge=self.models[name]
        reason=trace['refresh']
        assert meta['frame']==trace['frame'] and model.refresh_reason(prev,meta)==reason
        if edge is not None:
            edge.begin(prev,meta,spatial.shape[-2:],reason)
            if reason is None:
                model.coarse_executor.skip=mask
                edge.active=~mask
        model.coarse_projector.begin(prev,reason)
        self.layer.begin(prev,reason,True)
        (pred,state,_),cost=measured_step(model,spatial,prev,meta,replay_written(trace))
        stats={}
        if reason is None:
            exe=model.coarse_executor
            cost+=sum(self.costs[k]*v for k,v in exe.counts.items())
            assert (state['deep_age']==0).flatten().nonzero().flatten().tolist()==trace['written_indices']
            if edge is None:
                exe.inputs=None
            else:
                stats.update(addresses=edge.ledger['addresses'],bytes=edge.ledger['bytes'],units=edge.ledger['units'])
                if name=='drop':
                    value=state['value']
                    state['value']=torch.where(edge.read,(value/edge.quantum).round()*edge.quantum,value)
        if edge is not None:edge.commit(state,False)
        state['fusion_cache']=self.layer.new
        stats['core']=cost+self.layer.ledger['float_conv']
        return pred,state,stats


def numeric_mask(trace):
    if trace['refresh'] is not None:return None
    mask=torch.zeros(trace['middle_shape'],device='cuda',dtype=torch.bool)
    mask.view(-1)[trace['middle_indices']]=True
    return mask


def window_loss(exp,criterion,entries,name,theta):
    exp.set_gain(theta)
    losses=[]
    budgets=[]
    state=None
    for meta,spatial,target,trace in entries:
        pred,state,stats=exp.step(name,spatial.cuda(),meta,state,trace,numeric_mask(trace))
        if trace['refresh'] is None:budgets.append((stats['addresses'],stats['bytes']))
        if target is not None:
            losses.append(float(criterion(pred,to_device(target,torch.device('cuda')))))
    assert losses and all(math.isfinite(x) for x in losses)
    return sum(losses)/len(losses),budgets


def fit(args,exp,cfg):
    trace_dir=args.training
    audit=read(trace_dir/'report.json')
    plan=read(trace_dir/'plan.json')
    assert audit['plan']==digest(trace_dir/'plan.json') and audit['frames']==digest(trace_dir/'frames.jsonl')
    assert all(digest(Path(p))==v for p,v in plan['members'].items())
    trace={r['frame']:r for r in map(json.loads,(trace_dir/'frames.jsonl').read_bytes().splitlines())}
    assert len(trace)==len(plan['rows'])==211 and plan['book']==digest(args.book)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda().eval()
    names=('carry','drop')
    params={n:torch.zeros((),dtype=torch.float64) for n in names}
    opts={n:torch.optim.Adam([params[n]],lr=.01) for n in names}
    steps=0
    with (args.output/'steps.jsonl').open('x',buffering=1) as log:
        for segment,group in itertools.groupby(plan['rows'],key=lambda r:r['segment']):
            rows=list(group)
            _,stream=loader(rows,cfg,0,303)
            entries=[]
            for sample in stream:
                meta=sample['meta']
                ego=sample['ego']
                spatial=spatial_input(exp.source,to_device(ego,torch.device('cuda'))).cpu()
                entries.append((meta,spatial,ego.get('label_dict'),trace[meta['frame']]))
            record=dict(segment=segment,scans=len(rows),labels=sum(r['supervised'] for r in rows),arms={})
            common=None
            for name in names:
                theta=float(params[name])
                plus,first=window_loss(exp,criterion,entries,name,theta+.02)
                minus,second=window_loss(exp,criterion,entries,name,theta-.02)
                if common is None:common=first
                assert first==second==common and all(x==(428149,160556) for x in first)
                gradient=(plus-minus)/.04
                params[name].grad=torch.tensor(gradient,dtype=torch.float64)
                opts[name].step()
                params[name].clamp_(-.20,.20)
                record['arms'][name]=dict(before=theta,plus=plus,minus=minus,gradient=gradient,after=float(params[name]))
            record['ordinary']=len(common)
            log.write(json.dumps(record)+'\n')
            steps+=1
    assert steps==8
    return dict(status='COMPLETED',optimizer_steps_per_arm=steps,training_frames=211,labelled=168,
                theta={k:float(v) for k,v in params.items()},train_plan=audit['plan'],training_trace=audit['frames'])


def evaluate(args,exp,cfg,short):
    rows,traces,hashes=complete_scenes(args,short)
    prior=[json.loads(r) for r in args.prior.read_bytes().splitlines()]
    fit_report=read(args.fit/'report.json')
    assert fit_report['status']=='COMPLETED' and fit_report['optimizer_steps_per_arm']==8
    assert read(args.fit/'contract.json')['book']==digest(args.book)
    theta=dict(continuous=0.,**fit_report['theta'])
    support={r['frame']:r for r in map(json.loads,args.support.read_bytes().splitlines())}
    assert len(prior)==len(rows)==1014 and len(support)==458
    data,stream=loader(rows,cfg,0,303)
    with (args.output/'frames.jsonl').open('x',buffering=1) as log:
        for i,(sample,trace,ref) in enumerate(zip(stream,traces,prior)):
            meta=sample['meta']
            assert all(meta[k]==ref[k] for k in ('scene','frame','segment'))
            reason=trace['refresh']
            assert reason==ref['refresh']
            if i==0 or rows[i-1]['segment']!=meta['segment']:
                states=dict.fromkeys(theta,dict(scene='') if reason!='initial' else None)
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(exp.source,ego)
            gt=None
            if meta['supervised']:
                gt=data.post.generate_gt_bbx(dict(ego=ego))
                cpu_gt=data.post.generate_gt_bbx(dict(ego=sample['ego']))
                assert torch.allclose(cpu_gt,gt.cpu(),atol=1e-4,rtol=0)
                assert hashlib.sha256(cpu_gt.numpy().tobytes()).hexdigest()==support[meta['frame']]['gt_xyz_sha256']
            row={k:ref[k] for k in ('scene','frame','segment','refresh','frame_age')}
            row['arms']={}
            mask=None
            for name in ('continuous','carry','drop'):
                exp.set_gain(theta[name])
                pred,states[name],stats=exp.step(name,spatial,meta,states[name],trace,mask)
                if name=='continuous' and reason is None:
                    mask=exp.models[name][0].coarse_executor.skip
                    row['middle_mask_sha256']=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
                    assert row['middle_mask_sha256']==ref['middle_mask_sha256']
                if gt is not None:stats.update(score(data.post,ego,pred,gt,record_matches=True))
                if name=='continuous':
                    assert json.loads(json.dumps(stats))==ref['arms']['continuous']
                row['arms'][name]=stats
            if reason is None:
                assert all(row['arms'][n]['addresses']==428149 and row['arms'][n]['bytes']==160556 for n in ('carry','drop'))
            log.write(json.dumps(row)+'\n')
    assert i+1==len(rows)
    return dict(status='COMPLETED',scans=len(rows),training_fit=digest(args.fit/'report.json'),
                reference=digest(args.prior),native=hashes,theta=theta)


@torch.no_grad()
def run(args):
    source,cfg,short,_,_,_,q=prepare_stream(args.inputs,args.scans)
    book=read(args.book)
    assert book['levels']==[3,6,13,28] and book['cap']==428149
    exp=HistoryExperiment(source,q,book)
    write_json(args.output/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               book=digest(args.book),params=sum(p.numel() for p in source.parameters()),seed=303,
               trained_degrees_per_arm=1,criterion='PointPillarDirLoss',mode=args.mode))
    result=fit(args,exp,cfg) if args.mode=='fit' else evaluate(args,exp,cfg,short)
    write_json(args.output/'report.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['fit','eval'])
    for k in ('inputs','scans','book','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('training','fit','prior','support','long-base','vehicle'):p.add_argument('--'+k,type=Path)
    args=p.parse_args()
    required=['training'] if args.mode=='fit' else ['fit','prior','support','long_base','vehicle']
    if any(getattr(args,k) is None for k in required):p.error('missing inputs: '+', '.join(required))
    args.output.mkdir(parents=True,exist_ok=False)
    run(args)

