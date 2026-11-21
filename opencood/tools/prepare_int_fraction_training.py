"""Freeze labelled training windows and causal deep/middle update opportunities."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess

import torch
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.models.sub_modules.int_compact_middle_update import FineInnovationUpdate
from opencood.models.sub_modules.int_cross_scale_dw import CrossScaleCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_middle_decay_projection import MiddleDecayProjection
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_spike_stream import prepare_stream,read
from opencood.tools.eval_int_deep_block_online import ObservedPolicy,local_model,original_online_policy
from opencood.tools.eval_int_fusion_cache import replay_written
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader,to_device,write_json


def select_training(a):
    manifest=a.long_base/'r328_manifest.json'
    assert digest(manifest)=='184798f833dd52b6eee2cf2f3b04ab7684221be43b9302edccd4889f51d88738'
    native=read(manifest)
    scenes=set(read(a.book)['scenes'])
    assert len(scenes)==8 and not scenes.intersection(r['scene'] for r in native['val'])
    segments=defaultdict(list)
    for r in native['train']:
        if r['scene'] in scenes:segments[r['segment']].append(r)
    selected=set()
    rows=[]
    members={}
    for segment in segments.values():
        clip=segment[:32]
        scene=clip[0]['scene']
        if scene in selected or len(clip)<16 or sum(r['supervised'] for r in clip[1:])<8:continue
        selected.add(scene)
        for r in clip:
            assert r['split']=='train'
            for k,folder in (('pcd','velodyne'),('label','label/lidar')):
                if r[k] is None:continue
                path=a.vehicle/folder/Path(r[k]).name
                members[str(path)]=digest(path)
                r[k]=str(path)
            rows.append(r)
    assert selected==scenes and len(rows)==211 and sum(r['supervised'] for r in rows)==168
    return rows,members


@torch.no_grad()
def run(a):
    source,cfg,_,_,_,_,q=prepare_stream(a.inputs,a.scans)
    rows,members=select_training(a)
    write_json(a.output/'plan.json',dict(rows=rows,members=members,manifest=digest(a.long_base/'r328_manifest.json'),
               book=digest(a.book),seed=303,segments=list(dict.fromkeys(r['segment'] for r in rows))))
    teacher=local_model(source,indexed=True)
    control=CrossScaleCache(source,'sparse_amplitude',planner=plan_host_coalesced_refresh,
                            deep_projector=DeepSpikeProjection(source.core.deblocks[2],q,'amplitude'))
    control.coarse_interface=control.coarse_projector=MiddleDecayProjection(source.core.deblocks[1],'zero','amplitude')
    control.coarse_executor=FineInnovationUpdate(False)
    criterion=PointPillarDirLoss(cfg['loss']['args']).cuda().eval()
    _,stream=loader(rows,cfg,0,303)
    labelled=ordinary=0
    with (a.output/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta']
            if i==0 or rows[i-1]['segment']!=meta['segment']:
                ann=state=None
                policy=ObservedPolicy(teacher,original_online_policy)
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            reason=teacher.refresh_reason(ann,meta)
            assert control.refresh_reason(state,meta)==reason
            policy.observation=None
            _,ann,info=teacher.step_spatial(spatial,ann,meta,policy)
            trace=dict(frame=meta['frame'],scene=meta['scene'],segment=meta['segment'],refresh=reason,supervised=meta['supervised'])
            if reason is None:
                valid,required,extra=policy.observation
                written=required if extra is None else required|extra
                assert torch.equal(written,ann['deep_age']==0)
                trace.update(written_indices=written.flatten().nonzero().flatten().tolist(),
                             written=int(written.sum()),invalid=int((~valid).sum()))
            control.coarse_projector.begin(state,reason)
            pred,state,_=control.step_spatial(spatial,state,meta,replay_written(trace))
            if reason is None:
                assert torch.equal(state['deep_age']==0,written)
                skip=control.coarse_executor.skip
                trace.update(middle_indices=skip.flatten().nonzero().flatten().tolist(),middle_shape=list(skip.shape))
                control.coarse_executor.inputs=None
                ordinary+=1
            if meta['supervised']:
                trace['base_loss']=float(criterion(pred,ego['label_dict']))
                assert math.isfinite(trace['base_loss'])
                labelled+=1
            log.write(json.dumps(trace)+'\n')
    assert labelled==168 and i+1==211
    write_json(a.output/'report.json',dict(status='COMPLETED',scans=i+1,labelled=labelled,ordinary=ordinary,optimizer_steps=0,
               plan=digest(a.output/'plan.json'),frames=digest(a.output/'frames.jsonl'),
               code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('inputs','scans','book','long-base','vehicle','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    run(a)

