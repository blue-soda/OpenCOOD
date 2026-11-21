"""Freeze middle send quantums on training-scene counts, then evaluate held-out clips."""
import argparse
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.sub_modules.int_block_cache import block_lidar_range
from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.models.sub_modules.int_deep_spike_projection import DeepSpikeProjection
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.models.sub_modules.int_pyramid_regions import plan_host_coalesced_refresh
from opencood.tools.check_int_deep_spike_stream import prepare_stream, read
from opencood.tools.eval_int_deep_block_membrane import TaskStateInterface, score_arm
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import spatial_input
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.train_int_deep_block_adapt import local_request, trace_masks


GRID = tuple(i/128 for i in range(32,65))
TRAIN_SHA = '38d881766b73bdc1a3aaa3a185313d8f3406cd68571c32b3b64bd4368b72c5ec'


def detector(source, deep_quantum, mode, quantum):
    return DeepBlockCache(source, 8, 'light', crop_first=True, planner=plan_host_coalesced_refresh,
                          coarse_interface=TaskStateInterface(mode, quantum),
                          deep_projector=DeepSpikeProjection(source.core.deblocks[2], deep_quantum, 'amplitude'))


def aligned_previous(previous, new, spatial, meta, bounds):
    area = block_lidar_range(bounds, spatial.shape[-2:], new.shape[-2:], 4)
    return warp_bev(previous['coarse'], previous['pose'], meta['pose'], area)[0]


def event_counts(value, aligned, quantum):
    delta = torch.round(value / quantum) - torch.round(aligned / quantum)
    # Integer counts in float64 are exact for these tensors.
    return [int(delta.count_nonzero()), int(delta.abs().sum(dtype=torch.float64))]


def training_rows(inputs, training, val_plan):
    plan = read(training/'train_plan.json')
    assert digest(training/'train_plan.json') == TRAIN_SHA
    traces = read(inputs/'traces.json')['train'][::4]
    assert plan['split'] == 'train'
    assert not ({x[0] for x in plan['scenes'][::4]} & {x[0] for x in val_plan['scenes']})
    assert [[r['frame'] for r in trace] for trace in traces] == plan['clips'][::4]
    rows = read(training/'rows.json')
    assert [r['frame'] for r in rows] == [r['frame'] for trace in traces for r in trace]
    for row in rows:
        original = row['pcd']
        path = training/'pcd'/Path(original).name
        assert digest(path) == plan['sha256'][original]
        # Only the original frame, pose and point cloud are used for calibration.
        row.update(pcd=str(path), label=None, supervised=False)
    return rows, [r for clip in traces for r in clip], {c[0]['frame'] for c in traces}


def calibrate(inputs, training, cfg, source, q, out):
    rows, traces, starts = training_rows(inputs, training, read(inputs/'plan.json'))
    models = {m:detector(source,q,m,.25) for m in ('carry','drop')}
    totals = {x:[0,0] for x in GRID}
    target = [0,0]
    _, stream = loader(rows,cfg,0,303)
    with (out/'calibration_frames.jsonl').open('x', buffering=1) as raw:
        for sample, trace in zip(stream,traces):
            meta = sample['meta']
            assert meta['frame'] == trace['frame'] and not meta['supervised']
            spatial = spatial_input(source,to_device(sample['ego'],torch.device('cuda')))
            masks = trace_masks(trace,spatial.device)
            if meta['frame'] in starts:
                states = dict.fromkeys(models)
            record = dict(frame=meta['frame'],scene=meta['scene'],refresh=trace['refresh'])
            for mode, model in models.items():
                prev=states[mode]
                _, state, info=model.step_spatial(spatial,prev,meta,local_request(masks))
                states[mode]=state
                assert trace['refresh'] == info['full_refresh']
                if trace['refresh'] is not None:
                    continue
                value=state['coarse']
                old=aligned_previous(prev,value,spatial,meta,source.bounds)
                if mode=='carry':
                    # Neither the middle transition nor the fine sender consumes deep feedback.
                    # Carry retains the unrounded middle state at every candidate quantum.
                    curve=[event_counts(value,old,x) for x in GRID]
                    record['carry']=curve
                    for x,pair in zip(GRID,curve):
                        for j in (0,1):
                            totals[x][j]+=pair[j]
                else:
                    pair=event_counts(value,old,.25)
                    record['drop']=pair
                    for j in (0,1):
                        target[j]+=pair[j]
            raw.write(json.dumps(record)+'\n')
    assert sum(1 for _ in (out/'calibration_frames.jsonl').open())==len(rows)
    chosen = {key:min(GRID,key=lambda x:abs(totals[x][j]-target[j]))
              for j,key in enumerate(('address','unit'))}
    write_json(out/'calibration.json',dict(grid=GRID,totals=totals,target=target,chosen=chosen,
               training_plan_sha=TRAIN_SHA,rows_sha=digest(training/'rows.json'),scans=len(rows),
               scenes=sorted({r['scene'] for r in rows}),labels_used=0))
    return chosen


@torch.no_grad()
def run(inputs, training, scans, reference, out):
    source,cfg,rows,traces,starts,members,q=prepare_stream(inputs,scans)
    old_contract=read(reference/'contract.json')
    assert members==old_contract['members']
    chosen=calibrate(inputs,training,cfg,source,q,out)
    pos=GRID.index(chosen['unit'])
    quantums=sorted({.25,chosen['unit'],chosen['address'],GRID[max(0,pos-1)],GRID[min(len(GRID)-1,pos+1)]})
    modes={f'carry_{int(x*128)}':('carry',x) for x in quantums}
    modes['drop_32']=('drop',.25)
    models={k:detector(source,q,*v) for k,v in modes.items()}
    write_json(out/'contract.json',dict(code=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
               source=old_contract['source'],members=members,trace=digest(inputs/'traces.json'),
               old_stats=digest(reference/'ap_records.json'),modes=modes,
               chosen=chosen,grid=GRID,params=old_contract['params'],device=torch.cuda.get_device_name(),
               torch=torch.__version__,quality='R-320 frozen closed loop; calibration uses event counts only'))
    stats={k:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.5,.7)} for k in modes}
    data,stream=loader(rows,cfg,0,303)
    with (out/'frames.jsonl').open('x',buffering=1) as raw:
        for sample,trace in zip(stream,traces):
            meta=sample['meta']
            assert meta['frame']==trace['frame'] and meta['supervised']==trace['supervised']
            ego=to_device(sample['ego'],torch.device('cuda'))
            spatial=spatial_input(source,ego)
            masks=trace_masks(trace,spatial.device)
            if meta['frame'] in starts:
                states=dict.fromkeys(models)
            gt=data.post.generate_gt_bbx(dict(ego=ego)) if meta['supervised'] else None
            rec=dict(frame=meta['frame'],scene=meta['scene'],refresh=trace['refresh'],supervised=meta['supervised'],arms={})
            for k,model in models.items():
                prev=states[k]
                pred,state,info=model.step_spatial(spatial,prev,meta,local_request(masks))
                states[k]=state
                assert info['full_refresh']==trace['refresh']
                counts=[0,0]
                if trace['refresh'] is None:
                    value=state['coarse']
                    counts=event_counts(value,aligned_previous(prev,value,spatial,meta,source.bounds),modes[k][1])
                if gt is not None:
                    score_arm(data.post,dict(ego=ego),pred,gt,stats[k])
                rec['arms'][k]=dict(addresses=counts[0],units=counts[1],deep=model.deep_projector.ledger,
                                    score_end=len(stats[k][.7]['score']),gt_end=stats[k][.7]['gt'])
            raw.write(json.dumps(rec)+'\n')
    assert sum(1 for _ in (out/'frames.jsonl').open())==len(rows)
    old=read(reference/'ap_records.json')
    for m,k in [('middle_drop','drop_32'),('middle_carry','carry_32')]:
        assert {str(t):s for t,s in stats[k].items()}==old[m]
    write_json(out/'ap_records.json',stats)
    write_json(out/'report.json',dict(status='COMPLETED',scans=len(rows),old_detections_exact=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('inputs','training','scans','reference','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    run(args.inputs,args.training,args.scans,args.reference,args.output)
