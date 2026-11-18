"""Same-card P5 fixed-mask latency, with an untimed cost and trajectory pass."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from opencood.models.sub_modules.int_deep_block_cache import DeepBlockCache
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.eval_int_deep_block_rank import checked_plan, clip_streams, SEED, TILE
from opencood.tools.launch_int_memory import digest
from opencood.tools.probe_int_deep_block_cache import SOURCE_SHA, add_tiles, build, masks_by_tile, spatial_input
from opencood.tools.run_int_ego import to_device, write_json
from opencood.tools.time_int_periodic_compact import exclusive


PASSES = 3
WARMUP = 16


def fingerprint(pred, state):
    h = hashlib.sha256()
    for data in (pred, {k: state[k] for k in ('value','coarse','deep')}):
        for key in sorted(data):
            h.update(data[key].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fixed_trace(cache, cfg, rows, plan, out):
    records = []
    masks = []
    with (out/'trace.jsonl').open('x',buffering=1) as writer:
        for _, ids, _, stream in clip_streams(plan, rows, cfg):
            states = {'p5':None,'local':None}
            for index, sample in enumerate(stream):
                meta = sample['meta']
                assert meta['frame']==ids[index]
                ego = to_device(sample['ego'],torch.device('cuda'))
                spatial = spatial_input(cache.source,ego)
                mask = None
                tile = None
                if cache.refresh_reason(states['local'],meta) is None:
                    frame = cache.prepare(spatial,states['local'],meta)
                    required = cache.mandatory(frame).cpu()
                    eligible = list(masks_by_tile(required,TILE))
                    if eligible:
                        tile = eligible[index*7%len(eligible)]
                        mask = add_tiles(torch.zeros_like(required),(tile,),TILE).cuda()
                arms={}
                for name,model in (('p5',cache.source),('local',cache)):
                    request = mask if name=='local' else None
                    (pred,states[name],info),flops=measured_step(model,spatial,states[name],meta,request)
                    arms[name]=dict(core_conv_flops=flops,refresh=info['full_refresh'],
                                    fingerprint=fingerprint(pred,states[name]) if index%16==0 else None)
                assert arms['p5']['refresh']==arms['local']['refresh']
                assert arms['p5']['core_conv_flops']>=arms['local']['core_conv_flops']
                masks.append(mask)
                row=dict(frame=meta['frame'],tile=tile,arms=arms)
                records.append(row)
                writer.write(json.dumps(row)+'\n')
    return masks, records


def time_trace(cache,cfg,rows,plan,masks,trace,out,uuid):
    measurements=[]
    checks=0
    with (out/'frames.jsonl').open('x',buffering=1) as raw:
        for repeat in range(PASSES):
            exclusive(uuid)
            checks+=1
            for _,ids,_,stream in clip_streams(plan,rows,cfg):
                states={'p5':None,'local':None}
                for i,sample in enumerate(stream):
                    if i%16==0:
                        exclusive(uuid)
                        checks+=1
                    meta=sample['meta']; assert meta['frame']==ids[i]==trace[i]['frame']
                    ego=to_device(sample['ego'],torch.device('cuda'))
                    order=('p5','local') if (i+repeat)%2==0 else ('local','p5')
                    info={}
                    for name in order:
                        model=cache.source if name=='p5' else cache
                        request=masks[i] if name=='local' else None
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        spatial=spatial_input(cache.source,ego)
                        pred,states[name],step=model.step_spatial(spatial,states[name],meta,requests=request)
                        torch.cuda.synchronize()
                        elapsed=(time.perf_counter()-start)*1000
                        gold=trace[i]['arms'][name]
                        assert step['full_refresh']==gold['refresh']
                        if gold['fingerprint']:
                            assert fingerprint(pred,states[name])==gold['fingerprint']
                        info[name]=dict(model_step_ms=elapsed,refresh=step['full_refresh'],core_conv_flops=gold['core_conv_flops'])
                    row=dict(frame=meta['frame'],index=i,repeat=repeat,included=i>=WARMUP,order=order,arms=info)
                    measurements.append(row)
                    raw.write(json.dumps(row,allow_nan=False)+'\n')
                    if (i+1)%16==0:
                        write_json(out/'status.json',dict(status='RUNNING',phase='timing',pass_index=repeat,scans=i+1))
    exclusive(uuid)
    return measurements,checks+1


def worker(a,out):
    exclusive(a.uuid)
    plan=checked_plan(a.timing_plan)
    assert plan['split']=='val' and len(plan['clips'])==1 and len(plan['clips'][0])>WARMUP
    a.kind='geometry'
    source,cfg,rows,_=build(a)
    source.period,source.max_age=5,.55
    before=model_digest(source)
    cache=DeepBlockCache(source,TILE,'light')
    timing_scope='Pillar to detection heads, includes actual mandatory-mask work, coalescing, packing, pose warp and state write; excludes precomputed optional mask selection, loader, transfer, postprocess, and reference hooks.'
    write_json(out/'contract.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        plan_sha256=digest(Path(a.timing_plan)),source_sha256=SOURCE_SHA,device=torch.cuda.get_device_name(),uuid=a.uuid,
        torch=torch.__version__,passes=PASSES,warmup_each=WARMUP,period=5,tile=TILE,scope=timing_scope))
    with torch.no_grad():
        masks,trace=fixed_trace(cache,cfg,rows,plan,out)
        measurements,checks=time_trace(cache,cfg,rows,plan,masks,trace,out,a.uuid)
    assert model_digest(source)==before
    valid=[r for r in measurements if r['included']]
    metrics={}
    for mode in ('p5','local'):
        elapsed=[r['arms'][mode]['model_step_ms'] for r in valid]
        assert len(elapsed)==PASSES*(len(trace)-WARMUP)
        metrics[mode]=dict(parameters=sum(p.numel() for p in source.parameters()),samples=len(elapsed),ap50=None,ap70=None,
            model_step_ms_mean=float(np.mean(elapsed)),model_step_ms_p95=float(np.percentile(elapsed,95)),
            ordinary_step_ms_mean=float(np.mean([r['arms'][mode]['model_step_ms'] for r in valid if r['arms'][mode]['refresh'] is None])),
            core_conv_flops_mean=float(np.mean([r['arms'][mode]['core_conv_flops'] for r in valid])),
            pass_means=[float(np.mean([r['arms'][mode]['model_step_ms'] for r in valid if r['repeat']==i])) for i in range(PASSES)])
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,exclusive_checks=checks,scope=timing_scope,
            timing_valid=True,weights_unchanged=True,scans_per_pass=len(trace)))
    write_json(out/'status.json',dict(status='COMPLETED',scans_per_pass=len(trace)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','checkpoint','trained','readout','timing-plan','uuid','output'):
        parser.add_argument('--'+name,required=True)
    a=parser.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:
        write_json(out/'status.json',dict(status='RUNNING',phase='inputs_and_fixed_trace'))
        worker(a,out)
    except Exception as exc:
        write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False))
        raise


if __name__=='__main__':
    main()
