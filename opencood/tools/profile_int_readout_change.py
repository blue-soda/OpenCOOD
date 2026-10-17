"""Read-only full-stream readout innovation statistics; no prediction changes.

Tile scores are relative RMS feature differences after ego alignment. These are
candidate cache-reuse measurements, not measured AP, speedup or sparse execution.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.point_pillar_int import PointPillarINT
from opencood.models.sub_modules.int_feature_memory import warp_bev
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest


THRESHOLDS = (0., .05, .1, .2)


def tile_stats(current, previous, valid, tile=8):
    pool = lambda x: F.avg_pool2d(x, tile, tile, ceil_mode=True, count_include_pad=False)
    error = pool((current-previous).square().mean(1, keepdim=True))
    energy = torch.maximum(pool(current.square().mean(1, keepdim=True)),
                           pool(previous.square().mean(1, keepdim=True)))
    relative = (error/energy.clamp_min(1e-6)).sqrt()
    force = F.max_pool2d((~valid).float(), tile, tile, ceil_mode=True)>0
    result = {}
    for threshold in THRESHOLDS:
        active = (relative > threshold) | force
        result[str(threshold)] = dict(active_tiles_fraction=float(active.float().mean()),
            skipped_error_relative_rms=float((error.masked_fill(active, 0).sum()/energy.sum().clamp_min(1e-6)).sqrt()))
    return result


def check_tile_stats():
    x=torch.ones(1,2,8,16);valid=torch.ones(1,1,8,16,dtype=torch.bool)
    assert tile_stats(x,x,valid)['0.0']['active_tiles_fraction']==0
    other=x.clone();other[:,:,:,:8]=0
    assert tile_stats(x,other,valid)['0.1']['active_tiles_fraction']==.5
    valid[:,:,:,:8]=False
    assert tile_stats(x,x,valid)['0.0']['active_tiles_fraction']==.5


def worker(a):
    check_tile_stats()
    out=Path(a.output);train=Path(a.root)/('full_'+a.mode)
    runtime=json.loads((train/'effective_runtime.json').read_text())['arguments']
    contract=json.loads((train/'manifest.json').read_text())['contract']
    best=json.loads((train/'best.json').read_text())
    assert digest(best['checkpoint'])==best['sha256']
    assert digest(runtime['manifest'])==contract['manifest_sha256']
    assert digest(runtime['config'])==contract['config_sha256']
    rows=json.loads(Path(runtime['manifest']).read_text())['val']
    cfg=load_yaml(runtime['config']);cfg['data_augment']=[]
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    model=PointPillarINT(cfg['model']['args'],a.mode).cuda().eval()
    model.load_state_dict(torch.load(best['checkpoint'],map_location='cpu')['model'],strict=True)
    captured={}
    def capture(module,inputs,output):captured['readout']=output[0].detach()
    hook=model.feature_memory.fusion.register_forward_hook(capture)
    _,stream=loader(rows,cfg,4,runtime['seed'])
    state=previous=previous_pose=None;records=[]
    write_json(out/'status.json',dict(status='RUNNING'))
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample,expected in zip(stream,rows):
            meta=sample['meta']
            assert sample['ego'] is not None and meta['frame']==expected['frame']
            prediction,state,info=model.step(to_device(sample['ego'],torch.device('cuda')),state,meta)
            assert all(torch.isfinite(v).all() for v in prediction.values())
            current=captured['readout']
            if previous is None or info['reset']:
                aligned=torch.zeros_like(current)
                valid=torch.zeros_like(current[:,:1],dtype=torch.bool)
            else:
                aligned,valid=warp_bev(previous,previous_pose,meta['pose'],cfg['model']['args']['lidar_range'])
            stats=tile_stats(current,aligned,valid)
            record=dict(frame=meta['frame'],segment=meta['segment'],supervised=meta['supervised'],
                        reset=info['reset'],thresholds=stats)
            log.write(json.dumps(record)+'\n');records.append(record)
            previous=current;previous_pose=meta['pose']
            if len(records)%400==0:print('PROFILE',len(records),'/',len(rows),flush=True)
    hook.remove();assert len(records)==len(rows)
    summary={}
    for key in map(str,THRESHOLDS):
        summary[key]={metric:float(np.mean([r['thresholds'][key][metric] for r in records]))
                      for metric in ('active_tiles_fraction','skipped_error_relative_rms')}
    report=dict(status='COMPLETED',mode=a.mode,checkpoint=best,contract=contract,scans=len(records),
        labels=sum(r['supervised'] for r in records),reset_frames=sum(bool(r['reset']) for r in records),
        tile=8,thresholds=summary,scope='Unchanged model, full validation stream, teacher-forced one-step aligned readout differences; not accumulated cache error, AP or speedup',
        limitations=['Uses actual previous-frame full readout; does not simulate multi-frame cache staleness.',
                     'Does not propagate changed blocks through the backbone or account for its receptive-field halo.',
                     'Feature error magnitude does not establish detection accuracy under cache reuse.'])
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',default='spike_gru');p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        try:worker(a)
        except Exception as exc:
            write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<12000 or int(util)>5:raise ValueError('GPU busy')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.profile_int_readout_change','--root',str(Path(a.root).resolve()),
             '--output',str(out.resolve()),'--mode',a.mode,'--gpu',str(a.gpu),'--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(command=command,pid=proc.pid,gpu=a.gpu,created_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
