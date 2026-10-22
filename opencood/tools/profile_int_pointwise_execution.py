"""Bounded kernel feasibility audit; synthetic masks are not a reuse policy."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_pointwise_spatial_state import ContextualPointwiseStage
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def worker(a):
    out=Path(a.output)
    write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    model,cfg,_,source=setup(a)
    assert digest(a.memory)=='1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'
    assert digest(a.checkpoint)==a.sha256
    memory=torch.load(a.memory,map_location='cpu');saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['data']==memory['contract']['data']==source['contract']
    assert saved['contract']['source']['sha256']==source['checkpoint']['sha256']
    model.feature_memory=projected_gru(model.feature_memory,32)[0].cuda().eval()
    model.feature_memory.load_state_dict(memory['memory'],strict=True)
    stage=ContextualPointwiseStage(model.backbone.resnet.layer0).cuda().eval()
    stage.load_state_dict(saved['stage'],strict=True)
    for m in (model,stage):
        for p in m.parameters():p.requires_grad_(False)
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    rows=json.loads(Path(arguments['manifest']).read_text())['train'][:32]
    scope=('Kernel feasibility only, first32 train scans and fixed epoch1 weights. '
        'Precomputed deterministic random masks, zero spatial history/valid as phaseA. '
        'No motion alignment, selector, state recurrence, detection AP or end-to-end speedup claim. '
        'Compare original Conv2d layout and row-major functional linear with identical weights. '
        'Timing includes mature stem, depthwise context, pack, selected gather, update, scatter; '
        'excludes mask construction, warp and all other detector modules. First4 scans warmup; three repeats per scan.')
    write_json(out/'provenance.json',dict(source=source,checkpoint=a.checkpoint,sha256=a.sha256,memory=a.memory,scope=scope))
    fractions=(0.,.1,.25,.5,.75,1.)
    records={str(f):[] for f in fractions};records['dense']=[]
    records.update({'linear_'+str(f):[] for f in fractions})
    costs={};memory_state=None;max_error=0.;max_relative=0.
    generator=torch.Generator().manual_seed(303)
    order=torch.randperm(100*252,generator=generator)
    masks={str(f):torch.zeros(100*252,dtype=torch.bool) for f in fractions}
    for f in fractions:masks[str(f)][order[:round(f*100*252)]]=True
    masks={k:v.reshape(1,1,100,252).cuda() for k,v in masks.items()}
    masks.update({'linear_'+k:v for k,v in list(masks.items())})
    _,stream=loader(rows,cfg,2,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));p=ego['processed_lidar']
            if p['voxel_features'].shape[0]:
                batch={k:p[k] for k in ('voxel_features','voxel_coords','voxel_num_points')};batch['batch_size']=1
                spatial=model.scatter(model.pillar_vfe(batch))['spatial_features']
            else:
                nx,ny,_=model.scatter.model_cfg['grid_size'];spatial=model.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
            temporal,memory_state,_=model.feature_memory.step(spatial,memory_state,sample['meta'])
            old=temporal.new_zeros(1,128,100,252);valid=old[:,:1]
            def execute(name):
                if name=='dense':return stage(temporal)
                current,evidence=stage.observe(temporal)
                if name.startswith('linear_'):return stage.linear_selected_update(current,evidence,old,valid,masks[name])
                return stage.selected_update(current,evidence,old,valid,masks[name])
            dense=execute('dense')
            for name in masks:
                selected=execute(name);target=torch.where(masks[name],dense,old)
                error=float((selected-target).abs().max());max_error=max(max_error,error)
                max_relative=max(max_relative,float(((selected-target).square().mean()/dense.square().mean().clamp_min(1e-12)).sqrt()))
                assert torch.allclose(selected,target,atol=1e-4,rtol=1e-4),(name,error)
                assert torch.equal(selected.masked_select(~masks[name]),old.masked_select(~masks[name]))
            if index==0:
                for name in records:
                    counter=ConvCounter(stage,torch.nn.ModuleList())
                    try:execute(name)
                    finally:counter.close()
                    costs[name]=counter.total
                    if name.startswith('linear_'):costs[name]+=stage.last_selected_linear_macs
                # Fixed current observation cost is paid even for an empty mask.
                assert costs['0.0']<costs['dense'] and costs['1.0']==costs['dense']
                for name,mask in masks.items():
                    expected=costs['0.0']+(costs['dense']-costs['0.0'])*int(mask.sum())/(100*252)
                    assert abs(costs[name]-expected)<1.,(name,costs[name],expected)
            for repeat in range(3):
                names=list(records);offset=(index*3+repeat)%len(names);names=names[offset:]+names[:offset]
                for name in names:
                    torch.cuda.synchronize();tick=time.perf_counter();value=execute(name);torch.cuda.synchronize()
                    elapsed=1000*(time.perf_counter()-tick)
                    if index>=4:records[name].append(elapsed)
            if (index+1)%8==0:write_json(out/'status.json',dict(status='RUNNING',scans=index+1))
    results={name:dict(conv_gflops=2*costs[name]/1e9,stage_mean_ms=float(np.mean(values)),
        stage_p95_ms=float(np.percentile(values,95)),timed_samples=len(values),
        update_output_fraction=1. if name=='dense' else float(masks[name].float().mean())) for name,values in records.items()}
    write_json(out/'report.json',dict(status='COMPLETED',scans=32,results=results,max_abs_error=max_error,
        max_relative_rms=max_relative,scope=scope))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    parser=argparse.ArgumentParser()
    for key in ('root','memory','checkpoint','sha256','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--worker',action='store_true')
    a=parser.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Pointwise kernel audit exceeded30 minutes')
        signal.signal(signal.SIGALRM,expired);signal.alarm(1800)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.profile_int_pointwise_execution']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=1800,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
