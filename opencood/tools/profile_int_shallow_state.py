"""Fixed-weight train-prefix audit of a compact spatial-state scaffold."""
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
from opencood.models.sub_modules.int_projected_gru import projected_gru
from opencood.models.sub_modules.int_shallow_state import ShallowStateEncoder
from opencood.tools.train_int_anchor_residual import ConvCounter
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    for k,v in dict(mode='gru',stage='pilot',device='cuda',content_calibration=None,periodic_last=0,max_age=.25).items():setattr(a,k,v)
    model,cfg,_,source=setup(a)
    assert digest(a.checkpoint)==a.sha256=='1a267575ef735f203526fa3b8196334d32d9ff8a9080449c3b6206247913e5ef'
    saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['data']==source['contract'] and saved['contract']['source']['sha256']==source['checkpoint']['sha256']
    model.feature_memory=projected_gru(model.feature_memory,32)[0].cuda().eval()
    model.feature_memory.load_state_dict(saved['memory'],strict=True)
    original=model.backbone.resnet.layer0;bounds=cfg['model']['args']['lidar_range']
    modes={name:ShallowStateEncoder(original,bounds,execution='masked_dense' if name=='masked_dense' else 'sparse').cuda().eval()
        for name in ('dense_scaffold','sparse','masked_dense')}
    for module in [model]+list(modes.values()):
        for p in module.parameters():p.requires_grad_(False)
    arguments=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    rows=json.loads(Path(arguments['manifest']).read_text())['train'][:256]
    write_json(out/'provenance.json',dict(source=source,checkpoint=a.checkpoint,sha256=a.sha256,frames=len(rows),split='first256_train_scans',
        threshold=.5,tile=8,radius=4,period=3,fallback_fraction=.8,
        architecture='Current first BasicBlock stem, identity-initialized257to128 input mixer, two inherited BasicBlocks. One128-channel output state plus16-channel evidence reference at100x252. History mixer weights remain zero in this audit.',
        scope='Own spatial state recurrence; no detector AP. Dense scaffold equality is an initialization check, not learned temporal architecture quality. Counter-instrumented stage timing is not full model latency.'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    states={k:None for k in modes};memory=None;records={k:[] for k in ['original']+list(modes)};max_pair=max_dense=0.
    _,stream=loader(rows,cfg,2,303)
    with torch.no_grad():
        for index,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];p=ego['processed_lidar']
            if p['voxel_features'].shape[0]:
                batch={k:p[k] for k in ('voxel_features','voxel_coords','voxel_num_points')};batch['batch_size']=1
                spatial=model.scatter(model.pillar_vfe(batch))['spatial_features']
            else:
                nx,ny,_=model.scatter.model_cfg['grid_size'];spatial=model.cls_head.weight.new_zeros(1,64,int(ny),int(nx))
            temporal,memory,info=model.feature_memory.step(spatial,memory,meta)
            outputs={};names=list(records);offset=index%len(names);names=names[offset:]+names[:offset]
            for name in names:
                module=original if name=='original' else modes[name]
                counter=ConvCounter(module,torch.nn.ModuleList())
                try:
                    torch.cuda.synchronize();tick=time.perf_counter()
                    if name=='original':
                        value=original(temporal);detail=dict(actual_write_fraction=1.,executed_input_area_ratio=1.,fallback=True,refresh=True,state_bytes=0)
                    else:
                        value,states[name],detail=module.step(temporal,states[name],meta,reset=bool(info['reset']),dense=name=='dense_scaffold')
                    torch.cuda.synchronize();ms=1000*(time.perf_counter()-tick)
                finally:counter.close()
                assert torch.isfinite(value).all();outputs[name]=value
                records[name].append(dict(macs=counter.total,ms=ms,**detail))
            dense_error=float((outputs['dense_scaffold']-outputs['original']).abs().max())
            pair_error=float((outputs['sparse']-outputs['masked_dense']).abs().max())
            max_dense=max(max_dense,dense_error);max_pair=max(max_pair,pair_error)
            assert torch.allclose(outputs['dense_scaffold'],outputs['original'],atol=1e-4,rtol=1e-4),dense_error
            assert torch.allclose(outputs['sparse'],outputs['masked_dense'],atol=1e-3,rtol=1e-3),pair_error
            energy=outputs['original'].square().mean().clamp_min(1e-12)
            for name in records:records[name][-1]['relative_rms']=float(((outputs[name]-outputs['original']).square().mean()/energy).sqrt())
            if (index+1)%50==0:write_json(out/'status.json',dict(status='RUNNING',phase='train_prefix_audit',scans=index+1))
    results={}
    for name,values in records.items():
        results[name]=dict(conv_gflops=2*float(np.mean([v['macs'] for v in values]))/1e9,
            instrumented_stage_mean_ms=float(np.mean([v['ms'] for v in values[10:]])),
            instrumented_stage_p95_ms=float(np.percentile([v['ms'] for v in values[10:]],95)),
            mean_write_fraction=float(np.mean([v['actual_write_fraction'] for v in values])),
            mean_executed_update_input_area=float(np.mean([v['executed_input_area_ratio'] for v in values])),
            mean_planned_area_before_fallback=float(np.mean([v.get('crop_area_ratio_before_fallback',1.) for v in values])),
            fallback_frames=sum(int(v['fallback'] and not v['refresh']) for v in values),
            refresh_frames=sum(int(v['refresh']) for v in values),
            state_bytes=max(v['state_bytes'] for v in values),
            mean_relative_rms=float(np.mean([v['relative_rms'] for v in values])),
            p95_relative_rms=float(np.percentile([v['relative_rms'] for v in values],95)))
    write_json(out/'report.json',dict(status='COMPLETED',scans=len(rows),results=results,
        dense_identity_max_abs_error=max_dense,sparse_masked_max_abs_error=max_pair,
        scope='Measured stage-only convolution costs and instrumented timing. Frozen identity mixer; no training, AP, whole-model savings or learned history-prediction claim. Stage state owns its reused outputs; dense controls never overwrite it.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    parser=argparse.ArgumentParser()
    for key in ('root','checkpoint','sha256','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--worker',action='store_true');a=parser.parse_args();out=Path(a.output)
    if a.worker:
        def expired(s,f):raise TimeoutError('Spatial scaffold audit exceeded2 hours')
        signal.signal(signal.SIGALRM,expired);signal.alarm(7200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        finally:signal.alarm(0)
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('GPU occupied')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()),CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[sys.executable,'-m','opencood.tools.profile_int_shallow_state']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),timeout_seconds=7200,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
