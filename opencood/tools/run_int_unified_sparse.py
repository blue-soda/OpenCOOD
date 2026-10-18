"""Train-only gate calibration and paired dense/masked/sparse inference pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.models.sub_modules.int_unified_bev import UnifiedBEVDetector
from opencood.models.sub_modules.int_unified_sparse import SelectiveUnifiedBEV
from opencood.models.sub_modules.int_cache_repair import refresh_tiles
from opencood.tools.run_int_unified_bev import engineering
from opencood.tools.run_int_cache_repair import setup,evaluate
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def wait_checkpoint(a,out):
    path=Path(a.checkpoint);deadline=time.monotonic()+a.wait_minutes*60
    while not path.exists() or not (path.parent/'report.json').exists():
        write_json(out/'status.json',dict(status='WAITING_CHECKPOINT',checkpoint=str(path)))
        status_file=path.parent.parent/'status.json'
        if status_file.exists() and json.loads(status_file.read_text()).get('status')=='FAILED':
            raise RuntimeError('Source training failed before checkpoint was ready')
        if time.monotonic()>=deadline:raise RuntimeError('Timed out waiting for immutable epoch checkpoint/report')
        time.sleep(20)
    report=json.loads((path.parent/'report.json').read_text())
    assert report['checkpoint_sha256']==digest(path)
    return torch.load(str(path),map_location='cpu')


def integration(a,student,cfg,rows,out):
    full=SelectiveUnifiedBEV(student,0.,a.tile,execution='full')
    state=reference=None;errors=[];captured=[]
    hook=student.core.update.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()) if not captured else None)
    _,stream=loader(rows[:8],cfg,2,303)
    with torch.no_grad():
        try:
            for sample in stream:
                ego=to_device(sample['ego'],torch.device('cuda'))
                expected,reference,_=student.step(ego,reference,sample['meta'])
                actual,state,_=full.step(ego,state,sample['meta'])
                err=max(float((expected[k]-actual[k]).abs().max()) for k in expected)
                assert all(torch.allclose(expected[k],actual[k],atol=1e-5,rtol=1e-5) for k in expected)
                errors.append(err)
        finally:hook.remove()
        x=captured[0];expected=student.core.update(x);mask=torch.zeros_like(expected[:,:1],dtype=torch.bool)
        mask[...,0,0]=True;mask[...,50,126]=True;mask[...,-1,-1]=True
        actual,written,stats=refresh_tiles(student.core.update,x,torch.zeros_like(expected),mask,1,2,a.tile,allow_fallback=False,
                                         layout=a.layout,batch_size=a.crop_batch)
        selected=written.expand_as(actual);error=float((actual-expected).abs()[selected].max())
        assert torch.allclose(actual[selected],expected[selected],atol=1e-4,rtol=1e-4)
        assert torch.equal(actual[~selected],torch.zeros_like(actual[~selected]))
        gather_error=None
        if a.executor=='gather':
            gathered,gw,_=refresh_tiles(student.core.update,x,torch.zeros_like(expected),mask,1,2,a.tile,
                                       allow_fallback=False,execution='gather')
            assert torch.equal(written,gw)
            gather_error=float((gathered-expected).abs()[selected].max())
            assert torch.allclose(gathered[selected],expected[selected],atol=1e-4,rtol=1e-4)
            assert torch.equal(gathered[~selected],torch.zeros_like(gathered[~selected]))
    result=dict(status='PASSED',dense_wrapper_prediction_max_errors=errors,real_weight_crop_max_error=error,
                real_weight_crop_stats=stats,real_weight_gather_max_error=gather_error,checkpoint_sha256=digest(Path(a.checkpoint)))
    write_json(out/'selective_engineering.json',result)


def calibrate(a,student,cfg,rows,out):
    wrapper=SelectiveUnifiedBEV(student,0.,a.tile,execution='full')
    state=None;values=[];chosen=rows[:a.calibration_frames]
    _,stream=loader(chosen,cfg,4,303)
    with torch.no_grad(),(out/'calibration_frames.jsonl').open('x',buffering=1) as log:
        for index,(row,sample) in enumerate(zip(chosen,stream)):
            assert sample['meta']['frame']==row['frame']
            ego=to_device(sample['ego'],torch.device('cuda'))
            _,state,info=wrapper.step(ego,state,sample['meta'])
            scores=wrapper.last_scores[wrapper.last_eligible].cpu().numpy()
            if len(scores):values.append(scores)
            log.write(json.dumps(dict(frame=row['frame'],segment=row['segment'],reset=info['reset'],eligible_tiles=len(scores)))+'\n')
            if (index+1)%200==0:write_json(out/'status.json',dict(status='RUNNING',phase='calibrate_train',scans=index+1))
    assert index+1==len(chosen) and values
    values=np.concatenate(values);assert np.isfinite(values).all()
    thresholds={name:float(np.quantile(values,q)) for name,q in [('q50',.5),('q75',.75)]}
    result=dict(status='COMPLETED',split='train',scans=len(chosen),eligible_tiles=len(values),thresholds=thresholds,
        tile=a.tile,max_age=a.max_age,fallback_fraction=.8,layout=a.layout,crop_batch=a.crop_batch,checkpoint_sha256=digest(Path(a.checkpoint)),
        descriptor='Tile RMS evidence difference / max current/reference RMS; all evidence channels; no GT or current full-update output',
        caveat='Fixed training prefix, dense-history calibration; quantiles are not guaranteed update budgets under accumulated sparse history.')
    write_json(out/'calibration.json',result)
    return result


def worker(a):
    out=Path(a.output);saved=wait_checkpoint(a,out)
    assert saved['contract']['version']=='unified_bev_dense_v2' and saved['contract']['memory_policy']=='recurrent'
    for key,value in dict(mode='gru',stage='pilot',content_calibration=None,periodic_last=0,oracle_stage0=False,
                          oracle_index=0,support_profile=False,zero_uncovered=False).items():setattr(a,key,value)
    write_json(out/'status.json',dict(status='RUNNING',phase='setup'))
    teacher,cfg,val,provenance=setup(a);teacher.eval()
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract']
    write_json(out/'provenance.json',provenance)
    student=UnifiedBEVDetector(teacher,cfg,'v2','recurrent').cuda().eval()
    student.load_state_dict(saved['model'],strict=True)
    for p in student.parameters():p.requires_grad_(False)
    a.engineering=str(out/'engineering.json')
    engineering(student,teacher,cfg,val,out);integration(a,student,cfg,val,out)
    runtime=json.loads((Path(a.root)/'full_gru'/'effective_runtime.json').read_text())['arguments']
    train=json.loads(Path(runtime['manifest']).read_text())['train']
    calibration=calibrate(a,student,cfg,train,out)
    write_json(out/'experiment_contract.json',dict(checkpoint=a.checkpoint,checkpoint_sha256=digest(Path(a.checkpoint)),
        source_epoch=saved['epoch'],source_training_contract=saved['contract'],tile=a.tile,max_cell_age=a.max_age,
        layout=a.layout,crop_batch=a.crop_batch,executor=a.executor,
        calibration=calibration,validation_scans=len(val) if a.frames==0 else min(a.frames,len(val)),
        rule='Fixed thresholds from training prefix; force unseen/aged cells; fallback if halo crop input area>=.8; no feature training',
        timing='Same GPU sequential runs, dense control before/after; includes selector/cache/warp/crops/readout; other GPUs may be active'))
    plans=[('dense_before',None,None),('sparse_q50',a.executor,'q50'),('masked_q75','masked_dense','q75'),
           ('sparse_q75',a.executor,'q75'),('dense_after',None,None)]
    reports={}
    for name,mode,q in plans:
        directory=out/name;directory.mkdir()
        write_json(out/'status.json',dict(status='RUNNING',phase=name))
        subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],
                       stdout=(directory/'gpu_snapshot.csv').open('w'),check=True)
        wrapper=student if mode is None else SelectiveUnifiedBEV(student,calibration['thresholds'][q],a.tile,a.max_age,mode,
                                                               layout=a.layout,batch_size=a.crop_batch)
        a.stage='full' if a.frames==0 else 'pilot'
        torch.cuda.reset_peak_memory_stats()
        report=evaluate(a,teacher,cfg,val,directory,wrapper)
        records=[json.loads(s) for s in (directory/'frames.jsonl').read_text().splitlines()]
        chosen=val if a.frames==0 else val[:a.frames]
        assert [(x['frame'],x['segment'],x['supervised']) for x in records]==[(x['frame'],x['segment'],x['supervised']) for x in chosen]
        report.update(name=name,execution=mode,checkpoint_sha256=digest(Path(a.checkpoint)),source_epoch=saved['epoch'],
            scope='Full validation' if a.frames==0 else 'Early-checkpoint validation prefix; engineering/latency pilot only',
            caveats=['Fixed-weight sparse inference intervention; not a trained sparse model or convergence result.',
                     'Evidence encoder/context and readout/head remain dense. Reference+age increase cache and warp costs.',
                     'Same GPU paired sequential model time excludes IO/voxelization/postprocess; other GPU training may influence host load.',
                     'Do not select gate thresholds on validation AP or report prefix AP as full-set AP.'])
        if mode is not None:
            report['update_summary']={k:float(np.mean([x['update_stats'][k] for x in records])) for k in
                ['actual_write_fraction','executed_input_area_ratio','crop_area_ratio_before_fallback','fallback','tiles']}
            report['reuse_frames']=sum(x['reused_fraction']>0 for x in records)
            report['max_cell_age']=max(x['max_cell_age'] for x in records)
            assert report['max_cell_age']<a.max_age+1e-6
        write_json(directory/'report.json',report);reports[name]=report
    assert reports['dense_before']['metrics']==reports['dense_after']['metrics']
    left=[json.loads(x) for x in (out/'masked_q75'/'frames.jsonl').read_text().splitlines()]
    right=[json.loads(x) for x in (out/'sparse_q75'/'frames.jsonl').read_text().splitlines()]
    assert all(x['update_stats']['actual_write_fraction']==y['update_stats']['actual_write_fraction'] and
               x['update_stats']['fallback']==y['update_stats']['fallback'] for x,y in zip(left,right))
    final=dict(status='COMPLETED',checkpoint_sha256=digest(Path(a.checkpoint)),source_epoch=saved['epoch'],
        tile=a.tile,executor=a.executor,calibration=calibration,reports=reports,masked_sparse_write_masks_counts_agree=True,
        dense_before_after_ms_ratio=reports['dense_after']['model_step_ms_mean']/reports['dense_before']['model_step_ms_mean'])
    write_json(out/'report.json',final);write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--tile',type=int,choices=[8,16,32],default=16);p.add_argument('--max-age',type=float,default=.5)
    p.add_argument('--frames',type=int,default=512,help='Validation prefix; 0 selects full validation')
    p.add_argument('--calibration-frames',type=int,default=1024);p.add_argument('--wait-minutes',type=float,default=30.)
    p.add_argument('--layout',choices=['merged','fixed'],default='merged');p.add_argument('--crop-batch',type=int,default=8)
    p.add_argument('--executor',choices=['sparse','gather'],default='sparse')
    p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.frames<0 or a.calibration_frames<8 or a.max_age<=0:p.error('Invalid frame/age limits')
    if a.worker:
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<22000 or int(util)>5:raise RuntimeError('Requested GPU is busy')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.run_int_unified_sparse']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(command=command,pid=proc.pid,gpu=a.gpu,created_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
