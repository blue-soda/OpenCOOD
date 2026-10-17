"""Isolated fixed-weight engineering/pilot/full evaluation of local cache repair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.point_pillar_int import PointPillarINT
from opencood.models.sub_modules.int_cache_repair import CacheRepairDetector, PeriodicStageDetector, refresh_tiles
from opencood.models.sub_modules.int_cache_gate import DESCRIPTOR_VERSION
from opencood.tools.run_int_ego import loader, to_device, write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def setup(a):
    root=Path(a.root);train=root/('full_'+a.mode)
    args=json.loads((train/'effective_runtime.json').read_text())['arguments']
    contract=json.loads((train/'manifest.json').read_text())['contract']
    best=json.loads((train/'best.json').read_text())
    assert digest(best['checkpoint'])==best['sha256']
    assert digest(args['config'])==contract['config_sha256']
    assert digest(args['manifest'])==contract['manifest_sha256']
    cfg=load_yaml(args['config']);cfg['data_augment']=[]
    split='train' if a.stage in ['calibrate','gate-check'] else 'val'
    a.content_threshold=None;a.content_calibration_sha256=None
    if a.content_calibration:
        policy=json.loads(Path(a.content_calibration).read_text())
        assert policy['status']=='COMPLETED' and policy['split']=='train'
        assert policy['descriptor']==DESCRIPTOR_VERSION
        assert policy['checkpoint_sha256']==best['sha256'] and policy['contract']==contract
        assert a.periodic_last==policy['max_period'] and a.max_age==policy['max_age']
        a.content_threshold=float(policy['selected']['threshold'])
        assert np.isfinite(a.content_threshold) and a.content_threshold>=0
        a.content_calibration_sha256=digest(a.content_calibration)
    rows=json.loads(Path(args['manifest']).read_text())[split]
    torch.manual_seed(303);np.random.seed(303)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=PointPillarINT(cfg['model']['args'],a.mode).cuda().eval()
    model.load_state_dict(torch.load(best['checkpoint'],map_location='cpu')['model'],strict=True)
    return model,cfg,rows,dict(contract=contract,checkpoint=best,arguments=vars(a),split=split)


def engineering(a,model,cfg,rows,out):
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test',
                           '-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(result.stdout)
    if result.returncode:raise RuntimeError('INT unit tests failed; see unit_tests.log')
    _,stream=loader(rows[:8],cfg,2,303)
    wrapper=CacheRepairDetector(model,cfg,dense=True)
    original_state=cache_state=periodic_state=content_state=None;errors=[];crop_errors=[];periodic_errors=[];content_errors=[]
    periodic=PeriodicStageDetector(model,cfg,period=1)
    content=PeriodicStageDetector(model,cfg,period=1,content_threshold=0.)
    captured=[]
    hook=model.backbone.resnet.layer0.register_forward_pre_hook(lambda m,x:captured.append(x[0].detach()))
    with torch.no_grad():
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta']
            expected,original_state,_=model.step(ego,original_state,meta)
            actual,cache_state,_=wrapper.step(ego,cache_state,meta)
            periodic_pred,periodic_state,_=periodic.step(ego,periodic_state,meta)
            periodic_error=max(float((expected[k]-periodic_pred[k]).abs().max()) for k in expected)
            if periodic_error>1e-5:raise RuntimeError('Periodic full-refresh mismatch')
            periodic_errors.append(periodic_error)
            content_pred,content_state,_=content.step(ego,content_state,meta)
            content_error=max(float((expected[k]-content_pred[k]).abs().max()) for k in expected)
            if content_error>1e-5:raise RuntimeError('Content full-refresh mismatch')
            content_errors.append(content_error)
            error=max(float((expected[k]-actual[k]).abs().max()) for k in expected)
            if error>1e-5:raise RuntimeError('Dense integration mismatch %s'%error)
            errors.append(error)
        hook.remove()
        x=captured[-1]
        for i,n in enumerate(wrapper.layers):
            stage=getattr(model.backbone.resnet,'layer%d'%i)
            reference=stage(x);mask=torch.zeros_like(reference[:,:1],dtype=torch.bool)
            mask[...,0,0]=True;mask[...,-1,-1]=True
            mask[...,reference.shape[-2]//2,reference.shape[-1]//2]=True
            actual,written,_=refresh_tiles(stage,x,torch.zeros_like(reference),mask,2,4*n-1,8,allow_fallback=False)
            error=float((actual-reference).abs().masked_select(written.expand_as(actual)).max())
            crop_errors.append(error)
            if not torch.allclose(actual[written.expand_as(actual)],reference[written.expand_as(reference)],atol=1e-4,rtol=1e-4):
                raise RuntimeError('Real-weight crop integration mismatch %s'%error)
            x=reference
    return dict(status='PASSED',dense_prediction_max_errors=errors,real_weight_stage_crop_max_errors=crop_errors,
                periodic_full_refresh_max_errors=periodic_errors,
                content_full_refresh_max_errors=content_errors,
                tests='All test_int*.py; synthetic boundaries/phase/nonzero BN; eight real frames and three real-weight crop checks')


def evaluate(a,model,cfg,rows,out,wrapper=None):
    if a.content_calibration:
        check=json.loads(Path(a.content_check).read_text())
        assert check['status']=='PASSED' and check['content_calibration_sha256']==a.content_calibration_sha256
    if a.stage=='full':
        gate=json.loads(Path(a.engineering).read_text())
        assert gate['status']=='PASSED'
        if a.periodic_last:assert 'periodic_full_refresh_max_errors' in gate
        if a.content_calibration:assert 'content_full_refresh_max_errors' in gate
        provenance=json.loads((Path(a.engineering).parent/'provenance.json').read_text())
        assert provenance['checkpoint']['sha256']==digest(json.loads((Path(a.root)/('full_'+a.mode)/'best.json').read_text())['checkpoint'])
    chosen=rows if a.stage=='full' else rows[:a.frames]
    dataset,stream=loader(chosen,cfg,4,303)
    if wrapper is None:
        wrapper=CacheRepairDetector(model,cfg,a.fraction,a.tile,a.max_age,a.dense,a.occupancy_policy,a.oracle_stage0,a.support_profile,a.oracle_index)
    if a.periodic_last:wrapper=PeriodicStageDetector(model,cfg,a.periodic_last,a.max_age,a.content_threshold,a.zero_uncovered)
    state=None;records=[];times=[]
    ap={t:dict(tp=[],fp=[],gt=0,score=[]) for t in [.3,.5,.7]}
    started=time.monotonic()
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample,row in zip(stream,chosen):
            meta=sample['meta'];assert meta['frame']==row['frame'] and sample['ego'] is not None
            ego=to_device(sample['ego'],torch.device('cuda'))
            torch.cuda.synchronize();tick=time.perf_counter()
            prediction,state,info=wrapper.step(ego,state,meta)
            torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tick)
            assert all(torch.isfinite(v).all() for v in prediction.values())
            if len(records)>=10:times.append(elapsed)
            if meta['supervised']:
                data={'ego':ego}
                boxes,scores=dataset.post.post_process(data,{'ego':prediction})
                gt=dataset.post.generate_gt_bbx(data)
                for threshold in ap:eval_utils.caluclate_tp_fp(boxes,scores,gt,ap,threshold)
            entry=dict(frame=meta['frame'],segment=meta['segment'],supervised=meta['supervised'],ms=elapsed,**info)
            log.write(json.dumps(entry)+'\n');records.append(entry)
            if len(records)%200==0:print('EVAL',len(records),'/',len(chosen),flush=True)
    assert len(records)==len(chosen)
    stages=[]
    for i in range(3):
        stages.append({k:float(np.mean([r['cache_stages'][i][k] for r in records])) for k in records[0]['cache_stages'][i]})
    return dict(status='COMPLETED',scope='Fixed-weight inference intervention; full validation' if a.stage=='full' else 'Engineering prefix only; not a full validation result',
        scans=len(records),labels=sum(r['supervised'] for r in records),reset_frames=sum(bool(r['reset']) for r in records),
        metrics={'bev_ap%d'%int(t*100):eval_utils.calculate_ap(ap,t)[0] for t in ap},stage_means=stages,
        model_step_ms_p50=float(np.median(times)),model_step_ms_p95=float(np.percentile(times,95)),
        model_step_ms_mean=float(np.mean(times)),
        timing_scope='Synchronized model, including FM/warp/gate/cache/crops/decoder/head; excludes IO, voxelization, postprocess; instrumentation included',
        state_bytes=records[-1]['cache_bytes'],peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
        oracle_stage0=a.oracle_stage0,
        oracle_index=a.oracle_index if a.oracle_stage0 else None,
        support_profile=a.support_profile,
        periodic_last=a.periodic_last,
        content_calibration_sha256=a.content_calibration_sha256,content_threshold=a.content_threshold,
        zero_uncovered=a.zero_uncovered,
        uncovered_fraction_mean=float(np.mean([r['uncovered_fraction'] for r in records])) if a.zero_uncovered else None,
        periodic_refresh_counts={reason or 'reuse':sum(r.get('periodic_refresh_reason')==reason for r in records)
                                 for reason in ['initial','period','age','coverage','change',None]} if a.periodic_last else None,
        caveats=['Oracle, when enabled, caches only oracle_index and computes that full current stage solely to rank error; extra work is counted; not deployable or a formal optimal AP bound.',
                 'zero_uncovered, when enabled, is a diagnostic that leaves new-view cells zero; it is not learned boundary repair and may miss newly visible objects.',
                 'Ideal required convolution MACs, when profiled, exclude oracle, warp, selector, decoder, memory traffic and launch overhead; no sparse kernel or speedup is established.',
                 'Stage input-area ratios include repeated halo work, are not measured FLOPs.',
                 'Current FM and decoder/head remain dense; no dynamic-object motion prediction.',
                 'Upstream reused features can alter even refreshed downstream outputs.',
                 'Concurrent GPU experiments are not a controlled latency comparison.'])


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING'))
    model,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    if a.stage=='calibrate':
        from opencood.tools.int_cache_gate_calibration import calibrate
        report=calibrate(a,model,cfg,rows,out,provenance)
    elif a.stage=='gate-check':
        from opencood.tools.int_cache_gate_check import check_online_gate
        report=check_online_gate(a,model,cfg,rows,out)
    else:
        report=engineering(a,model,cfg,rows,out) if a.stage=='engineering' else evaluate(a,model,cfg,rows,out)
    write_json(out/'report.json',report);write_json(out/'status.json',dict(status='COMPLETED'))
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',default='spike_gru');p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--stage',choices=['engineering','pilot','full','calibrate','gate-check'],required=True)
    p.add_argument('--fraction',type=float,default=.5);p.add_argument('--tile',type=int,default=8)
    p.add_argument('--max-age',type=float,default=1.);p.add_argument('--frames',type=int,default=256)
    p.add_argument('--occupancy-policy',choices=['cell','tile'],default='cell')
    p.add_argument('--oracle-stage0',action='store_true')
    p.add_argument('--oracle-index',type=int,choices=[0,1,2],default=0,
                   help='Cached stage for the legacy --oracle-stage0 diagnostic flag; all other stages execute densely')
    p.add_argument('--support-profile',action='store_true')
    p.add_argument('--periodic-last',type=int,default=0,help='Positive period for whole last-stage refresh; 1 is paired dense control')
    p.add_argument('--content-calibration');p.add_argument('--content-check')
    p.add_argument('--zero-uncovered',action='store_true',help='Diagnostic periodic-only ablation: zero-fill uncovered cells instead of forcing a full refresh')
    p.add_argument('--dense',action='store_true');p.add_argument('--engineering')
    p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.oracle_index and not a.oracle_stage0:p.error('--oracle-index requires --oracle-stage0')
    if a.periodic_last<0:p.error('--periodic-last must be nonnegative')
    if a.periodic_last and (a.oracle_stage0 or a.support_profile or a.dense):
        p.error('Periodic mode cannot combine with oracle/profile/dense flags')
    if a.stage=='full' and not a.engineering:p.error('Full evaluation requires engineering report')
    if a.stage=='gate-check' and not a.content_calibration:p.error('gate-check requires content calibration')
    if a.content_calibration and not a.periodic_last:p.error('Content gate requires periodic guard')
    if a.zero_uncovered and (not a.periodic_last or a.content_calibration):p.error('Zero-fill diagnostic requires periodic-only mode')
    if a.content_calibration and a.stage in ['pilot','full'] and not a.content_check:p.error('Content evaluation requires gate-check report')
    if a.worker:
        try:worker(a)
        except Exception as exc:
            write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free,util=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split(',')
    if int(free)<12000 or int(util)>5:raise ValueError('GPU busy')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,'-m','opencood.tools.run_int_cache_repair']+sys.argv[1:]+['--worker']
    with (out/'run.log').open('x') as log:
        proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launch.json',dict(command=command,pid=proc.pid,gpu=a.gpu,created_unix=time.time(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
