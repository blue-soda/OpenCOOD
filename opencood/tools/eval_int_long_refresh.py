"""Preregistered ten-arm frozen train-prefix diagnostic of anchor lifetime."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_pyramid_anchor import AnchoredLatePyramidDetector
from opencood.models.sub_modules.int_pyramid_anchor_patch import PatchedLatePyramidDetector
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_artifact_paths import artifact_path
from opencood.utils import eval_utils


CONFIGS=((2,.25),(3,.25),(4,.25),(4,.45),(6,.65))


def make_models(base,full=False):
    models={}
    for period,age in (((2,.25),(3,.25),(4,.45)) if full else CONFIGS):
        for kind in ('strict','patch'):
            name='p%d_a%d_%s'%(period,round(age*100),kind)
            cls=AnchoredLatePyramidDetector if kind=='strict' else PatchedLatePyramidDetector
            models[name]=cls(base,period=period,max_age=age)
    for p in base.parameters():p.requires_grad_(False)
    return models


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    full=a.split=='val';total=5794 if full else 256;labels=1738 if full else 192
    frame_file='full_frames.jsonl' if full else 'initial_frames.jsonl'
    ap_file='full_ap_records.json' if full else 'initial_ap_records.json'
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    reference=Path(a.reference);rv=json.loads(Path(a.reference_verification).read_text())
    assert rv['status']=='VERIFIED'
    for n in ('report.json','contract.json',frame_file,ap_file):
        assert digest(reference/n)==rv['source_sha256'][n]
    rc=json.loads((reference/'contract.json').read_text())
    gold=[json.loads(x) for x in (reference/frame_file).read_text().splitlines()]
    assert rc['checkpoint_sha256']==digest(Path(a.checkpoint))
    data=json.loads(Path(a.data_audit).read_text())
    write_json(out/'status.json',dict(status='RUNNING',phase='data_identity'))
    if full:
        assert data['frames']==5794 and len(data['files'])==7532
        assert a.prefix and a.prefix_verification
        prefix=Path(a.prefix);pv=json.loads(Path(a.prefix_verification).read_text())
        assert pv['status']=='VERIFIED' and pv['run']=='R-122'
        for n in ('report.json','contract.json','frames.jsonl','ap_records.json','unit_tests.log'):
            assert digest(prefix/n)==pv['source_sha256'][n]
        assert json.loads((prefix/'contract.json').read_text())['checkpoint_sha256']==rc['checkpoint_sha256']
        assert json.loads((reference/'report.json').read_text())['torch_version']==torch.__version__
        for row in data['files']:
            path=Path(artifact_path(row['path']));assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
    else:
        assert data['status']=='VERIFIED' and data['frames']==768
        for row in data['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    torch.cuda.set_per_process_memory_fraction(.50)
    a.mode='gru';a.stage='full' if full else 'calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    selected=rows if full else rows[512:768]
    assert [r['frame'] for r in selected]==([r['frame'] for r in gold] if full else rc['evaluation_frames'])
    assert sorted({r[k] for r in (rows if full else rows[:768]) for k in ('pcd','label') if r[k]})==sorted(artifact_path(r['path']) if full else r['relocated_path'] for r in data['files'])
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256']
    assert saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True);models=make_models(base,full)
    period1=AnchoredLatePyramidDetector(base,period=1)
    references={n:PatchedLatePyramidDetector(base,period=m.period,max_age=m.max_age,dense_reference=True)
                for n,m in models.items() if n.endswith('patch')}
    hashes={n:model_digest(m) for n,m in models.items()}
    assert all(sum(p.numel() for p in m.parameters())==32756052 for m in models.values())
    write_json(out/'contract.json',dict(checkpoint_sha256=digest(Path(a.checkpoint)),
        reference_verification_sha256=digest(Path(a.reference_verification)),reference_sha256=digest(reference/'contract.json'),
        data_audit_sha256=digest(Path(a.data_audit)),evaluation_frames=[r['frame'] for r in selected],
        configurations={n:dict(period=m.period,max_age=m.max_age,coverage=m.coverage) for n,m in models.items()},
        split=a.split,seed=303,scans=total,labels=labels,trained=False,timing_valid=False,adjacent_gap_reset_seconds=.25,
        prefix_verification_sha256=digest(Path(a.prefix_verification)) if full else None,
        cost_scope='Executed core Conv2d/ConvTranspose2d FLOPs=2*MAC, includes repeated patch halos; excludes pillar/nonconv/warp/packing.',
        boundary='Fixed complete validation, not independent test' if full else 'Fixed train-prefix diagnostic, not full validation; no AP selection of periods, ages or thresholds.'))
    states={n:None for n in models};reference_states={n:None for n in references}
    base_state=period1_state=None;records=[]
    ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    dataset,stream=loader(selected,cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));captured=[]
            hook=base.core.observation.register_forward_pre_hook(lambda m,x:captured.append(x[0]))
            try:expected,base_state,_=base.step(ego,base_state,meta)
            finally:hook.remove()
            x=captured[0]
            if i<32:
                p1,period1_state,_=period1.step_spatial(x,period1_state,meta)
                assert all(torch.equal(p1[k],expected[k]) for k in p1)
            predictions={};entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            assert entry['frame']==gold[i]['frame'] and entry['supervised']==gold[i]['supervised']
            for name,model in models.items():
                previous=states[name]
                (pred,states[name],info),cost=measured_step(model,x,previous,meta)
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                if info['late_refresh'] is None:
                    assert states[name]['anchor']['value'] is previous['anchor']['value']
                    assert states[name]['anchor']['timestamp_us']==previous['anchor']['timestamp_us']
                if i<32 and name in references:
                    rp,reference_states[name],ri=references[name].step_spatial(x,reference_states[name],meta)
                    assert all(torch.allclose(pred[k],rp[k],atol=1e-4,rtol=1e-4) for k in pred)
                    assert info['late_refresh']==ri['late_refresh']
                entry['arms'][name]=dict(core_conv_flops=cost,**info)
                if name=='p2_a25_strict':
                    prior=gold[i]['arms']['strict']
                    assert cost==prior['core_conv_flops'] and info['late_refresh']==prior['refresh'] and info['cache_bytes']==prior['cache_bytes']
                predictions[name]=pred
            if meta['supervised']:
                batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch)
                for name,pred in predictions.items():
                    boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for threshold in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[name],threshold)
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if (i+1)%16==0:write_json(out/'status.json',dict(status='RUNNING',phase='full' if full else 'prefix',scans=i+1,total=total))
    assert len(records)==total and sum(r['supervised'] for r in records)==labels
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'ap_records.json',ap)
    assert json.loads((out/'ap_records.json').read_text())['p2_a25_strict']==json.loads((reference/ap_file).read_text())['strict']
    metrics={}
    for n in models:
        entries=[r['arms'][n] for r in records]
        patches=[r['repair'] for r in entries if r['repair'] and r['repair']['active']]
        metrics[n]=dict(parameters=32756052,core_conv_flops_mean=sum(r['core_conv_flops'] for r in entries)/len(entries),
            model_step_ms_mean=None,timing_invalid_reason='Instrumented multi-arm diagnostic; no valid latency',
            ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0],
            cache_bytes=entries[0]['cache_bytes'],refresh_counts=dict(Counter(r['late_refresh'] or 'reuse' for r in entries)),
            patch_frames=len(patches),fallback_frames=sum(r['fallback'] for r in patches),
            mean_patch_write_fraction=sum(r['actual_write_fraction'] for r in patches)/max(1,len(patches)),
            mean_patch_stage_conv_ratio=sum(r['stage_conv_ratio'] for r in patches)/max(1,len(patches)),
            max_anchor_age=max(r['anchor_age_before_refresh'] for r in entries))
    write_json(out/'report.json',dict(status='COMPLETED',split=a.split,scans=total,labels=labels,metrics=metrics,
        checkpoint_sha256=digest(Path(a.checkpoint)),all_weights_bn_unchanged=True,strict_raw_ap_replayed=True,
        period1_first32_exact=True,patch_same_mask_first32_allclose=True,anchor_identity_on_reuse=True,
        trained=False,timing_valid=False,torch_version=torch.__version__))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','reference','reference-verification','data-audit','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--split',choices=('train','val'),default='train');p.add_argument('--prefix');p.add_argument('--prefix-verification')
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Long-refresh hard budget')));signal.alarm(7200 if a.split=='val' else 3600)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<14500:raise RuntimeError('Requires14.5GB free; cap50%')
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','opencood.tools.eval_int_long_refresh']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
