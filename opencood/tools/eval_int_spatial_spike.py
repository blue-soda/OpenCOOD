"""Frozen six-arm full validation of the completed spatial activation pilot."""
import argparse
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
from opencood.models.sub_modules.int_spatial_spike import SpatialSpikeAnchor
from opencood.tools.pilot_int_spatial_spike import evaluate
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import write_json
from opencood.tools.launch_int_memory import digest


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    pilot=Path(a.pilot);prior=json.loads((pilot/'report.json').read_text());pc=json.loads((pilot/'contract.json').read_text())
    source=Path(a.strict_reference);reference=json.loads((source/'report.json').read_text());gold=[json.loads(x) for x in (source/'frames.jsonl').read_text().splitlines()]
    assert prior['status']=='COMPLETED' and prior['initial_ann_exact'] and prior['strict_raw_ap_replayed']
    assert reference['status']=='COMPLETED' and len(gold)==reference['scans']==5794 and reference['labels']==1738
    assert prior['checkpoint_sha256']==reference['checkpoint_sha256']==digest(Path(a.checkpoint))
    torch.cuda.set_per_process_memory_fraction(.65)
    a.mode='gru';a.stage='full';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    assert len(rows)==5794 and [r['frame'] for r in rows]==[r['frame'] for r in gold]
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model'],strict=True);strict=AnchoredLatePyramidDetector(base);models={};source_shas={}
    for mode in ('ann','lif','leaky'):
        path=pilot/(mode+'_target.pth');source_shas[mode]=digest(path);assert source_shas[mode]==prior['training'][mode]['checkpoint_sha256']
        restored=torch.load(str(path),map_location='cpu');assert restored['contract']==pc and restored['mode']==mode and restored['updates']==71
        model=SpatialSpikeAnchor(base,mode,pc['scale']).cuda().eval()
        with torch.no_grad():
            model.core.coarse_encoder[3].conv2.weight.copy_(restored['target_weight'])
            assert torch.equal(model.activation_memory.scale.cpu(),restored['scale'])
        models[mode]=model
    arms=dict(strict=strict,**models);arms.update(lif_reset=models['lif'],leaky_reset=models['leaky'])
    for model in arms.values():
        for p in model.parameters():p.requires_grad_(False)
    before={n:model_digest(m) for n,m in arms.items()}
    contract=dict(checkpoint_sha256=prior['checkpoint_sha256'],pilot_report_sha256=digest(pilot/'report.json'),target_checkpoint_sha256=source_shas,
        strict_reference_report_sha256=digest(source/'report.json'),strict_reference_frames_sha256=digest(source/'frames.jsonl'),
        calibration_sha256=pc['calibration_sha256'],split='val',scans=5794,labels=1738,seed=303,trained=False,timing_valid=False,
        frozen_all_parameters=True,selection='All three fixed one-pass weights and both auxiliary membrane resets; no tuning/epoch extension',
        budget_minutes=90,boundary='Full validation of a validation-selected parent, not independent test or multi-seed significance. Actual Conv dense; ideal target operations exclude hardware overhead.')
    write_json(out/'contract.json',contract)
    result,_=evaluate(arms,base,cfg,rows,out,'full',reference_rows=gold)
    assert result['scans']==5794 and result['labels']==1738
    for metric,key in [('ap50','bev_ap50'),('ap70','bev_ap70')]:assert abs(result['metrics']['strict'][metric]-reference['metrics'][key])<1e-6
    assert result['metrics']['strict']['core_conv_flops_mean']==reference['core_conv_flops_mean']
    assert all(model_digest(m)==before[n] for n,m in arms.items())
    assert all(digest(pilot/(n+'_target.pth'))==sha for n,sha in source_shas.items())
    write_json(out/'report.json',dict(status='COMPLETED',split='val',checkpoint_sha256=contract['checkpoint_sha256'],evaluation=result,
        strict_full_reference_verified=True,all_weights_bn_scale_unchanged=True,trained=False,timing_valid=False,boundary=contract['boundary']))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','pilot','strict-reference','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90min full-validation budget')));signal.alarm(5400)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<20000:raise RuntimeError('Requires20GB free on one idle GPU')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.eval_int_spatial_spike']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd()),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
