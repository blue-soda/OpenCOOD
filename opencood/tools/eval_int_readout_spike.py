"""Frozen full validation of all matched readout membrane and precision arms."""
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
from opencood.models.sub_modules.int_readout_spike import ReadoutSpikeAnchor
from opencood.tools.pilot_int_readout_spike import TARGETS,evaluation_arms,evaluate
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.int_artifact_paths import artifact_path


def worker(a):
    out=Path(a.output);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout)
    if tests.returncode:raise RuntimeError('INT tests failed')
    pilot=Path(a.pilot);pr=json.loads((pilot/'report.json').read_text());pc=json.loads((pilot/'contract.json').read_text())
    pv=json.loads(Path(a.pilot_verification).read_text());weights=json.loads(Path(a.checkpoint_verification).read_text())
    assert pv['status']==weights['status']=='VERIFIED' and pr['status']=='COMPLETED'
    assert pv['source_sha256']['report.json']==weights['source_report_sha256']==digest(pilot/'report.json')
    for n in ('contract.json','preflight.json'):assert pv['source_sha256'][n]==digest(pilot/n)
    assert all(v['optimizer_steps']==71 for v in pr['training'].values()) and pc['adaptation_parameters']==1475072
    source=Path(a.full_reference);sr=json.loads((source/'report.json').read_text())
    sv=json.loads(Path(a.full_verification).read_text())
    assert sv['status']=='VERIFIED' and sr['status']=='COMPLETED'
    assert sr['torch_version']==torch.__version__, 'Replay baseline in this environment first'
    for n in ('report.json','contract.json','full_frames.jsonl','full_ap_records.json'):
        assert sv['source_sha256'][n]==digest(source/n)
    assert pr['checkpoint_sha256']==sr['checkpoint_sha256']==digest(Path(a.checkpoint))
    gold=[json.loads(line) for line in (source/'full_frames.jsonl').read_text().splitlines()]
    data=json.loads(Path(a.data_manifest).read_text())
    assert data['frames']==5794 and len(data['files'])==7532
    write_json(out/'status.json',dict(status='RUNNING',phase='validation_data_identity'))
    for row in data['files']:
        path=Path(artifact_path(row['path']))
        assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],str(path)
    torch.cuda.set_per_process_memory_fraction(.50)
    a.mode='gru';a.stage='full';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    assert len(rows)==len(gold)==5794 and sum(r['supervised'] for r in rows)==1738
    assert [r['frame'] for r in rows]==[r['frame'] for r in gold]
    assert sorted({r[k] for r in rows for k in ('pcd','label') if r[k]})==sorted(artifact_path(r['path']) for r in data['files'])
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'] and saved['contract']['version']==base.version
    base.load_state_dict(saved['model'],strict=True);strict=AnchoredLatePyramidDetector(base);models={}
    for mode in ('ann','lif','leaky'):
        path=pilot/(mode+'_readout.pth')
        assert digest(path)==pr['training'][mode]['checkpoint_sha256']==weights['checkpoints'][mode]['sha256']
        target=torch.load(path,map_location='cpu');assert target['contract']==pc and target['mode']==mode and target['updates']==71
        model=ReadoutSpikeAnchor(base,mode,pc['scale'],torch.float16).cuda().eval()
        assert set(target['targets'])==set(TARGETS) and torch.equal(target['scale'].reshape(-1),torch.tensor(pc['scale']))
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in TARGETS:p.copy_(target['targets'][n].to(p))
        models[mode]=model
    arms=evaluation_arms(strict,models);before={n:model_digest(m) for n,m in arms.items()}
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),pilot_report_sha256=digest(pilot/'report.json'),
        pilot_verification_sha256=digest(Path(a.pilot_verification)),checkpoint_verification_sha256=digest(Path(a.checkpoint_verification)),
        full_reference_report_sha256=digest(source/'report.json'),full_reference_verification_sha256=digest(Path(a.full_verification)),
        target_sha256={n:pr['training'][n]['checkpoint_sha256'] for n in models},data_manifest_sha256=digest(Path(a.data_manifest)),
        data_files_byte_verified=7532,split='val',seed=303,scans=5794,labels=1738,
        parameters=32756052,adaptation_parameters=1475072,trained=False,scale=pc['scale'],tau=.2,threshold=1.,
        storage='FP16 learned arms, fixed-weight FP32 inference controls',torch_version=torch.__version__,timing_valid=False,
        scope='Core Conv FLOPs count actual own strict refresh schedule. Shared-GPU instrumented model execution; no valid latency.')
    write_json(out/'contract.json',contract)
    result,ap=evaluate(arms,base,cfg,rows,out,'full',reference_rows=gold)
    assert json.loads((out/'full_ap_records.json').read_text())['strict']==json.loads((source/'full_ap_records.json').read_text())['strict']
    assert all(model_digest(m)==before[n] for n,m in arms.items())
    write_json(out/'report.json',dict(status='COMPLETED',checkpoint_sha256=contract['checkpoint_sha256'],evaluation=result,
        all_weights_bn_scale_unchanged=True,strict_raw_ap_replayed=True,strict_full_schedule_replayed=True,trained=False,timing_valid=False,
        split='val',torch_version=torch.__version__,boundary='Full frozen validation; fixed one-pass adaptation, no threshold/scale search. Validation used in research, not independent test. Dense GPU computation; no physical hardware energy claim.'))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','pilot','pilot-verification','checkpoint-verification','full-reference','full-verification','data-manifest','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min frozen full-val budget')));signal.alarm(7200)
        try:worker(a)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().strip())
    if free<14500:raise RuntimeError('Requires14.5GB free; process cap50%, no valid shared-GPU timing')
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.eval_int_readout_spike']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
