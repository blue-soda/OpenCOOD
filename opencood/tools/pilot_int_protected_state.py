"""Protected full-refresh functions and matched ordinary-only adaptation."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector,TemporalPyramidDetector,reset_reason
from opencood.tools.pilot_int_cached_state import train_one
from opencood.tools.pilot_int_gated_state import evaluate,expected_updates
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.data_utils.int_tbptt import window_ends
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import write_json
from opencood.tools.launch_int_memory import digest
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss


def schedule(rows):
    ends=window_ends(rows,4);state=None;reasons=[];updates=[];label=False;ordinary=False
    for i,row in enumerate(rows):
        reason,_=reset_reason(state,row);stamp=int(row['timestamp_us']);refresh=reason
        if refresh is None:
            if state['since_refresh']+1>=4:refresh='period'
            elif (stamp-state['refresh_timestamp_us'])/1e6>=.45-1e-6:refresh='age'
        state=dict(scene=row['segment'],timestamp_us=stamp,since_refresh=0 if refresh else state['since_refresh']+1,
            refresh_timestamp_us=stamp if refresh else state['refresh_timestamp_us'])
        reasons.append(refresh);label|=bool(row['supervised']);ordinary|=not bool(refresh)
        if i in ends:
            if label and ordinary:updates.append(row['frame'])
            label=ordinary=False
    return reasons,updates


def worker(a,out):
    write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    gate=json.loads(Path(a.gate_verification).read_text());prior=json.loads(Path(a.prior_verification).read_text())
    assert gate['status']==prior['status']=='VERIFIED' and gate['run']=='R-134' and prior['run']=='R-131'
    audit=json.loads(Path(a.audit).read_text());assert audit['status']=='VERIFIED'
    continuing=bool(a.extend_from)
    if continuing:
        assert a.resume_verification and audit['train_range']==[768,2816] and audit['evaluation_range']==[512,768] and not audit['missing']
        resume=json.loads(Path(a.resume_verification).read_text());assert resume['status']=='VERIFIED' and resume['run']=='R-137'
    else:assert audit['frames']==768 and a.resume_verification is None
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256']==prior['checkpoint_sha256']
    torch.cuda.set_per_process_memory_fraction(.6);a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model']);del teacher,saved
    for p in base.parameters():p.requires_grad_(False)
    models=dict(frozen=TemporalPyramidDetector(base,'original').cuda().eval())
    for n,light in (('protected_dense',False),('protected_light',True)):
        torch.manual_seed(303);m=ProtectedMultiscaleRefreshDetector(base,light).cuda().eval()
        prefixes=('core.coarse_update.','core.deep_update.')+(('core.coarse_light.','core.deep_light.') if light else ())
        for name,p in m.named_parameters():p.requires_grad_(name.startswith(prefixes))
        models[n]=m
    assert model_digest(models['protected_dense'])==model_digest(models['protected_light'])
    if continuing:
        assert resume['checkpoint_sha256']==gate['checkpoint_sha256']
        for n in ('protected_dense','protected_light'):
            load_targets(models[n],Path(a.extend_from)/(n+'_final.pth'),resume['training'][n]['checkpoint_sha256'])
    training,evaluation=(rows[768:2816] if continuing else rows[:512]),rows[512:768];reasons,updates=schedule(training)
    if continuing:assert audit['train_frames']==[r['frame'] for r in training] and audit['evaluation_frames']==[r['frame'] for r in evaluation]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),gate_verification_sha256=digest(Path(a.gate_verification)),prior_verification_sha256=digest(Path(a.prior_verification)),
        audit_sha256=digest(Path(a.audit)),train_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],
        expected_refresh_reasons=reasons,expected_update_frames=updates,skip_refresh_only_windows=True,
        skipped_refresh_only_update_frames=[f for f in expected_updates(training) if f not in updates],
        seed=303,tbptt=4,epochs=1,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=10.,feature_loss_weight=1.,
        loss='Supervised-window mean detection plus all-window normalized coarse/deep teacher-state MSE; update only if labelled and an ordinary frame occurs.',
        teacher='Targets only; refresh executes frozen original functions using student own history, never resets to teacher state.',
        initialization='Fresh parent, identical initial state_dict, not R130-trained weights. Both use protected original refresh and gated ordinary updates; dense control executes heavy observations on ordinary frames.',
        unused_dense_light_parameters=168448,protected_refresh_parameters=sum(p.numel() for p in models['protected_light'].core.refresh_updates.parameters()),
        boundary='New matched budget predetermined from frame metadata; pure-refresh supervised windows have no trainable functions and are skipped for BOTH arms. Not comparable as equal71-updates to R130. One fixed pass, no full-val/convergence claim.')
    contract['training_labels']=sum(r['supervised'] for r in training)
    if continuing:
        contract.update(resume_verification_sha256=digest(Path(a.resume_verification)),
            resume_checkpoints={n:str(Path(a.extend_from)/(n+'_final.pth')) for n in ('protected_dense','protected_light')},
            resume_checkpoint_sha256={n:resume['training'][n]['checkpoint_sha256'] for n in ('protected_dense','protected_light')},
            prior_update_counts={n:resume['training'][n]['updates'] for n in ('protected_dense','protected_light')},
            initialization='Matched continuation from each R137 arm: weights, AdamW moments/step counters and RNG restored; new stream starts with empty task state at prescribed segment. No teacher history injection.',
            boundary='Fixed next2048 new training scans, one pass, excludes train256 diagnostic frames. Same continuation budget for both arms; no full-val or convergence claim.')
    write_json(out/'contract.json',contract);criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    preflight={n:train_one(models[n],base,cfg,training,out,n,criterion,contract,True) for n in ('protected_dense','protected_light')};write_json(out/'preflight.json',preflight)
    initial,initial_ap=evaluate(models,base,cfg,evaluation,out,'initial')
    learned={n:train_one(models[n],base,cfg,training,out,n,criterion,contract) for n in ('protected_dense','protected_light')}
    final,final_ap=evaluate(models,base,cfg,evaluation,out,'final');assert initial_ap['frozen']==final_ap['frozen']
    write_json(out/'report.json',dict(status='COMPLETED',initial=initial,final=final,training=learned,preflight=preflight,
        checkpoint_sha256=contract['checkpoint_sha256'],initial_weights_exact_across_trainable_arms=not continuing,matched_continuation=continuing,frozen_raw_ap_exact=True,trained=True,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','gate-verification','prior-verification','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--extend-from');p.add_argument('--resume-verification')
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min protected learning')));signal.alarm(7200)
        try:worker(a,out)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    assert int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip())<200
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.pilot_int_protected_state']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
