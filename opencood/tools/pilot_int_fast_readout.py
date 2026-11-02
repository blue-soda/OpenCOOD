"""Matched dense and P4 readout adaptation; frozen upstream, own cache state."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector
from opencood.models.sub_modules.int_readout_refresh import ReadoutRefreshDetector
from opencood.tools.pilot_int_gated_state import evaluate,expected_updates,detach_state
from opencood.tools.pilot_int_cached_state import frozen_digest
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss


def train_one(model,teacher,cfg,rows,out,name,criterion,contract,preflight=False):
    before=frozen_digest(model);initial=model_digest(model);teacher_hash=model_digest(teacher)
    opened=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(opened,lr=1e-4,weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True);state=teacher_state=None;detection=[];local=[];updates=[];losses=[];labels=0
    ends=window_ends(rows,4);heavy=light=0;_,stream=loader(rows,cfg,2,303);tag=name+('_preflight' if preflight else '_train')
    with (out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));target={};assert meta['frame']==rows[i]['frame']
            hook=teacher.core.readout.register_forward_hook(lambda m,x,y:target.update(value=y))
            try:
                with torch.no_grad():x,teacher_state=capture(teacher,ego,teacher_state,meta)
            finally:hook.remove()
            pred,state,info=model.step_spatial(x,state,meta,detach_state=False)
            heavy+=int(info['readout_executed']);light+=int(not info['readout_executed'])
            kd=(state['readout']-target['value']).square().mean()/(target['value'].square().mean()+1e-6)
            assert bool(torch.isfinite(kd));local.append(kd)
            row=dict(frame=meta['frame'],supervised=meta['supervised'],full_refresh=info['full_refresh'],heavy_readout=info['readout_executed'],local_loss=float(kd.detach()),optimizer_step=False)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss));detection.append(loss);labels+=1;row['detection_loss']=float(loss.detach())
            if i in ends:
                if detection:
                    objective=torch.stack(detection).mean()+torch.stack(local).mean();objective.backward()
                    for n,p in model.named_parameters():
                        if not p.requires_grad:assert p.grad is None;continue
                        unused=(n.startswith('core.readout.') and heavy==0) or (n.startswith('core.fast_readout.') and light==0)
                        assert (unused and p.grad is None) or (p.grad is not None and bool(torch.isfinite(p.grad).all()))
                    norm=torch.nn.utils.clip_grad_norm_(opened,10.);assert bool(torch.isfinite(norm)) and float(norm)>0
                    row.update(gradient_norm_before_clip=float(norm),window_loss=float(objective.detach()),heavy_frames_in_window=heavy,light_frames_in_window=light)
                    if preflight:
                        log.write(json.dumps(row)+'\n');assert model_digest(model)==initial;model.zero_grad(set_to_none=True)
                        return dict(status='PASSED',first_update_frame=meta['frame'],gradient_norm=float(norm),weights_unchanged=True)
                    optimizer.step();updates.append(meta['frame']);losses.append(float(objective.detach()));row['optimizer_step']=True
                    assert all(bool(torch.isfinite(p).all()) for p in opened)
                optimizer.zero_grad(set_to_none=True);detection=[];local=[];heavy=light=0;state=detach_state(state)
                if not preflight and row['optimizer_step'] and len(updates)%16==0:
                    torch.save(dict(targets={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict(),
                        own_state=detach_state(state,cpu=True),teacher_state=detach_state(teacher_state,cpu=True),next_scan=i+1,updates=updates,
                        contract=contract,mode=name,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state()),out/('%s_update_%03d.pth'%(name,len(updates))))
            log.write(json.dumps(row)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=512,updates=len(updates)))
    assert not preflight and len(updates)==71 and updates==expected_updates(rows) and labels==232
    assert frozen_digest(model)==before and model_digest(teacher)==teacher_hash and model_digest(model)!=initial
    path=out/(name+'_final.pth');torch.save(dict(targets={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
        optimizer=optimizer.state_dict(),contract=contract,mode=name,updates=updates,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),resume='Fixed pass complete; continuation separately registered'),path)
    return dict(scans=512,labels=232,updates=71,update_frames=updates,checkpoint_sha256=digest(path),frozen_weights_bn_unchanged=True,teacher_unchanged=True,mean_window_loss=sum(losses)/len(losses))


def worker(a,out):
    write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    gate=json.loads(Path(a.gate_verification).read_text());assert gate['status']=='VERIFIED' and gate['run']=='R-132'
    audit=json.loads(Path(a.audit).read_text());assert audit['status']=='VERIFIED' and audit['frames']==768
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256']
    torch.cuda.set_per_process_memory_fraction(.6);a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model']);del teacher,saved
    for p in base.parameters():p.requires_grad_(False)
    models=dict(frozen=TemporalPyramidDetector(base,'original').cuda().eval())
    for n,mode in (('dense','dense'),('fast_p4','fast')):
        torch.manual_seed(303);m=ReadoutRefreshDetector(base,mode).cuda().eval()
        prefixes=('core.readout.',)+(('core.fast_readout.',) if mode=='fast' else ())
        for name,p in m.named_parameters():p.requires_grad_(name.startswith(prefixes))
        models[n]=m
    assert model_digest(models['dense'])==model_digest(models['fast_p4'])
    training,evaluation=rows[:512],rows[512:768]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),gate_verification_sha256=digest(Path(a.gate_verification)),audit_sha256=digest(Path(a.audit)),
        train_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],expected_update_frames=expected_updates(training),
        seed=303,tbptt=4,epochs=1,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=10.,feature_loss_weight=1.,
        loss='Mean supervised detection plus all-window normalized readout MSE. Context-only windows do not optimize.',
        schedule='Both train full readout; fast additionally trains ordinary-frame corrector. Full readout gradients only when executed or linked within TBPTT.',
        initialization='Same full state_dict; frozen original parent upstream; current projection seed303, correction final zero. Not composed with middle/deep light.',
        teacher='Frozen original parent readout is loss target only; never inject teacher readout into student history or refresh.',
        unused_dense_fast_parameters=sum(p.numel() for p in models['dense'].core.fast_readout.parameters()),
        boundary='Short matched train-prefix pilot; same 512 scans/71 updates, different active parameter subsets disclosed; no full-val/convergence claim.')
    write_json(out/'contract.json',contract);criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    preflight={n:train_one(models[n],base,cfg,training,out,n,criterion,contract,True) for n in ('dense','fast_p4')};write_json(out/'preflight.json',preflight)
    initial,initial_ap=evaluate(models,base,cfg,evaluation,out,'initial');assert initial_ap['dense']==initial_ap['frozen']
    learned={n:train_one(models[n],base,cfg,training,out,n,criterion,contract) for n in ('dense','fast_p4')}
    final,final_ap=evaluate(models,base,cfg,evaluation,out,'final');assert initial_ap['frozen']==final_ap['frozen']
    write_json(out/'report.json',dict(status='COMPLETED',initial=initial,final=final,preflight=preflight,training=learned,
        checkpoint_sha256=contract['checkpoint_sha256'],initial_weights_exact_across_trainable_arms=True,frozen_raw_ap_exact=True,trained=True,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','gate-verification','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.output)
    if a.worker:
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min readout learning pilot')));signal.alarm(7200)
        try:worker(a,out)
        except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
        return
    assert int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip())<200
    out.mkdir(parents=True,exist_ok=False);command=[sys.executable,'-m','opencood.tools.pilot_int_fast_readout']+sys.argv[1:]+['--worker']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(Path.cwd())+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (out/'run.log').open('x') as log:proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    write_json(out/'launch.json',dict(pid=proc.pid,gpu=a.gpu,command=command,created_unix=time.time(),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()))
    print(json.dumps(dict(pid=proc.pid,output=str(out))))


if __name__=='__main__':main()
