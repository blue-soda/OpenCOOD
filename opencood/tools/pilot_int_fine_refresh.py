"""Matched fine-cache adaptation using own states, frozen teacher targets only."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_fine_refresh import FineRefreshDetector
from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.pilot_int_cached_state import frozen_digest
from opencood.tools.pilot_int_gated_state import detach_state
from opencood.tools.pilot_int_protected_state import schedule
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.utils import eval_utils


def train_one(model,teacher,cfg,rows,out,name,criterion,contract,preflight=False):
    before=frozen_digest(model);initial=model_digest(model);teacher_hash=model_digest(teacher)
    opened=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(opened,lr=1e-4,weight_decay=1e-4);optimizer.zero_grad(set_to_none=True)
    state=teacher_state=None;losses=[];local=[];updates=[];values=[];labels=0;ordinary=0;ends=window_ends(rows,4);tag=name+('_preflight' if preflight else '_train');norms=[]
    _,stream=loader(rows,cfg,2,303)
    with (out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));assert meta['frame']==rows[i]['frame']
            with torch.no_grad():x,teacher_state=capture(teacher,ego,teacher_state,meta)
            pred,state,info=model.step_spatial(x,state,meta,detach_state=False)
            assert info['full_refresh']==contract['expected_refresh_reasons'][i]
            kd=(state['value']-teacher_state['value']).square().mean()/(teacher_state['value'].square().mean()+1e-6)
            assert bool(torch.isfinite(kd));local.append(kd);ordinary+=int(not bool(info['full_refresh']))
            row=dict(frame=meta['frame'],supervised=meta['supervised'],refresh=info['full_refresh'],fine_state_mse=float(kd.detach()),optimizer_step=False)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss));losses.append(loss);labels+=1;row['detection_loss']=float(loss.detach())
            if i in ends:
                if losses and ordinary:
                    objective=torch.stack(losses).mean()+torch.stack(local).mean();objective.backward()
                    for parameter_name,p in model.named_parameters():
                        if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),parameter_name
                        else:assert p.grad is None,parameter_name
                    norm=torch.nn.utils.clip_grad_norm_(opened,10.);assert bool(torch.isfinite(norm)) and float(norm)>0
                    row.update(window_loss=float(objective.detach()),gradient_norm_before_clip=float(norm),ordinary_frames=ordinary)
                    if preflight:
                        log.write(json.dumps(row)+'\n');assert model_digest(model)==initial and frozen_digest(model)==before and model_digest(teacher)==teacher_hash
                        model.zero_grad(set_to_none=True)
                        return dict(status='PASSED',first_update_frame=meta['frame'],gradient_norm=float(norm),weights_unchanged=True)
                    optimizer.step();updates.append(meta['frame']);values.append(float(objective.detach()));norms.append(float(norm));row['optimizer_step']=True
                    assert all(torch.isfinite(p).all() for p in opened)
                optimizer.zero_grad(set_to_none=True);losses=[];local=[];ordinary=0;state=detach_state(state)
                if not preflight and row['optimizer_step'] and len(updates)%16==0:
                    torch.save(dict(targets={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict(),own_state=detach_state(state,cpu=True),teacher_state=detach_state(teacher_state,cpu=True),next_scan=i+1,updates=updates,contract=contract,mode=name,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state()),out/('%s_update_%03d.pth'%(name,len(updates))))
            log.write(json.dumps(row)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=len(rows),updates=len(updates)))
    assert not preflight and updates==contract['expected_update_frames'] and labels==contract['training_labels']
    assert frozen_digest(model)==before and model_digest(teacher)==teacher_hash and model_digest(model)!=initial
    path=out/(name+'_final.pth');torch.save(dict(targets={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict(),contract=contract,mode=name,updates=updates,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),resume='Completed predetermined pass; any continuation separately registered'),path)
    return dict(scans=len(rows),labels=labels,updates=len(updates),update_frames=updates,checkpoint_sha256=digest(path),frozen_weights_bn_unchanged=True,teacher_unchanged=True,mean_window_loss=sum(values)/len(values),mean_gradient_norm=sum(norms)/len(norms))


def evaluate(models,base,cfg,rows,out,tag):
    states={n:None for n in models};base_state=None;records=[];hashes={n:model_digest(m) for n,m in models.items()};dataset,stream=loader(rows,cfg,2,303)
    ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    with torch.no_grad(),(out/(tag+'_frames.jsonl')).open('x',buffering=1) as f:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(torch.isfinite(v).all() for v in pred.values())
                entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],reset=info['reset'])
                if meta['supervised']:
                    batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch);boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            records.append(entry);f.write(json.dumps(entry)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=256))
    assert len(records)==256 and sum(r['supervised'] for r in records)==192 and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/(tag+'_ap_records.json'),ap)
    result=dict(scans=256,labels=192,metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/256,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()})
    return result,ap


def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text());fv=json.loads(Path(a.fine_verification).read_text())
    assert audit['status']==gate['status']==cv['status']==fv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145' and fv['run']=='R-158'
    assert audit['train_range']==[768,2816] and audit['evaluation_range']==[512,768] and not audit['missing']
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256']==fv['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256'] and digest(Path(a.fine_initialization))==fv['initialization_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval();models=dict(frozen=source)
    fine=torch.load(a.fine_initialization,map_location='cpu')
    for n,light in (('fine_dense',False),('fine_light',True)):
        torch.manual_seed(303);m=FineRefreshDetector(source,light).cuda().eval();m.core.fine_update.load_state_dict(fine['fine_update']);m.core.fine_light.load_state_dict(fine['fine_light']);models[n]=m
    assert model_digest(models['fine_dense'])==model_digest(models['fine_light'])
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    for n in ('fine_dense','fine_light'):
        prefixes=('core.fine_update.',)+(('core.fine_light.',) if n=='fine_light' else ())
        for name,p in models[n].named_parameters():p.requires_grad_(name.startswith(prefixes))
    training,evaluation=rows[768:2816],rows[512:768];assert audit['train_frames']==[r['frame'] for r in training] and audit['evaluation_frames']==[r['frame'] for r in evaluation]
    reasons,updates=schedule(training)
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),compact_verification_sha256=digest(Path(a.compact_verification)),fine_verification_sha256=digest(Path(a.fine_verification)),audit_sha256=digest(Path(a.audit)),trained_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),fine_initialization_sha256=digest(Path(a.fine_initialization)),train_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],training_labels=sum(r['supervised'] for r in training),expected_refresh_reasons=reasons,expected_update_frames=updates,seed=303,tbptt=4,epochs=1,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=10.,feature_loss_weight=1.,precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        loss='Window mean labelled detection + all-window normalized fine task-state MSE to frozen protected+rank128 teacher; update only labelled windows containing an ordinary frame. No history injection; own recurrent states through full refresh.',
        boundary='Both arms same2048/metadata-derived updates, only fine ordinary candidate/gate and light observation learn. Full-refresh/middle/deep/readout/head/BN frozen; dense unused fine_light counted in total parameters. Training scans reused from earlier parent adaptation, heldout train256 diagnostic only; no full-val/convergence or timing claim. StrictFP32.')
    write_json(out/'contract.json',contract);criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();names=('fine_dense','fine_light')
    preflight={n:train_one(models[n],source,cfg,training,out,n,criterion,contract,True) for n in names};write_json(out/'preflight.json',preflight)
    initial,raw0=evaluate(models,base,cfg,evaluation,out,'initial');learned={n:train_one(models[n],source,cfg,training,out,n,criterion,contract) for n in names};final,raw1=evaluate(models,base,cfg,evaluation,out,'final');assert raw0['frozen']==raw1['frozen']
    write_json(out/'sort_orders.json',dict(numpy_version=np.__version__,source_sha256={tag:digest(out/(tag+'_ap_records.json')) for tag in ('initial','final')},orders={tag:{n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in raw.items()} for tag,raw in (('initial',raw0),('final',raw1))}))
    write_json(out/'report.json',dict(status='COMPLETED',initial=initial,final=final,training=learned,preflight=preflight,frozen_raw_ap_exact=True,initial_weights_exact=True,trained=True,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','fine-initialization','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min matched fine-cache learning')));signal.alarm(7200)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
