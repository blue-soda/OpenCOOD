"""Frozen joint quantized interface for ordinary-state adaptation; no real training."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_fine_integer import FINE_POLICY
from opencood.models.sub_modules.int_fine_readout_integer import FineReadoutIntegerDetector
from opencood.models.sub_modules.int_joint_state_qat import JointStateQATDetector,STATE_QAT_PREFIXES
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.int_artifact_paths import artifact_path
import copy,numpy as np
from opencood.models.sub_modules.int_joint_state_qat import STATE_QAT_PREFIXES,causal_update_indices
from opencood.tools.pilot_int_protected_state import schedule
from opencood.tools.pilot_int_gated_state import detach_state
from opencood.tools.pilot_int_causal_readout_qat import buffer_digest
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.utils import eval_utils


def frozen_digest(model):
    import hashlib
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        if not n.startswith(STATE_QAT_PREFIXES):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_one(model,cfg,rows,out,name,criterion,contract,preflight=False):
    before=model_digest(model);frozen=frozen_digest(model);buffers=buffer_digest(model)
    opened=[p for p in model.parameters() if p.requires_grad]
    assert all(p.requires_grad==n.startswith(STATE_QAT_PREFIXES) for n,p in model.named_parameters()) and sum(p.numel() for p in opened)==2449664
    optimizer=torch.optim.AdamW(opened,lr=1e-4,weight_decay=1e-4);optimizer.zero_grad(set_to_none=True)
    ends=window_ends(rows,4);state=None;losses=[];ordinary=False;updates=[];labels=0;loss_values=[];tag=name+('_preflight' if preflight else '_train')
    _,stream=loader(rows,cfg,2,303)
    with (out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));assert meta['frame']==rows[i]['frame']
            pred,state,info=model.step(ego,state,meta,detach_state=False);assert all(bool(torch.isfinite(v).all()) for v in pred.values())
            assert info['full_refresh']==contract['expected_refresh_reasons'][i]
            ordinary|=not bool(info['full_refresh']);row=dict(frame=meta['frame'],supervised=meta['supervised'],reset=info['reset'],refresh=info['full_refresh'],optimizer_step=False)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss));losses.append(loss);labels+=1;row['detection_loss']=float(loss.detach())
            if i in ends:
                if i in contract['expected_update_indices']:
                    assert losses and ordinary
                    objective=torch.stack(losses).mean();objective.backward()
                    for parameter_name,p in model.named_parameters():
                        if p.requires_grad:assert p.grad is not None and bool(torch.isfinite(p.grad).all()),parameter_name
                        else:assert p.grad is None,parameter_name
                    norm=torch.nn.utils.clip_grad_norm_(opened,10.);assert bool(torch.isfinite(norm)) and float(norm)>0
                    row.update(gradient_norm_before_clip=float(norm),window_loss=float(objective.detach()))
                    if preflight:
                        log.write(json.dumps(row)+'\n');assert model_digest(model)==before and frozen_digest(model)==frozen and buffer_digest(model)==buffers
                        model.zero_grad(set_to_none=True)
                        return dict(status='PASSED',first_update_frame=meta['frame'],scans=i+1,gradient_norm=float(norm),weights_unchanged=True,trainable_parameters=2449664)
                    optimizer.step();updates.append(meta['frame']);loss_values.append(float(objective.detach()));row['optimizer_step']=True
                    assert all(bool(torch.isfinite(p).all()) for p in opened)
                optimizer.zero_grad(set_to_none=True);losses=[];ordinary=False;state=detach_state(state)
                if not preflight and row['optimizer_step'] and len(updates)%64==0:
                    torch.save(dict(targets={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict(),own_state=detach_state(state,cpu=True),next_scan=i+1,updates=updates,contract=contract,mode=name,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state()),out/('%s_update_%03d.pth'%(name,len(updates))))
            log.write(json.dumps(row)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=len(rows),updates=len(updates)))
    assert not preflight and updates==contract['expected_update_frames'] and labels==contract['training_labels']
    assert frozen_digest(model)==frozen and buffer_digest(model)==buffers and model_digest(model)!=before
    path=out/(name+'_final.pth');torch.save(dict(targets={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict(),contract=contract,mode=name,updates=updates,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),resume='Fixed pass complete, continuation separately registered'),path)
    return dict(scans=len(rows),labels=labels,updates=len(updates),update_frames=updates,checkpoint_sha256=digest(path),trainable_parameters=2449664,frozen_weights_bn_unchanged=True,mean_window_loss=sum(loss_values)/len(loss_values))


def evaluate(models,base,cfg,rows,out,tag,initial=False):
    states={n:None for n in models};base_state=None;records=[];hashes={n:model_digest(m) for n,m in models.items()}
    ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models};dataset,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});predictions={}
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);assert all(bool(torch.isfinite(v).all()) for v in pred.values());predictions[n]=pred
                fine=m.fine_reference_flops() if isinstance(m,JointStateQATDetector) else 0;readout=getattr(m.core.readout,'last_reference_flops',0)
                row['arms'][n]=dict(core_fp32_conv_flops=fp32,fine_fp64_reference_flops=fine,readout_fp64_reference_flops=readout,logical_core_conv_flops=fp32+fine+readout,cache_bytes=info['cache_bytes'],reset=info['reset'],refresh=info['full_refresh'])
            if initial:
                for a0,b0 in (('frozen_ann','ann'),('frozen_qat','qat')):
                    assert all(torch.equal(predictions[a0][k],predictions[b0][k]) for k in predictions[a0])
                    assert all(torch.equal(states[a0][k],states[b0][k]) for k in base.state_keys)
            if meta['supervised']:
                batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch)
                for n,pred in predictions.items():
                    boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            records.append(row);log.write(json.dumps(row)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=len(rows)))
    assert len(records)==256 and sum(r['supervised'] for r in records)==192 and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/(tag+'_ap_records.json'),ap)
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),**{k+'_mean':sum(r['arms'][n][k] for r in records)/256 for k in ('core_fp32_conv_flops','fine_fp64_reference_flops','readout_fp64_reference_flops','logical_core_conv_flops','cache_bytes')},model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    return dict(scans=256,labels=192,metrics=metrics),ap


def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    torch.cuda.set_per_process_memory_fraction(.65)
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert (audit.get('status')=='VERIFIED' if not a.validation_prefix else audit['frames']==5794 and len(audit['files'])==7532);assert gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for item in audit['files']:assert digest(Path(artifact_path(item['path'])) if a.validation_prefix else Path(item['relocated_path']))==item['sha256']
    a.mode='gru';a.stage='full' if a.validation_prefix else 'calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    for p in source.parameters():p.requires_grad_(False)
    
    fv=json.loads(Path(a.fine_verification).read_text());assert fv['status']=='VERIFIED' and fv['run']=='R-200'
    assert digest(Path(a.calibration))==fv['source_sha256']['calibration.json'] and fv['checkpoint_sha256']==gate['checkpoint_sha256']
    pv=json.loads(Path(a.primitive_verification).read_text());assert pv['status']=='VERIFIED' and pv['run']=='R-204' and pv['tests']==446
    calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    assert audit['train_range']==[768,2816] and audit['evaluation_range']==[512,768] and not audit['missing']
    proof=json.loads(Path(a.fine_stream_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-205' and proof['calibration_sha256']==digest(Path(a.calibration))
    iv=json.loads(Path(a.implicit_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-201' and iv['checkpoint_sha256']==gate['checkpoint_sha256'] and iv['calibration_sha256']==digest(Path(a.readout_calibration))
    rc=json.loads(Path(a.readout_calibration).read_text());assert rc['relative_quantum']==.125 and rc['floor']==1e-4
    qv=json.loads(Path(a.interface_verification).read_text());qfull=json.loads(Path(a.quality_verification).read_text())
    assert qv['status']==qfull['status']=='VERIFIED' and qv['run']=='R-219' and qfull['run']=='R-212' and qv['checkpoint_sha256']==qfull['checkpoint_sha256']==gate['checkpoint_sha256']
    assert qv['calibration_sha256']==qfull['calibration_sha256']==digest(Path(a.calibration)) and qv['readout_calibration_sha256']==digest(Path(a.readout_calibration))
    prior=json.loads(Path(a.prior_learning_verification).read_text());assert prior['status']=='VERIFIED' and prior['run']=='R-185';goldpath=Path(a.prior_learning_source)/'initial_ap_records.json';assert digest(goldpath)==prior['source_sha256']['initial_ap_records.json'];gold=json.loads(goldpath.read_text())
    models=dict(frozen_ann=source,frozen_qat=JointStateQATDetector(source,calibration['layers'],rc['quanta'],True).cuda().eval(),ann=JointStateQATDetector(source,calibration['layers'],rc['quanta'],False).cuda().eval(),qat=JointStateQATDetector(source,calibration['layers'],rc['quanta'],True).cuda().eval())
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    for n in ('ann','qat'):models[n].open_ordinary_updates()
    training,evaluation=rows[768:2816],rows[512:768];assert audit['train_frames']==[r['frame'] for r in training] and audit['evaluation_frames']==[r['frame'] for r in evaluation]
    reasons,legacy_updates=schedule(training);ends=window_ends(training,4);indices=causal_update_indices(training,reasons,ends);updates=[training[i]['frame'] for i in indices];assert len(training)==2048 and len(evaluation)==256
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),calibration_sha256=digest(Path(a.calibration)),readout_calibration_sha256=digest(Path(a.readout_calibration)),interface_verification_sha256=digest(Path(a.interface_verification)),quality_verification_sha256=digest(Path(a.quality_verification)),prior_learning_verification_sha256=digest(Path(a.prior_learning_verification)),audit_sha256=digest(Path(a.audit)),train_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],training_labels=sum(r['supervised'] for r in training),expected_refresh_reasons=reasons,expected_update_frames=updates,expected_update_indices=indices,window_end_indices=sorted(ends),training_supervised=[bool(r['supervised']) for r in training],skipped_legacy_window_frames=[f for f in legacy_updates if f not in updates],seed=303,tbptt=4,epochs=1,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=10.,trainable_prefixes=STATE_QAT_PREFIXES,trainable_parameters=2449664,cuda_memory_fraction=.65,precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        loss='Matched labelled-window mean detection loss, update only if an ordinary frame occurs at or before a labelled frame within its TBPTT window, without intervening task reset. No teacher target/history, no event penalty. Same2048 already-registered training frames and update schedule; one additional adaptation pass, not unseen training data.',
        boundary='One quality GPU only. Frozen fine/readout/heads/deblocks/fine state/full-refresh/BN buffers; only ordinary coarse/deep update and light paths train. Fixed quanta and240min budget. Initial and final four-arm train256 diagnostics, not fullval/independent test. Real no-step gradient gate before any optimization. Every64-update recovery members kept, no best-epoch selection, no additional learning-rate search. QAT FP64/STE is training/reference, not actual event cost or independent SNN memory benefit.')
    write_json(out/'contract.json',contract);criterion=PointPillarDirLoss(cfg['loss']['args']).cuda()
    preflight={n:train_one(models[n],cfg,training,out,n,criterion,contract,True) for n in ('ann','qat')};write_json(out/'preflight.json',preflight)
    initial,raw0=evaluate(models,base,cfg,evaluation,out,'initial',True)
    serialized=json.loads((out/'initial_ap_records.json').read_text());assert serialized['frozen_ann']==serialized['ann']==gold['frozen'] and serialized['frozen_qat']==serialized['qat']
    learned={n:train_one(models[n],cfg,training,out,n,criterion,contract) for n in ('ann','qat')}
    final,raw1=evaluate(models,base,cfg,evaluation,out,'final')
    assert all(raw0[n]==raw1[n] for n in ('frozen_ann','frozen_qat'))
    write_json(out/'sort_orders.json',dict(source_sha256={tag:digest(out/(tag+'_ap_records.json')) for tag in ('initial','final')},orders={tag:{n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in raw.items()} for tag,raw in (('initial',raw0),('final',raw1))}))
    write_json(out/'checkpoint_manifest.json',dict(files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.glob('*.pth'))],resume_rehearsed=False))
    write_json(out/'report.json',dict(status='COMPLETED',initial=initial,final=final,training=learned,preflight=preflight,trained=True,timing_valid=False,frozen_raw_ap_exact=True,initial_pairs_exact=True,reference_ann_raw_ap_exact=True));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','primitive-verification','calibration','fine-stream-verification','implicit-verification','readout-calibration','interface-verification','quality-verification','prior-learning-verification','prior-learning-source','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('240min matched joint state learning')));signal.alarm(14400)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
