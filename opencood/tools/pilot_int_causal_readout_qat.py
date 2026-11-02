"""Matched single-coarse-state learning pilot; one GPU, fixed disjoint prefixes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector
from opencood.data_utils.int_tbptt import window_ends
from opencood.loss.point_pillar_dir_loss import PointPillarDirLoss
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.utils import eval_utils


PREFIX='core.readout.'


def frozen_digest(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        if not n.startswith(PREFIX):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def detach_state(state,cpu=False):
    return {k:(v.detach().cpu() if cpu else v.detach()) if torch.is_tensor(v) else v for k,v in state.items()}


def expected_updates(rows):
    ends=window_ends(rows,4);has_label=False;result=[]
    for i,r in enumerate(rows):
        has_label|=bool(r['supervised'])
        if i in ends:
            if has_label:result.append(r['frame'])
            has_label=False
    return result


def train_one(model,cfg,rows,out,name,criterion,contract,preflight=False):
    original_frozen=frozen_digest(model);before=model_digest(model);before_buffers=buffer_digest(model)
    opened=[p for p in model.parameters() if p.requires_grad]
    assert all(p.requires_grad==n.startswith(PREFIX) for n,p in model.named_parameters())
    optimizer=torch.optim.AdamW(opened,lr=1e-5,weight_decay=1e-4);optimizer.zero_grad(set_to_none=True)
    ends=window_ends(rows,4);state=None;losses=[];updates=[];loss_values=[];labels=0
    _,stream=loader(rows,cfg,2,303);tag=name+('_preflight' if preflight else '_train')
    with (out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'))
            assert meta['frame']==rows[i]['frame']
            pred,state,info=model.step(ego,state,meta,detach_state=False)
            assert all(bool(torch.isfinite(v).all()) for v in pred.values())
            row=dict(frame=meta['frame'],supervised=meta['supervised'],reset=info['reset'],optimizer_step=False)
            if meta['supervised']:
                loss=criterion(pred,ego['label_dict']);assert bool(torch.isfinite(loss))
                losses.append(loss);labels+=1;row['detection_loss']=float(loss.detach())
            if i in ends:
                if losses:
                    objective=torch.stack(losses).mean();objective.backward()
                    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(PREFIX))
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in opened)
                    norm=torch.nn.utils.clip_grad_norm_(opened,10.);assert bool(torch.isfinite(norm)) and float(norm)>0
                    row.update(gradient_norm_before_clip=float(norm),window_loss=float(objective.detach()))
                    if preflight:
                        log.write(json.dumps(row)+'\n');assert model_digest(model)==before
                        model.zero_grad(set_to_none=True)
                        return dict(status='PASSED',first_update_frame=meta['frame'],scans=i+1,gradient_norm=float(norm),weights_unchanged=True)
                    optimizer.step();updates.append(meta['frame']);loss_values.append(float(objective.detach()));row['optimizer_step']=True
                    assert all(bool(torch.isfinite(p).all()) for p in opened)
                optimizer.zero_grad(set_to_none=True);losses=[];state=detach_state(state)
                if not preflight and row['optimizer_step'] and len(updates)%16==0:
                    path=out/('%s_update_%03d.pth'%(name,len(updates)))
                    torch.save(dict(readout=model.core.readout.state_dict(),optimizer=optimizer.state_dict(),
                        own_state=detach_state(state,cpu=True),next_scan=i+1,updates=updates,contract=contract,
                        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),mode=name),path)
            log.write(json.dumps(row)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=len(rows),updates=len(updates)))
    assert not preflight and len(updates)==71 and updates==expected_updates(rows) and labels==232
    assert frozen_digest(model)==original_frozen and model_digest(model)!=before and buffer_digest(model)==before_buffers
    path=out/(name+'_readout_final.pth')
    torch.save(dict(readout=model.core.readout.state_dict(),optimizer=optimizer.state_dict(),contract=contract,
        mode=name,updates=updates,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),
        resume='Fixed pass complete; any continuation separately registered, start from empty stream state'),path)
    return dict(scans=512,labels=labels,updates=len(updates),update_frames=updates,frozen_weights_bn_unchanged=True,
        checkpoint_sha256=digest(path),all_buffers_unchanged=True,trainable_parameters=sum(p.numel() for p in opened),mean_window_loss=sum(loss_values)/len(loss_values))


def buffer_digest(model):
    h=hashlib.sha256()
    for n,v in model.named_buffers():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

import copy
import numpy as np
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.models.sub_modules.int_causal_readout_qat import CausalReadoutQAT
from opencood.tools.diagnose_int_cached_state import load_targets

def evaluate(models,base,cfg,rows,out,tag,identity=False):
    states={n:None for n in models};base_state=None;records=[]
    ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    hashes={n:model_digest(m) for n,m in models.items()};dataset,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/(tag+'_frames.jsonl')).open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta)
            entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});predictions={}
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta)
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                fp64=getattr(m.core.readout,'last_reference_flops',0);predictions[n]=pred;entry['arms'][n]=dict(core_conv_flops=cost+fp64,core_fp32_conv_flops=cost,readout_fp64_reference_flops=fp64,cache_bytes=info['cache_bytes'],reset=info['reset'])
            if identity:assert all(torch.equal(v,predictions['ann'][k]) for k,v in predictions['frozen'].items())
            assert all(torch.equal(states[n][k],states['frozen'][k]) for n in models for k in base.state_keys)
            if meta['supervised']:
                batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch)
                for n,pred in predictions.items():
                    boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase=tag,scans=i+1,total=len(rows)))
    assert len(records)==256 and sum(r['supervised'] for r in records)==192
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/(tag+'_ap_records.json'),ap)
    return dict(scans=256,labels=192,metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),
        trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
        core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/256,core_fp32_conv_flops_mean=sum(r['arms'][n]['core_fp32_conv_flops'] for r in records)/256,readout_fp64_reference_flops_mean=sum(r['arms'][n]['readout_fp64_reference_flops'] for r in records)/256,cache_bytes=records[0]['arms'][n]['cache_bytes'],
        model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}),ap


def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert (audit.get('status')=='VERIFIED' if not a.full else audit['frames']==5794 and len(audit['files'])==7532);assert gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for r in audit['files']:
        path=Path(artifact_path(r['path'])) if a.full else Path(r['relocated_path']);assert digest(path)==r['sha256']
    a.mode='gru';a.stage='full' if a.full else 'calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    ev=json.loads(Path(a.event_verification).read_text());rv=json.loads(Path(a.reference_verification).read_text());assert ev['status']==rv['status']=='VERIFIED' and ev['run']=='R-172' and rv['run']==('R-148' if a.full else 'R-168')
    event_contract=Path(a.calibration).parent/'contract.json';assert digest(event_contract)==ev['source_sha256']['contract.json'];ec=json.loads(event_contract.read_text());assert ec['verification_sha256']==digest(Path(a.verification)) and ec['initialization_sha256']==digest(Path(a.initialization)) and ev['checkpoint_sha256']==gate['checkpoint_sha256']
    assert ev['calibration_sha256']==digest(Path(a.calibration));calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    iv=json.loads(Path(a.integer_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-181'
    icp=Path(a.integer_source)/'contract.json';assert digest(icp)==iv['source_sha256']['contract.json'];ic=json.loads(icp.read_text());assert ic['checkpoint_sha256']==gate['checkpoint_sha256'] and ic['calibration_sha256']==ev['calibration_sha256']
    qv=json.loads(Path(a.qat_verification).read_text());assert qv['status']=='VERIFIED' and qv['run']=='R-183';qcp=Path(a.qat_source)/'contract.json';assert digest(qcp)==qv['source_sha256']['contract.json']
    qc=json.loads(qcp.read_text());assert qc['calibration_sha256']==ev['calibration_sha256'] and qc['checkpoint_sha256']==gate['checkpoint_sha256']
    quality=json.loads(Path(a.quality_verification).read_text());assert quality['status']=='VERIFIED' and quality['run']=='R-182';rp=Path(a.quality_source)/'ap_records.json';assert digest(rp)==quality['source_sha256']['ap_records.json'];gold=json.loads(rp.read_text())
    models=dict(frozen=source)
    for name,enabled in (('ann',False),('qat',True)):
        model=copy.deepcopy(source);model.core.readout=CausalReadoutQAT(source.core.readout,calibration['quanta'],enabled);models[name]=model.cuda().eval()
    for n,m in models.items():
        for name,p in m.named_parameters():p.requires_grad_(n!='frozen' and name.startswith(PREFIX))
    training,evaluation=rows[:512],rows[512:768];assert len(expected_updates(training))==71 and sum(r['supervised'] for r in training)==232
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),initialization_sha256=digest(Path(a.initialization)),qat_verification_sha256=digest(Path(a.qat_verification)),quality_verification_sha256=digest(Path(a.quality_verification)),calibration_sha256=digest(Path(a.calibration)),protected_trained_sha256=gate['training']['protected_light']['checkpoint_sha256'],train_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],expected_update_frames=expected_updates(training),seed=303,tbptt=4,epochs=1,optimizer='AdamW',lr=1e-5,weight_decay=1e-4,clip_norm=10.,precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),trainable_prefix=PREFIX,
        loss='Same mean labelled-frame detection loss in4-frame update windows. Frozen upstream/head/BN statistics and fixed activation quanta. ANN versus QAT readout-only parameters, no teacher history or teacher loss.',
        boundary='Fixed512/232 labels/71 updates each, lower predeclared1e-5 learning rate after prior R1471e-4 readout pilot degraded; not an LR search. QAT uses exact integer-grid/int16 forward with STE derivatives and per-output scale detached from gradients. All initial/final arms retained; no best-epoch selection. Eval train256 not fullval or independent test; no valid timing. No assertion of sparse deployment speed; no-step real gradient gate precedes optimization.')
    write_json(out/'contract.json',contract);criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();names=('ann','qat')
    preflight={n:train_one(models[n],cfg,training,out,n,criterion,contract,True) for n in names};write_json(out/'preflight.json',preflight)
    initial,raw0=evaluate(models,base,cfg,evaluation,out,'initial',True);assert raw0['frozen']==raw0['ann']==gold['control'] and raw0['qat']==gold['integer_dense']
    learned={n:train_one(models[n],cfg,training,out,n,criterion,contract) for n in names};final,raw1=evaluate(models,base,cfg,evaluation,out,'final');assert raw0['frozen']==raw1['frozen']
    write_json(out/'checkpoint_manifest.json',dict(files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(out.glob('*.pth'))],resume_rehearsed=False))
    write_json(out/'sort_orders.json',dict(numpy_version=np.__version__,source_sha256={tag:digest(out/(tag+'_ap_records.json')) for tag in ('initial','final')},orders={tag:{n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in raw.items()} for tag,raw in (('initial',raw0),('final',raw1))}))
    write_json(out/'report.json',dict(status='COMPLETED',initial=initial,final=final,training=learned,preflight=preflight,trained=True,timing_valid=False,frozen_raw_ap_exact=True,initial_raw_ap_exact=True));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','event-verification','calibration','reference-verification','integer-verification','integer-source','qat-verification','qat-source','quality-verification','quality-source','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.full=False;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min matched fixed512 ANN/QAT readout adaptation')));signal.alarm(7200)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
