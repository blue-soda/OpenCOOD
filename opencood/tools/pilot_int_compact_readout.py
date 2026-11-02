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
    original_frozen=frozen_digest(model);before=model_digest(model)
    opened=[p for p in model.parameters() if p.requires_grad]
    assert all(p.requires_grad==n.startswith(PREFIX) for n,p in model.named_parameters())
    optimizer=torch.optim.AdamW(opened,lr=1e-4,weight_decay=1e-4);optimizer.zero_grad(set_to_none=True)
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
    assert frozen_digest(model)==original_frozen and model_digest(model)!=before
    path=out/(name+'_readout_final.pth')
    torch.save(dict(readout=model.core.readout.state_dict(),optimizer=optimizer.state_dict(),contract=contract,
        mode=name,updates=updates,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),
        resume='Fixed pass complete; any continuation separately registered, start from empty stream state'),path)
    return dict(scans=512,labels=labels,updates=len(updates),update_frames=updates,frozen_weights_bn_unchanged=True,
        checkpoint_sha256=digest(path),mean_window_loss=sum(loss_values)/len(loss_values))



from opencood.models.sub_modules.int_factorized_readout import compact_readout
from opencood.tools.pilot_int_gated_state import evaluate
import numpy as np

def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());v=json.loads(Path(a.verification).read_text());assert audit['status']==v['status']=='VERIFIED' and v['run']=='R-145'
    assert digest(Path(a.checkpoint))==v['checkpoint_sha256'] and digest(Path(a.initialization))==v['rank128_checkpoint_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'];base.load_state_dict(saved['model']);del teacher,saved
    for p in base.parameters():p.requires_grad_(False)
    models={n:TemporalPyramidDetector(base,'original').cuda().eval() for n in ('frozen','dense','rank128')}
    models['rank128'].core.readout,_=compact_readout(base.core.readout,128);models['rank128'].core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);models['rank128'].eval()
    for n,m in models.items():
        for name,p in m.named_parameters():p.requires_grad_(n!='frozen' and name.startswith(PREFIX))
    training,evaluation=rows[:512],rows[512:768];contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),initialization_sha256=digest(Path(a.initialization)),train_frames=[r['frame'] for r in training],evaluation_frames=[r['frame'] for r in evaluation],expected_update_frames=expected_updates(training),seed=303,tbptt=4,epochs=1,optimizer='AdamW',lr=1e-4,weight_decay=1e-4,clip_norm=10.,precision=dict(matmul_tf32=False,cudnn_tf32=False),code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        loss='Identical supervised-window detection loss only, no global MSE. Only per-frame readout parameters learn, upstream/head/BN running statistics frozen.',boundary='Fixed512/71 matched adaptation. Rank128 loaded exact R145 SVD initialization; dense starts original. StrictFP32 for all arms differs from older default-TF32 AP protocol; own frozen control required. No fullval or protected-cache combination.')
    write_json(out/'contract.json',contract);criterion=PointPillarDirLoss(cfg['loss']['args']).cuda();names=('dense','rank128')
    preflight={n:train_one(models[n],cfg,training,out,n,criterion,contract,True) for n in names};write_json(out/'preflight.json',preflight)
    initial,raw0=evaluate(models,base,cfg,evaluation,out,'initial');assert raw0['dense']==raw0['frozen'];learned={n:train_one(models[n],cfg,training,out,n,criterion,contract) for n in names};final,raw1=evaluate(models,base,cfg,evaluation,out,'final');assert raw0['frozen']==raw1['frozen']
    write_json(out/'sort_orders.json',dict(numpy_version=np.__version__,source_sha256={tag:digest(out/(tag+'_ap_records.json')) for tag in ('initial','final')},orders={tag:{n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in raw.items()} for tag,raw in (('initial',raw0),('final',raw1))}))
    write_json(out/'report.json',dict(status='COMPLETED',initial=initial,final=final,training=learned,preflight=preflight,trained=True,timing_valid=False,frozen_raw_ap_exact=True));write_json(out/'status.json',dict(status='COMPLETED'))

def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','initialization','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90min compact readout learning')));signal.alarm(5400)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise

if __name__=='__main__':main()
