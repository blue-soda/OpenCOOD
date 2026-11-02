"""Fixed train256 task-membrane controls, including dense/event equivalence."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import TemporalPyramidDetector
from opencood.models.sub_modules.int_task_membrane_detector import TaskMembraneDetector
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text())
    assert audit['status']==gate['status']=='VERIFIED' and gate['run']=='R-143'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model']);del teacher,saved;rows=rows[512:768]
    models=dict(frozen=TemporalPyramidDetector(base,'original').cuda().eval())
    models.update({m:TaskMembraneDetector(base,m).cuda().eval() for m in ('continuous','reset','events_dense','events_phase')})
    control=TemporalPyramidDetector(base,'leaky').cuda().eval()
    for m in [base,control]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=control_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),frames=[r['frame'] for r in rows],seed=303,quantum=.25,tau=.2,projection_period=4,max_age=.45,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Frozen coarse task-membrane intervention on dense parent, not protected-cache combination or full SNN. Continuous v versus reset isolates this coarse inference memory only; fine/deep ANN history remains. Events send s to deep encoder and readout. Dense versus phase-event has identical quantization and recurrence. Train256 diagnostic only; no training or timing.',
        cost='Executed core conv hooks PLUS 2*actual sparse weighted-event MAC. Candidate and other network modules remain dense; warp/scatter/nonconv not included.')
    write_json(out/'contract.json',contract);ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models};dataset,stream=loader(rows,cfg,2,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);preds={};entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(bool(torch.isfinite(v).all()) for v in pred.values());preds[n]=pred
                event_macs=info.get('weighted_event_macs',0)
                entry['arms'][n]=dict(core_conv_flops=cost+2*event_macs,conv_hook_flops=cost,weighted_event_macs=event_macs,cache_bytes=info['cache_bytes'],reset=info['reset'],projection_refresh=info.get('projection_refresh'),events=info.get('events'))
            reference,control_state,_=control.step_spatial(x,control_state,meta)
            assert all(torch.equal(reference[k],preds['continuous'][k]) for k in reference)
            assert all(torch.equal(states['events_dense'][k],states['events_phase'][k]) for k in ('value','coarse','deep','sent'))
            entry['phase_prediction_max_error']=max(float((preds['events_dense'][k]-preds['events_phase'][k]).abs().max()) for k in reference)
            assert all(torch.allclose(preds['events_dense'][k],preds['events_phase'][k],atol=2e-4,rtol=1e-4) for k in reference)
            if meta['supervised']:
                batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch)
                for n,pred in preds.items():
                    boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            records.append(entry);log.write(json.dumps(entry)+'\n')
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=i+1,total=256))
    assert len(records)==256 and sum(r['supervised'] for r in records)==192 and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'ap_records.json',ap);write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/256,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=256,labels=192,continuous_exact=True,phase_states_exact=True,max_phase_prediction_error=max(r['phase_prediction_max_error'] for r in records),all_weights_unchanged=True,trained=False,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    write_json(out/'status.json',dict(status='RUNNING',phase='tests'));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min fixed task membrane diagnostic')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
