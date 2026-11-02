"""Fixed trained-gate ANN/membrane/quantized/event full-validation controls."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_multiscale_refresh import ProtectedMultiscaleRefreshDetector
from opencood.models.sub_modules.int_protected_events import ProtectedEventDetector
from opencood.tools.diagnose_int_cached_state import load_targets
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest
from opencood.utils import eval_utils


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());v=json.loads(Path(a.verification).read_text());tv=json.loads(Path(a.trained_verification).read_text())
    assert audit['status']==v['status']==tv['status']=='VERIFIED' and v['run']=='R-153' and tv['run']=='R-139';assert digest(Path(a.checkpoint))==v['checkpoint_sha256']==tv['checkpoint_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    rv=json.loads(Path(a.reference_verification).read_text());assert rv['status']=='VERIFIED' and rv['run']=='R-140'
    ref=Path(a.reference)/'ap_records.json';assert digest(ref)==rv['source_sha256']['ap_records.json'];gold=json.loads(ref.read_text())['protected_light']
    reference_rows=[json.loads(line) for line in (Path(a.reference)/'frames.jsonl').read_text().splitlines()]
    assert digest(Path(a.reference)/'frames.jsonl')==rv['source_sha256']['frames.jsonl']
    a.mode='gru';a.stage='full';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved;assert len(rows)==5794
    assert [r['frame'] for r in rows]==[r['frame'] for r in reference_rows]
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',tv['training']['protected_light']['checkpoint_sha256']);models=dict(ann=source)
    models.update({n:ProtectedEventDetector(source,n).cuda().eval() for n in ('continuous','quant_dense','event_phase')})
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),trained_verification_sha256=digest(Path(a.trained_verification)),trained_sha256=tv['training']['protected_light']['checkpoint_sha256'],reference_verification_sha256=digest(Path(a.reference_verification)),frames=[r['frame'] for r in rows],quantum=.25,period=4,max_age=.45,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Full5794/1738 fixed frozen validation comparison, no training or threshold search. Continuous is identical learned ANN state reinterpreted as membrane, not extra memory or independent SNN accuracy. Quantized sent feeds deep and readout; dense/event controls isolate execution. Core cost includes sparse weighted MAC, nonconv overhead excluded; no valid latency.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,2,303);ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);preds[n]=pred;assert all(torch.isfinite(v).all() for v in pred.values());mac=info.get('weighted_event_macs',0)
                entry['arms'][n]=dict(conv_hook_flops=cost,weighted_event_macs=mac,core_conv_flops=cost+2*mac,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],reset=info['reset'],events=info.get('events'))
            assert all(torch.equal(preds['ann'][k],preds['continuous'][k]) for k in preds['ann'])
            assert all(torch.equal(states['quant_dense'][k],states['event_phase'][k]) for k in (*source.state_keys,'sent'))
            assert all(torch.allclose(preds['quant_dense'][k],preds['event_phase'][k],atol=2e-4,rtol=1e-4) for k in preds['ann'])
            entry['max_phase_prediction_error']=max(float((preds['quant_dense'][k]-preds['event_phase'][k]).abs().max()) for k in preds['ann'])
            if meta['supervised']:
                batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch)
                for n,pred in preds.items():
                    boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            f.write(json.dumps(entry)+'\n');records.append(entry)
            if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=i+1,total=5794))
    assert len(records)==5794 and sum(r['supervised'] for r in records)==1738 and all(model_digest(m)==hashes[n] for n,m in models.items())
    write_json(out/'ap_records.json',ap);raw=json.loads((out/'ap_records.json').read_text());assert raw['ann']==raw['continuous']==gold
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/5794,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=5794,labels=1738,ann_raw_ap_exact=True,continuous_exact=True,quantized_states_exact=True,max_phase_prediction_error=max(r['max_phase_prediction_error'] for r in records),weights_unchanged=True,trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained-verification','trained','reference','reference-verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min protected event full quality')));signal.alarm(7200)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
