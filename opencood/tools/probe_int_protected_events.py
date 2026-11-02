"""Trained protected state identity and true sparse event engineering, no AP."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
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


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());assert audit['status']==gate['status']=='VERIFIED' and gate['run']=='R-139'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256']
    for r in audit['files']:assert digest(Path(r['relocated_path']))==r['sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract'];base.load_state_dict(saved['model']);del teacher,saved;rows=rows[512:544]
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();path=Path(a.trained)/'protected_light_final.pth';load_targets(source,path,gate['training']['protected_light']['checkpoint_sha256'])
    models=dict(ann=source);models.update({n:ProtectedEventDetector(source,n).cuda().eval() for n in ('continuous','quant_dense','event_phase')})
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),trained_sha256=digest(path),frames=[r['frame'] for r in rows],quantum=.25,period=4,max_age=.45,code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        boundary='Trained ANN gates/protected refresh are unchanged membrane update; no extra leakage or duplicated ANN history. Quantized sent feeds deep and readout; quant_dense/event_phase matched. Candidate and encoders remain ANN. Engineering no AP/training/timing; actual sparse MAC added to Conv-hook count.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);entry=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);preds[n]=pred;assert all(torch.isfinite(v).all() for v in pred.values())
                mac=info.get('weighted_event_macs',0);entry['arms'][n]=dict(conv_hook_flops=cost,weighted_event_macs=mac,actual_core_flops=cost+2*mac,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],reset=info['reset'],events=info.get('events'))
            for k in preds['ann']:assert torch.equal(preds['ann'][k],preds['continuous'][k]);assert torch.allclose(preds['quant_dense'][k],preds['event_phase'][k],atol=2e-4,rtol=1e-4)
            for k in source.state_keys:assert torch.equal(states['ann'][k],states['continuous'][k]);assert torch.equal(states['quant_dense'][k],states['event_phase'][k])
            assert torch.equal(states['ann']['coarse'],states['quant_dense']['coarse']) and torch.equal(states['quant_dense']['sent'],states['event_phase']['sent'])
            entry['max_phase_prediction_error']=max(float((preds['quant_dense'][k]-preds['event_phase'][k]).abs().max()) for k in preds['ann']);records.append(entry);log.write(json.dumps(entry)+'\n')
    assert all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['actual_core_flops'] for r in records)/32,cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=32,continuous_exact=True,quantized_states_exact=True,weights_unchanged=True,max_phase_prediction_error=max(r['max_phase_prediction_error'] for r in records),trained=False,ap_evaluated=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min trained protected event engineering')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
