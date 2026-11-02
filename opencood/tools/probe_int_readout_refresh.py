"""Readout fast-path real-input engineering before any learning/AP experiment."""
import argparse,json,signal,subprocess,sys
from pathlib import Path
import torch
from opencood.models.sub_modules.int_pyramid_state import PyramidStateDetector
from opencood.models.sub_modules.int_readout_refresh import ReadoutRefreshDetector
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.check_int_temporal_feature import model_digest
from opencood.tools.run_int_cache_repair import setup
from opencood.tools.run_int_ego import loader,to_device,write_json
from opencood.tools.launch_int_memory import digest


def worker(a,out):
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());assert audit['status']=='VERIFIED' and audit['frames']==768
    for row in audit['files']:assert digest(Path(row['relocated_path']))==row['sha256']
    assert digest(Path(a.checkpoint))==audit['checkpoint_sha256']
    a.mode='gru';a.stage='calibrate';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,provenance=setup(a);write_json(out/'provenance.json',provenance)
    base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==provenance['checkpoint']['sha256'] and saved['contract']['data_contract']==provenance['contract']
    base.load_state_dict(saved['model']);del teacher,saved
    models={}
    for n,mode,period in (('dense','dense',4),('period1','fast',1),('fast','fast',4),('reuse','reuse',4)):
        torch.manual_seed(303);models[n]=ReadoutRefreshDetector(base,mode,period).cuda().eval()
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};base_hash=model_digest(base);states={n:None for n in models};base_state=None
    rows=rows[512:544];contract=dict(frames=[r['frame'] for r in rows],checkpoint_sha256=digest(Path(a.checkpoint)),audit_sha256=digest(Path(a.audit)),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),boundary='32 real scans, no AP/training/timing; closed-loop readout states, upstream remains dense and identical to parent; random fast projection with zero correction.')
    write_json(out/'contract.json',contract);records=[];_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for i,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));captured={}
            h=base.core.readout.register_forward_hook(lambda m,x,y:captured.update(target=y))
            h0=base.core.observation.register_forward_pre_hook(lambda m,x:captured.update(spatial=x[0]))
            try:expected,base_state,base_info=base.step(ego,base_state,meta)
            finally:h.remove();h0.remove()
            x=captured['spatial']
            entry=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                calls=[];h=m.core.readout.register_forward_hook(lambda *_:calls.append(1))
                try:(pred,states[n],info),cost=measured_step(m,x,states[n],meta)
                finally:h.remove()
                assert len(calls)==int(info['readout_executed'])
                assert all(torch.equal(states[n][k],base_state[k]) for k in base.state_keys)
                if n in ('dense','period1'):assert all(torch.equal(pred[k],expected[k]) for k in pred)
                assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                err=float((states[n]['readout']-captured['target']).square().mean())
                entry['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],heavy_readout_calls=len(calls),
                    full_refresh=info['full_refresh'],readout_mse=err,holes_fraction=info['holes_fraction'])
            log.write(json.dumps(entry)+'\n');records.append(entry)
    assert model_digest(base)==base_hash and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/32,
        cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=None,ap70=None,
        heavy_readout_calls=sum(r['arms'][n]['heavy_readout_calls'] for r in records)) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=32,weights_unchanged=True,trained=False,ap_evaluated=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min readout engineering')));signal.alarm(1800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise


if __name__=='__main__':main()
