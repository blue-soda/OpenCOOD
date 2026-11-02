"""Frozen task-subthreshold memory interface proof, no AP or real optimization."""
from opencood.tools.pilot_int_causal_readout_qat import *
from opencood.tools.int_artifact_paths import artifact_path
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadoutDetector
from opencood.models.sub_modules.int_compact_causal_integer_state import CompactIntegerReadoutDetector
from torch import nn
from opencood.models.sub_modules.int_subthreshold_membrane import SubthresholdMembraneDetector,quantize_ste

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
    models=dict(control=source);models.update({n:SubthresholdMembraneDetector(source,n,.25).cuda().eval() for n in ('continuous','carry','drop','reset')});rows=rows[512:544]
    for m in [base]+list(models.values()):
        for p in m.parameters():p.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),verification_sha256=digest(Path(a.verification)),initialization_sha256=digest(Path(a.initialization)),quantum=.25,frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),precision=dict(matmul_tf32=False,cudnn_tf32=False),boundary='Frozen32 single coarse task-state interface. Continuous must exactly replay original protected ANN; carry saves v, drop saves Q(v), both communicate Q(v) every frame, reset clears coarse history only. No extra persistent state or new parameters; dense updates and convolutions retained. Nearest fixed.25 chosen from earlier task-membrane work, no AP threshold search. Synthetic STE derivative tests only; no real backward/optimizer/AP/model timing.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);preds[n]=pred;assert all(torch.isfinite(z).all() for z in pred.values())
                row['arms'][n]=dict(core_conv_flops=cost,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],reset=info['reset'],membrane=info.get('membrane'))
                assert {k for k,v in states[n].items() if torch.is_tensor(v)}==set(base.state_keys)
            assert all(torch.equal(preds['control'][k],preds['continuous'][k]) for k in preds['control'])
            assert all(torch.equal(states['control'][k],states['continuous'][k]) for k in base.state_keys)
            assert all(row['arms'][n]['core_conv_flops']==row['arms']['control']['core_conv_flops'] and row['arms'][n]['cache_bytes']==22579200 and row['arms'][n]['refresh']==row['arms']['control']['refresh'] for n in models)
            assert torch.equal(states['drop']['coarse'],quantize_ste(states['drop']['coarse'],.25))
            if row['arms']['control']['reset']:assert all(torch.equal(preds['carry'][k],preds['drop'][k]) for k in preds['control'])
            row['carry_drop_coarse_different']=not torch.equal(states['carry']['coarse'],states['drop']['coarse'])
            records.append(row);f.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=32))
    assert len(records)==32 and all(model_digest(m)==hashes[n] for n,m in models.items()) and any(r['carry_drop_coarse_different'] for r in records)
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),core_conv_flops_mean=sum(r['arms'][n]['core_conv_flops'] for r in records)/32,cache_bytes=22579200,model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=32,weights_unchanged=True,continuous_predictions_exact=True,continuous_task_states_exact=True,reset_carry_drop_predictions_exact=True,same_state_budget=True,trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))

def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.full=False;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('45min frozen subthreshold task-state proof')));signal.alarm(2700)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
