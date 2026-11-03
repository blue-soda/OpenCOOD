"""Frozen fine event fanout and pointwise temporal-support diagnostic."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_fine_integer import FineIntegerDetector,FINE_POLICY
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.int_artifact_paths import artifact_path
from opencood.models.sub_modules.int_fine_support_audit import FineSupportAudit
from opencood.models.sub_modules.int_multiscale_refresh import reset_reason
def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
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
    models=dict(control=source,**{'fine_'+mode:FineIntegerDetector(source,calibration['layers'],mode,True).cuda().eval() for mode in ('dense','static','hybrid')})
    total=256;selected=rows[512:768];assert len(selected)==total
    tg=json.loads(Path(a.timing_verification).read_text());assert tg['status']=='VERIFIED' and tg['run']=='R-208' and tg['checkpoint_sha256']==gate['checkpoint_sha256'] and tg['calibration_sha256']==digest(Path(a.calibration))
    tp=Path(a.timing_source)/'frames.jsonl';assert digest(tp)==tg['source_sha256']['frames.jsonl'];gold=[json.loads(x) for x in tp.read_text().splitlines()][:256];assert len(gold)==256
    observer=FineSupportAudit(models['fine_hybrid'].core.observation)
    proof=json.loads(Path(a.fine_stream_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-205' and proof['checkpoint_sha256']==gate['checkpoint_sha256'] and proof['calibration_sha256']==digest(Path(a.calibration))
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(timing_verification_sha256=digest(Path(a.timing_verification)),scans=total,frame_scope='train256',fine_stream_verification_sha256=digest(Path(a.fine_stream_verification)),checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),fine_verification_sha256=digest(Path(a.fine_verification)),primitive_verification_sha256=digest(Path(a.primitive_verification)),calibration_sha256=digest(Path(a.calibration)),
        frames=[r['frame'] for r in selected],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),policy=FINE_POLICY,fine_execution_reset='scene/gap/start/explicit only; no extra P4 flush; rebuild via sparse current integer inputs',precision=dict(matmul_tf32=False,cudnn_tf32=False),
        boundary='Fixed train[512:768]256, no AP/training/timing or policy changes. Four original fine layouts retain own-state equality. Read-only hooks on hybrid fine Conv/BN/ReLU clone values before inplace ReLU; structural output support from nonzero executed input union, actual accumulator updates and observed temporal float output changes recorded separately. Shadow activations are audit-only, not deployment state; initial outputs charged fully. Static input support is current-input work, not temporal support. Event/operation ledger must replay first pass R208. Dense diagnostic reductions/pooling incur work, no pointwise skipping implemented.')
    write_json(out/'contract.json',contract);_,stream=loader(selected,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            observer.begin(bool(reset_reason(states['fine_hybrid'],sample['meta'])[0]))
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);preds[n]=pred;assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                e=info.get('fine_integer',{});fp64=sum(z['actual_fp64_flops'] for z in e.values());mac=sum(z['actual_integer_macs'] for z in e.values())
                row['arms'][n]=dict(core_fp32_conv_flops=fp32,fine_fp64_reference_flops=fp64,fine_integer_macs=mac,logical_core_conv_flops=fp32+(41287680000 if e else 0),cache_bytes=info['cache_bytes'],audit_scratch_bytes=info.get('audit_scratch_bytes',0),reset=info['reset'],refresh=info['full_refresh'],layers=e)
            reference=models['fine_dense'].core.observation.leaves()
            for n in ('fine_static','fine_hybrid'):
                assert all(torch.equal(preds['fine_dense'][k],preds[n][k]) for k in preds[n])
                assert all(torch.equal(states['fine_dense'][k],states[n][k]) for k in base.state_keys)
                for name,leaf in models[n].core.observation.leaves().items():assert all(torch.equal(a,b) for a,b in zip(reference[name].audit,leaf.audit))
            row['support']=observer.end()
            g=gold[index];assert g['frame']==meta['frame']
            for n in ('control','fine_static','fine_hybrid'):
                z=row['arms'][n];old=g['arms'][n];assert all(z[k]==old[k] for k in ('layers','core_fp32_conv_flops','fine_fp64_reference_flops','fine_integer_macs','logical_core_conv_flops','cache_bytes','reset')) and z['refresh']==old['full_refresh']
            for n in ('fine_dense','fine_static','fine_hybrid'):models[n].clear_audit()
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=total))
    assert len(records)==total and all(model_digest(m)==hashes[n] for n,m in models.items())
    observer.close()
    summary={}
    for key in records[0]['support']['modules']:
        summary[key]={}
        for scope in ('all','reuse'):
            values=[r['support']['modules'][key] for r in records if scope=='all' or not r['support']['modules'][key]['cold']]
            fields=[k for k,v in values[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
            summary[key][scope]=dict(samples=len(values),kind=values[0]['kind'],means={k:sum(z[k] for z in values)/len(values) for k in fields if all(z[k] is not None for z in values)})
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),**{key+'_mean':sum(r['arms'][n][key] for r in records)/total for key in ('core_fp32_conv_flops','fine_fp64_reference_flops','fine_integer_macs','logical_core_conv_flops','cache_bytes','audit_scratch_bytes')},cache_bytes_min=min(r['arms'][n]['cache_bytes'] for r in records),cache_bytes_max=max(r['arms'][n]['cache_bytes'] for r in records),model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,support_summary=summary,audit_shadow_max_bytes=max(r['support']['simultaneous_shadow_bytes'] for r in records),source_timing_trajectory_exact=True,weights_unchanged=True,integer_predictions_bit_exact=True,integer_task_states_exact=True,integer_inputs_accumulators_exact=True,trained=False,ap_evaluated=False,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','primitive-verification','calibration','timing-verification','timing-source','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--fine-stream-verification')
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    budget=3600
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('%dmin fine integer frozen stream'%(budget//60))));signal.alarm(budget)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
