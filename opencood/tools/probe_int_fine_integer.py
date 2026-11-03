"""Real fine-encoder integer residual stream: dense/current/hybrid equivalence."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_fine_integer import FineIntegerDetector,FINE_POLICY
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.int_artifact_paths import artifact_path
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
    total=a.validation_prefix or 32;selected=rows[:total] if a.validation_prefix else rows[512:544];assert len(selected)==total
    if a.validation_prefix:
        proof=json.loads(Path(a.fine_stream_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-205' and proof['checkpoint_sha256']==gate['checkpoint_sha256'] and proof['calibration_sha256']==digest(Path(a.calibration))
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(scans=total,frame_scope='validation_prefix' if a.validation_prefix else 'train32',fine_stream_verification_sha256=digest(Path(a.fine_stream_verification)) if a.validation_prefix else None,checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),fine_verification_sha256=digest(Path(a.fine_verification)),primitive_verification_sha256=digest(Path(a.primitive_verification)),calibration_sha256=digest(Path(a.calibration)),
        frames=[r['frame'] for r in selected],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),policy=FINE_POLICY,fine_execution_reset='scene/gap/start/explicit only; no extra P4 flush; rebuild via sparse current integer inputs',precision=dict(matmul_tf32=False,cudnn_tf32=False),
        boundary=('Validation prefix%d'%total if a.validation_prefix else 'Fixed train32')+' real own-state streams. Only fine Conv quantized; original float readout, recurrent updates, BN/ReLU/residual placement unchanged. Dense FP64 integer, static actual integer events and preselected hybrid delta use equal quantized inputs/predictions/tasks; ANN is quality reference only. No AP/optimizer/timing. Audit n/A buffers are transient and cleared after external comparison; persistent state reported separately from scratch/static tables.')
    write_json(out/'contract.json',contract);_,stream=loader(selected,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
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
            for n in ('fine_dense','fine_static','fine_hybrid'):models[n].clear_audit()
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=total))
    assert len(records)==total and all(model_digest(m)==hashes[n] for n,m in models.items())
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),**{key+'_mean':sum(r['arms'][n][key] for r in records)/total for key in ('core_fp32_conv_flops','fine_fp64_reference_flops','fine_integer_macs','logical_core_conv_flops','cache_bytes','audit_scratch_bytes')},cache_bytes_min=min(r['arms'][n]['cache_bytes'] for r in records),cache_bytes_max=max(r['arms'][n]['cache_bytes'] for r in records),model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,weights_unchanged=True,integer_predictions_bit_exact=True,integer_task_states_exact=True,integer_inputs_accumulators_exact=True,trained=False,ap_evaluated=False,timing_valid=False))
    write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','primitive-verification','calibration','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--validation-prefix',type=int,choices=(0,512,5794),default=0);p.add_argument('--fine-stream-verification')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    budget=14400 if a.validation_prefix==5794 else (5400 if a.validation_prefix else 2700)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('%dmin fine integer frozen stream'%(budget//60))));signal.alarm(budget)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
