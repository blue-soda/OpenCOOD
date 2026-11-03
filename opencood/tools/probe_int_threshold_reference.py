"""Frozen integer threshold reference decode versus original float reconstruction."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_implicit_integer_reference import ImplicitReferenceIntegerDetector
from opencood.models.sub_modules.int_threshold_reference import ThresholdImplicitReferenceDetector
import hashlib,numpy as np
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
    
    iv=json.loads(Path(a.implicit_verification).read_text());full=json.loads(Path(a.implicit_full_verification).read_text())
    assert iv['status']==full['status']=='VERIFIED' and iv['run']=='R-201' and full['run']=='R-202' and iv['checkpoint_sha256']==full['checkpoint_sha256']==gate['checkpoint_sha256']
    assert iv['calibration_sha256']==full['calibration_sha256']==digest(Path(a.calibration))
    threshold_proof=json.loads(Path(a.threshold_verification).read_text()) if a.threshold_verification else None
    if a.validation_prefix:assert threshold_proof and threshold_proof['status']=='VERIFIED' and threshold_proof['run']=='R-215' and threshold_proof['checkpoint_sha256']==gate['checkpoint_sha256'] and threshold_proof['calibration_sha256']==digest(Path(a.calibration))
    calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    source_proof=full if a.validation_prefix else iv
    gp=Path(a.implicit_source)/'frames.jsonl';assert digest(gp)==source_proof['source_sha256']['frames.jsonl'];gold=[json.loads(x) for x in gp.read_text().splitlines()];assert len(gold)==(5794 if a.validation_prefix else 32)
    models=dict(implicit=ImplicitReferenceIntegerDetector(source,calibration['quanta'],True).cuda().eval(),threshold=ThresholdImplicitReferenceDetector(source,calibration['quanta']).cuda().eval())
    models['threshold'].core.readout.capture_reference_audit=True
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    rows=rows[:512] if a.validation_prefix else rows[512:544];assert len(rows)==(512 if a.validation_prefix else 32)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),calibration_sha256=digest(Path(a.calibration)),implicit_verification_sha256=digest(Path(a.implicit_verification)),implicit_full_verification_sha256=digest(Path(a.implicit_full_verification)),threshold_verification_sha256=digest(Path(a.threshold_verification)) if a.threshold_verification else None,validation_prefix=a.validation_prefix,frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),levels=4096,accumulator_bound=2**31-2,threshold_dtype='torch.int32',orientation_dtype='torch.int8',precision=dict(matmul_tf32=False,cudnn_tf32=False),
        boundary='Fixed train32 or val512 original implicit float old-reference decode versus monotone integer threshold lookup. Both own-state predictions/task/n0/A exact and old internal references exact. Fixed signed4096-level tables compiled on original spatial geometry, int32 range guard and whole-layer original float fallback preserve unsupported values. Current decoder/nonlinearities and all convolution work unchanged. Table capacity and one-time float compilation counted separately. Audit old n only for external comparison then cleared. No AP/training/model timing or independent task-memory/neural-energy benefit.')
    write_json(out/'contract.json',contract);_,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={};previous=states['implicit']
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);preds[n]=pred;assert all(bool(torch.isfinite(v).all()) for v in pred.values());e=info['readout_events'];fp64=sum(z['actual_fp64_conv_flops'] for z in e['layers']);mac=sum(z['actual_integer_macs'] for z in e['layers'])
                row['arms'][n]=dict(core_fp32_conv_flops=fp32,readout_fp64_reference_flops=fp64,readout_integer_macs=mac,logical_core_conv_flops=fp32+40461926400,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],events=e,old_reference_float_elements=e.get('threshold_float_fallback_elements',e['reconstructed_reference_elements']))
            assert all(torch.equal(preds['implicit'][k],preds['threshold'][k]) for k in preds['implicit'])
            assert all(torch.equal(states['implicit'][k],states['threshold'][k]) for k in base.state_keys)
            assert torch.equal(states['implicit']['readout'][0],states['threshold']['readout'][0]) and all(torch.equal(a,b) for a,b in zip(states['implicit']['readout_accumulators'],states['threshold']['readout_accumulators']))
            for k,value in row['arms']['implicit']['events'].items():
                if k!='storage_policy':assert row['arms']['threshold']['events'][k]==value
            decoder=models['threshold'].core.readout
            if decoder.reference_audit is not None:
                reference=models['implicit'].core.readout.reconstruct_references(previous['readout'],previous['readout_accumulators'])[1:]
                assert all(torch.equal(a,b) for a,b in zip(reference,decoder.reference_audit))
            decoder.reference_audit=None
            g=gold[index];assert row['frame']==g['frame']
            for k in ('core_fp32_conv_flops','readout_fp64_reference_flops','readout_integer_macs','logical_core_conv_flops','cache_bytes','refresh','events'):assert row['arms']['implicit'][k]==g['arms']['implicit'][k]
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=len(rows)))
    assert all(model_digest(m)==hashes[n] for n,m in models.items());maps=models['threshold'].core.readout.threshold_maps;assert len(maps)==3
    arrays={};tables=[]
    for i,m in enumerate(maps):
        for name,value in [('thresholds',m.thresholds),('orientation',m.orientation)]:arrays[name+'_'+str(i)]=value.cpu().numpy()
        tables.append(dict(index=i,shape=list(m.thresholds.shape),table_bytes=m.table_bytes,build_oracle_calls=m.build_oracle_calls,build_float_elements=m.build_float_elements,threshold_sha256=hashlib.sha256(m.thresholds.cpu().numpy().tobytes()).hexdigest(),orientation_sha256=hashlib.sha256(m.orientation.cpu().numpy().tobytes()).hexdigest()))
    if threshold_proof:
        for current,expected in zip(tables,threshold_proof['tables']):assert current==expected
    np.savez_compressed(out/'threshold_tables.npz',**arrays)
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),**{k+'_mean':sum(r['arms'][n][k] for r in records)/len(records) for k in ('core_fp32_conv_flops','readout_fp64_reference_flops','readout_integer_macs','logical_core_conv_flops','cache_bytes','old_reference_float_elements')},model_step_ms_mean=None,ap50=None,ap70=None) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,tables=tables,tables_sha256=digest(out/'threshold_tables.npz'),weights_unchanged=True,predictions_task_states_n0_A_exact=True,internal_references_exact=True,source_trajectory_exact=True,trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','calibration','implicit-verification','implicit-full-verification','implicit-source','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--threshold-verification')
    p.add_argument('--validation-prefix',type=int,choices=(0,512),default=0)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min threshold reference proof')));signal.alarm(3600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
