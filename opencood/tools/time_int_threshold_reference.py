"""Counterbalanced ANN/implicit/threshold old-reference model timing."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_implicit_integer_reference import ImplicitReferenceIntegerDetector
from opencood.models.sub_modules.int_threshold_reference import ThresholdImplicitReferenceDetector
import hashlib
from opencood.tools.time_int_periodic_compact import exclusive
import os,time,numpy as np
def worker(a,out):
    uuid=os.environ["CUDA_VISIBLE_DEVICES"];exclusive(uuid)
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
    
    proof=json.loads(Path(a.threshold_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-215' and proof['checkpoint_sha256']==gate['checkpoint_sha256'] and proof['calibration_sha256']==digest(Path(a.calibration))
    calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    models=dict(control=source,implicit=ImplicitReferenceIntegerDetector(source,calibration['quanta'],True).cuda().eval(),threshold=ThresholdImplicitReferenceDetector(source,calibration['quanta']).cuda().eval());selected=rows[512:768];assert len(selected)==256
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),threshold_verification_sha256=digest(Path(a.threshold_verification)),calibration_sha256=digest(Path(a.calibration)),frames=[r['frame'] for r in selected],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),uuid=uuid,device=torch.cuda.get_device_name(),torch_version=torch.__version__,passes=2,warmup_scans_per_pass=16,pass_mean_ratio_limit=1.15,precision=dict(matmul_tf32=False,cudnn_tf32=False),
        boundary='Same GPU-exclusive each frame, host shared. Three co-resident models/own states; train256 twice,16 warmups each,480 samples/arm; cyclic3/reversed order. Synchronized full model.step includes Pillar/scatter/warp/current decoder/old reference decode and built-in diagnostics; no hooks or external audit reference capture. Threshold compilation occurs inside first warmup; fixed tables retained across second pass, one-time compiler work separately recorded. AP/untrained; pass mean ratio>1.15 invalidates timing without retries. Dense/FP64/integer operations and static tables separate; no neural-hardware energy/latency claim.')
    write_json(out/'contract.json',contract)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for repeat in range(2):
            states={n:None for n in models};_,stream=loader(selected,cfg,2,303)
            for i,sample in enumerate(stream):
                exclusive(uuid);meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));names=list(models);offset=i%3;order=names[offset:]+names[:offset]
                if repeat:order=list(reversed(order))
                row=dict(repeat=repeat,index=i,frame=meta['frame'],included=i>=16,execution_order=order,arms={});preds={}
                for n in order:
                    torch.cuda.synchronize();start=time.perf_counter();pred,states[n],info=models[n].step(ego,states[n],meta);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000;preds[n]=pred
                    assert all(bool(torch.isfinite(v).all()) for v in pred.values());e=info.get('readout_events',{});logical=231067468800 if info['full_refresh'] else 105679929600
                    assert e.get('threshold_reference_audit_bytes',0)==0
                    row['arms'][n]=dict(model_step_ms=ms,refresh=info['full_refresh'],logical_core_conv_flops=logical,core_fp32_conv_flops=logical-(40461926400 if e else 0),readout_fp64_reference_flops=sum(z['actual_fp64_conv_flops'] for z in e.get('layers',[])),readout_integer_macs=sum(z['actual_integer_macs'] for z in e.get('layers',[])),cache_bytes=info['cache_bytes'],events=e,old_reference_float_elements=e.get('threshold_float_fallback_elements',e.get('reconstructed_reference_elements',0)))
                assert all(torch.equal(preds['implicit'][k],preds['threshold'][k]) for k in preds['implicit'])
                assert all(torch.equal(states['implicit'][k],states['threshold'][k]) for k in base.state_keys)
                assert torch.equal(states['implicit']['readout'][0],states['threshold']['readout'][0]) and all(torch.equal(a,b) for a,b in zip(states['implicit']['readout_accumulators'],states['threshold']['readout_accumulators']))
                for key,value in row['arms']['implicit']['events'].items():
                    if key!='storage_policy':assert row['arms']['threshold']['events'][key]==value
                records.append(row);log.write(json.dumps(row)+'\n')
                if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',repeat=repeat,scans=i+1,total=256))
    exclusive(uuid);assert all(model_digest(m)==hashes[n] for n,m in models.items());metrics={}
    for n,m in models.items():
        chosen=[r for r in records if r['included']];values=[r['arms'][n]['model_step_ms'] for r in chosen];assert len(values)==480;pp=[float(np.mean([r['arms'][n]['model_step_ms'] for r in chosen if r['repeat']==rep])) for rep in range(2)];phases={}
        for phase,refresh in (('refresh',True),('ordinary',False)):
            z=[r['arms'][n] for r in chosen if bool(r['arms'][n]['refresh'])==refresh];phases[phase]=dict(samples=len(z),model_step_ms_mean=float(np.mean([x['model_step_ms'] for x in z])))
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),samples=480,model_step_ms_mean=float(np.mean(values)),model_step_ms_p50=float(np.median(values)),model_step_ms_p95=float(np.percentile(values,95)),per_pass_ms=pp,pass_mean_ratio=max(pp)/min(pp),phase_metrics=phases,**{k+'_mean':sum(r['arms'][n][k] for r in chosen)/480 for k in ('logical_core_conv_flops','core_fp32_conv_flops','readout_fp64_reference_flops','readout_integer_macs','cache_bytes','old_reference_float_elements')},ap50=None,ap70=None)
    tables=models['threshold'].core.readout.threshold_maps;assert len(tables)==3
    for m,g in zip(tables,proof['tables']):assert hashlib.sha256(m.thresholds.cpu().numpy().tobytes()).hexdigest()==g['threshold_sha256'] and hashlib.sha256(m.orientation.cpu().numpy().tobytes()).hexdigest()==g['orientation_sha256']
    build=sum(r['arms']['threshold']['events']['threshold_build_float_elements'] for r in records);assert build==proof['one_time_build_float_elements'] and not any(r['arms']['threshold']['events']['threshold_build_float_elements'] for r in records if r['included'])
    valid=all(z['pass_mean_ratio']<=1.15 for z in metrics.values());write_json(out/'report.json',dict(status='COMPLETED',timing_valid=valid,metrics=metrics,threshold_table_bytes=sum(m.table_bytes for m in tables),one_time_build_float_elements=build,table_hashes_replay=True,weights_unchanged=True,predictions_task_n0_A_exact=True,trained=False,ap_evaluated=False));write_json(out/'status.json',dict(status='COMPLETED',timing_valid=valid))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','calibration','threshold-verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min threshold reference paired timing')));signal.alarm(3600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False));raise
if __name__=='__main__':main()
