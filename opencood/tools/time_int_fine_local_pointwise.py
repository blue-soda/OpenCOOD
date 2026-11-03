"""Counterbalanced ANN/hybrid/local-BN1 fine-integer model timing."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_fine_integer import FineIntegerDetector,FINE_POLICY
from opencood.tools.time_int_periodic_compact import exclusive
from opencood.models.sub_modules.int_fine_local_pointwise import FineLocalPointwiseDetector
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
    
    fv=json.loads(Path(a.fine_verification).read_text());assert fv['status']=='VERIFIED' and fv['run']=='R-200'
    assert digest(Path(a.calibration))==fv['source_sha256']['calibration.json'] and fv['checkpoint_sha256']==gate['checkpoint_sha256']
    pv=json.loads(Path(a.primitive_verification).read_text());assert pv['status']=='VERIFIED' and pv['run']=='R-204' and pv['tests']==446
    calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    proof=json.loads(Path(a.fine_stream_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-218' and proof['checkpoint_sha256']==gate['checkpoint_sha256'] and proof['calibration_sha256']==digest(Path(a.calibration))
    models=dict(control=source,local=FineLocalPointwiseDetector(source,calibration['layers'],False).cuda().eval(),fine_hybrid=FineIntegerDetector(source,calibration['layers'],'hybrid',False).cuda().eval());selected=rows[512:768];assert len(selected)==256
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};records=[]
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),fine_verification_sha256=digest(Path(a.fine_verification)),fine_stream_verification_sha256=digest(Path(a.fine_stream_verification)),calibration_sha256=digest(Path(a.calibration)),frames=[r['frame'] for r in selected],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),uuid=uuid,device=torch.cuda.get_device_name(),torch_version=torch.__version__,passes=2,warmup_scans_per_pass=16,pass_mean_ratio_limit=1.15,policy=FINE_POLICY,precision=dict(matmul_tf32=False,cudnn_tf32=False),
        boundary='GPU-exclusive guard each frame; host shared; three models/own states co-resident. Fixed train256x2/16 warmups each/480 samples; cyclic3/reversed order. CUDA synchronize surrounds model.step including pillar/scatter/warp/task/readout/head and built-in integer range/event diagnostics; no Conv hooks, external comparisons/loader/H2D/AP excluded. Capture audit buffers disabled, static fanout tables remain. Local block1 BN1/ReLU output cache, clone/gather/scatter/support generation included in full step; all current Conv decode unchanged. Baseline/local integer predictions and task states exact. Pass mean ratio>1.15 invalidates entire timing without retry. No AP/optimization/neural-hardware latency.')
    write_json(out/'contract.json',contract)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for repeat in range(2):
            states={n:None for n in models};_,stream=loader(selected,cfg,2,303)
            for i,sample in enumerate(stream):
                exclusive(uuid);meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));names=list(models);offset=i%3;order=names[offset:]+names[:offset]
                if repeat:order=list(reversed(order))
                row=dict(repeat=repeat,index=i,frame=meta['frame'],included=i>=16,execution_order=order,arms={});preds={}
                for n in order:
                    torch.cuda.synchronize();start=time.perf_counter();pred,states[n],info=models[n].step(ego,states[n],meta);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
                    preds[n]=pred;assert all(bool(torch.isfinite(v).all()) for v in pred.values());e=info.get('fine_integer',{});logical=231067468800 if info['full_refresh'] else 105679929600
                    assert not any(z['actual_fp64_flops'] for z in e.values()) and info.get('audit_scratch_bytes',0)==0
                    row['arms'][n]=dict(model_step_ms=ms,full_refresh=info['full_refresh'],reset=info['reset'],logical_core_conv_flops=logical,core_fp32_conv_flops=logical-(41287680000 if e else 0),fine_fp64_reference_flops=0,fine_integer_macs=sum(z['actual_integer_macs'] for z in e.values()),cache_bytes=info['cache_bytes'],layers=e,pointwise=info.get('fine_pointwise'),bn1_elements=info.get('fine_pointwise',{}).get('processed_bn_elements',3225600))
                assert all(torch.equal(preds['local'][k],preds['fine_hybrid'][k]) for k in preds['local'])
                assert all(torch.equal(states['local'][k],states['fine_hybrid'][k]) for k in base.state_keys)
                for name,z in row['arms']['local']['layers'].items():
                    other=row['arms']['fine_hybrid']['layers'][name];assert z==other
                records.append(row);log.write(json.dumps(row)+'\n')
                if (i+1)%32==0:write_json(out/'status.json',dict(status='RUNNING',phase='timing',repeat=repeat,scans=i+1,total=256))
    exclusive(uuid);assert all(model_digest(m)==hashes[n] for n,m in models.items());metrics={}
    for n,m in models.items():
        chosen=[r for r in records if r['included']];values=[r['arms'][n]['model_step_ms'] for r in chosen];assert len(values)==480;pp=[float(np.mean([r['arms'][n]['model_step_ms'] for r in chosen if r['repeat']==rep])) for rep in range(2)]
        phases={}
        for phase,refresh in (('refresh',True),('ordinary',False)):
            z=[r['arms'][n] for r in chosen if bool(r['arms'][n]['full_refresh'])==refresh];phases[phase]=dict(samples=len(z),model_step_ms_mean=float(np.mean([x['model_step_ms'] for x in z])))
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),samples=480,model_step_ms_mean=float(np.mean(values)),model_step_ms_p50=float(np.median(values)),model_step_ms_p95=float(np.percentile(values,95)),per_pass_ms=pp,pass_mean_ratio=max(pp)/min(pp),phase_metrics=phases,
            **{k+'_mean':sum(r['arms'][n][k] for r in chosen)/480 for k in ('logical_core_conv_flops','core_fp32_conv_flops','fine_fp64_reference_flops','fine_integer_macs','cache_bytes','bn1_elements')},cache_bytes_min=min(r['arms'][n]['cache_bytes'] for r in chosen),cache_bytes_max=max(r['arms'][n]['cache_bytes'] for r in chosen),ap50=None,ap70=None)
    valid=all(z['pass_mean_ratio']<=1.15 for z in metrics.values());write_json(out/'report.json',dict(status='COMPLETED',timing_valid=valid,metrics=metrics,weights_unchanged=True,integer_predictions_bit_exact=True,integer_task_states_exact=True,trained=False,ap_evaluated=False,relative_to='control',relative_changes={n:z['model_step_ms_mean']/metrics['control']['model_step_ms_mean']-1 for n,z in metrics.items()}));write_json(out/'status.json',dict(status='COMPLETED',timing_valid=valid))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','primitive-verification','calibration','fine-stream-verification','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min fixed local pointwise model timing')));signal.alarm(3600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc),timing_valid=False));raise
if __name__=='__main__':main()
