"""Frozen joint quantized interface for ordinary-state adaptation; no real training."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_fine_integer import FINE_POLICY
from opencood.models.sub_modules.int_fine_readout_integer import FineReadoutIntegerDetector
from opencood.models.sub_modules.int_joint_state_qat import JointStateQATDetector,STATE_QAT_PREFIXES
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
    proof=json.loads(Path(a.fine_stream_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-205' and proof['calibration_sha256']==digest(Path(a.calibration))
    iv=json.loads(Path(a.implicit_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-201' and iv['checkpoint_sha256']==gate['checkpoint_sha256'] and iv['calibration_sha256']==digest(Path(a.readout_calibration))
    rc=json.loads(Path(a.readout_calibration).read_text());assert rc['relative_quantum']==.125 and rc['floor']==1e-4
    jv=json.loads(Path(a.joint_verification).read_text());assert jv['status']=='VERIFIED' and jv['run']=='R-209' and jv['checkpoint_sha256']==gate['checkpoint_sha256'] and jv['calibration_sha256']==digest(Path(a.calibration)) and jv['readout_calibration_sha256']==digest(Path(a.readout_calibration))
    gp=Path(a.joint_source)/'frames.jsonl';assert digest(gp)==jv['source_sha256']['frames.jsonl'];gold=[json.loads(x) for x in gp.read_text().splitlines()];assert len(gold)==32
    models=dict(control=source,ann=JointStateQATDetector(source,calibration['layers'],rc['quanta'],False).cuda().eval(),joint_dense=FineReadoutIntegerDetector(source,calibration['layers'],rc['quanta'],'dense',False).cuda().eval(),qat=JointStateQATDetector(source,calibration['layers'],rc['quanta'],True).cuda().eval())
    selected=rows[512:544];assert len(selected)==32
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[]
    contract=dict(joint_verification_sha256=digest(Path(a.joint_verification)),checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),calibration_sha256=digest(Path(a.calibration)),readout_calibration_sha256=digest(Path(a.readout_calibration)),frames=[r['frame'] for r in selected],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),precision=dict(matmul_tf32=False,cudnn_tf32=False),future_trainable_prefixes=STATE_QAT_PREFIXES,
        boundary='Fixed32 frozen own-state no AP/optimizer/real gradient pilot. ANN adapter exactly replays source, joint QAT forward exactly replays joint dense integer predictions/task states and fine layer ledger; original control/joint_dense operation trace replays R209. Frozen fine runs no_grad; existing readout STE can transmit gradients to later ordinary state functions in synthetic tests only. Fine/readout weights and BN stats fixed. QAT has only task state22.5792MB; not deployment state saving or latency. Future matched training requires sole quality GPU and separate real no-step gradient gate.')
    write_json(out/'contract.json',contract);_,stream=loader(selected,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);preds[n]=pred;assert all(bool(torch.isfinite(v).all()) for v in pred.values())
                layers=info.get('fine_integer',{})
                if n=='qat':layers={k:dict(v.statistics) for k,v in m.core.observation.leaves().items()}
                fine_fp64=sum(z['actual_fp64_flops'] for z in layers.values());readout_fp64=sum(z['actual_fp64_conv_flops'] for z in info.get('readout_events',{}).get('layers',[]))
                if n in ('ann','qat'):readout_fp64=m.core.readout.last_reference_flops
                row['arms'][n]=dict(core_fp32_conv_flops=fp32,fine_fp64_reference_flops=fine_fp64,readout_fp64_reference_flops=readout_fp64,logical_core_conv_flops=fp32+fine_fp64+readout_fp64,cache_bytes=info['cache_bytes'],reset=info['reset'],refresh=info['full_refresh'],layers=layers)
            for a0,b0 in (('control','ann'),('joint_dense','qat')):
                assert all(torch.equal(preds[a0][k],preds[b0][k]) for k in preds[a0])
                assert all(torch.equal(states[a0][k],states[b0][k]) for k in base.state_keys)
                assert all(row['arms'][a0][k]==row['arms'][b0][k] for k in ('core_fp32_conv_flops','fine_fp64_reference_flops','readout_fp64_reference_flops','logical_core_conv_flops','refresh','reset','layers'))
            g=gold[index];assert row['frame']==g['frame']
            for name,old in (('control','control'),('joint_dense','fine_dense')):
                assert all(row['arms'][name][k]==g['arms'][old][k] for k in ('core_fp32_conv_flops','fine_fp64_reference_flops','readout_fp64_reference_flops','logical_core_conv_flops','cache_bytes','refresh','reset','layers'))
            records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=32))
    assert all(model_digest(m)==hashes[n] for n,m in models.items());metrics={}
    for n,m in models.items():
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),future_trainable_parameters=sum(p.numel() for k,p in m.named_parameters() if k.startswith(STATE_QAT_PREFIXES)),**{k+'_mean':sum(r['arms'][n][k] for r in records)/32 for k in ('core_fp32_conv_flops','fine_fp64_reference_flops','readout_fp64_reference_flops','logical_core_conv_flops','cache_bytes')},model_step_ms_mean=None,ap50=None,ap70=None)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,weights_unchanged=True,ann_predictions_tasks_exact=True,qat_predictions_tasks_exact=True,source_trajectory_exact=True,trained=False,real_gradient_tested=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','primitive-verification','calibration','fine-stream-verification','implicit-verification','readout-calibration','joint-verification','joint-source','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min joint state QAT interface')));signal.alarm(3600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
