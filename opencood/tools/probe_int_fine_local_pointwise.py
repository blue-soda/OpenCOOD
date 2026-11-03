"""One real local BN/ReLU path with matched closed-loop fine integer control."""
from opencood.tools.profile_int_compact_cache import *
from opencood.models.sub_modules.int_fine_integer import FineIntegerDetector,FINE_POLICY
from opencood.tools.check_int_pyramid_cache import measured_step
from opencood.tools.calibrate_int_event_refresh import capture
from opencood.tools.int_artifact_paths import artifact_path
from opencood.models.sub_modules.int_fine_local_pointwise import FineLocalPointwiseDetector
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
    proof=json.loads(Path(a.support_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-213' and proof['checkpoint_sha256']==gate['checkpoint_sha256'] and proof['calibration_sha256']==digest(Path(a.calibration))
    gp=Path(a.support_source)/'frames.jsonl';assert digest(gp)==proof['source_sha256']['frames.jsonl'];gold=[json.loads(x) for x in gp.read_text().splitlines()];assert len(gold)==256
    models=dict(fine_hybrid=FineIntegerDetector(source,calibration['layers'],'hybrid',True).cuda().eval(),local=FineLocalPointwiseDetector(source,calibration['layers'],True).cuda().eval())
    selected=rows[512:768];assert len(selected)==256
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];handles=[];outputs={};calls={}
    def capture_output(name):
        def hook(module,inputs):outputs[name]=inputs[0].detach().clone()
        return hook
    def count_bn(name):
        def hook(module,inputs,output):calls[name].append(inputs[0].numel())
        return hook
    for n,m in models.items():
        handles.append(m.core.observation.layers[1].conv2.register_forward_pre_hook(capture_output(n)))
        handles.append(m.core.observation.layers[1].bn1.register_forward_hook(count_bn(n)))
    contract=dict(support_verification_sha256=digest(Path(a.support_verification)),checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),calibration_sha256=digest(Path(a.calibration)),frames=[r['frame'] for r in selected],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),policy=FINE_POLICY,precision=dict(matmul_tf32=False,cudnn_tf32=False),
        boundary='Fixed train256, only block1 BN1/first ReLU evaluated on structural candidate positions. Dense Conv output decoding remains. New FP32 post-ReLU cache12.9024MB plus clone/gather/scatter/support work, counts separate from core Conv. Cold/full reset evaluates every point. Own-state prediction/task/n-A and intermediate post-ReLU exact, reference event/cost trajectory replays R213. Same frozen weights/quanta/policy, no AP/training/timing. Hooks/counts/output copies audit-only; not model latency.')
    write_json(out/'contract.json',contract);_,stream=loader(selected,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for index,sample in enumerate(stream):
            calls={n:[] for n in models};outputs={};meta=sample['meta'];ego=to_device(sample['ego'],torch.device('cuda'));x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],arms={});preds={}
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);preds[n]=pred;assert all(bool(torch.isfinite(v).all()) for v in pred.values());layers=info['fine_integer']
                row['arms'][n]=dict(core_fp32_conv_flops=fp32,fine_fp64_reference_flops=sum(z['actual_fp64_flops'] for z in layers.values()),fine_integer_macs=sum(z['actual_integer_macs'] for z in layers.values()),logical_core_conv_flops=fp32+41287680000,cache_bytes=info['cache_bytes'],audit_scratch_bytes=info['audit_scratch_bytes'],layers=layers,reset=info['reset'],refresh=info['full_refresh'],bn_call_elements=calls[n],pointwise=info.get('fine_pointwise'))
            assert torch.equal(outputs['fine_hybrid'],outputs['local']), 'Intermediate local BN/ReLU not bit exact'
            assert torch.equal(states['local']['fine_pointwise'],outputs['local'])
            assert all(torch.equal(preds['fine_hybrid'][k],preds['local'][k]) for k in preds['local'])
            assert all(torch.equal(states['fine_hybrid'][k],states['local'][k]) for k in base.state_keys)
            for name,leaf in models['local'].core.observation.leaves().items():assert all(torch.equal(a,b) for a,b in zip(models['fine_hybrid'].core.observation.leaves()[name].audit,leaf.audit))
            g=gold[index];assert row['frame']==g['frame']
            a0=row['arms']['fine_hybrid'];b0=row['arms']['local'];assert all(a0[k]==g['arms']['fine_hybrid'][k] for k in ('layers','core_fp32_conv_flops','fine_fp64_reference_flops','fine_integer_macs','logical_core_conv_flops','cache_bytes','reset','refresh'))
            assert all(a0[k]==b0[k] for k in ('layers','core_fp32_conv_flops','fine_fp64_reference_flops','fine_integer_macs','logical_core_conv_flops','reset','refresh'))
            q=b0['pointwise'];assert calls['fine_hybrid']==[3225600] and sum(calls['local'])==q['processed_bn_elements'] and q['processed_bn_elements']==q['processed_relu_elements']
            expected=3225600 if q['cold'] else g['support']['modules']['1.conv1:0']['candidate_elements'];assert q['processed_bn_elements']==expected and bool(q['cold'])==bool(b0['reset']) and b0['cache_bytes']==a0['cache_bytes']+12902400
            row['pointwise_audit_copy_bytes']=sum(z.numel()*z.element_size() for z in outputs.values());assert row['pointwise_audit_copy_bytes']==25804800
            for m in models.values():m.clear_audit()
            outputs={};records.append(row);log.write(json.dumps(row)+'\n');write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=256))
    for h in handles:h.remove()
    assert all(model_digest(m)==hashes[n] for n,m in models.items());metrics={}
    for n,m in models.items():
        metrics[n]=dict(parameters=sum(p.numel() for p in m.parameters()),**{k+'_mean':sum(r['arms'][n][k] for r in records)/256 for k in ('core_fp32_conv_flops','fine_fp64_reference_flops','fine_integer_macs','logical_core_conv_flops','cache_bytes','audit_scratch_bytes')},bn1_elements_mean=sum(sum(r['arms'][n]['bn_call_elements']) for r in records)/256,model_step_ms_mean=None,ap50=None,ap70=None)
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,weights_unchanged=True,predictions_task_n_A_exact=True,intermediate_pointwise_exact=True,source_trajectory_exact=True,pointwise_audit_copy_bytes=25804800,trained=False,ap_evaluated=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','primitive-verification','calibration','support-verification','support-source','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.validation_prefix=0;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min local pointwise proof')));signal.alarm(3600)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
