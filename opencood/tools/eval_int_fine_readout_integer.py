"""Full fixed joint fine/readout integer quality; no isolated-AP substitution."""
from opencood.tools.pilot_int_causal_readout_qat import *
from opencood.tools.int_artifact_paths import artifact_path
from opencood.models.sub_modules.int_fine_readout_integer import FineReadoutIntegerDetector
def worker(a,out):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','opencood/test','-p','test_int*.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'unit_tests.log').write_bytes(tests.stdout);assert tests.returncode==0
    audit=json.loads(Path(a.audit).read_text());gate=json.loads(Path(a.verification).read_text());cv=json.loads(Path(a.compact_verification).read_text())
    assert audit['frames']==5794 and len(audit['files'])==7532;assert gate['status']==cv['status']=='VERIFIED' and gate['run']=='R-139' and cv['run']=='R-145'
    assert digest(Path(a.checkpoint))==gate['checkpoint_sha256'] and digest(Path(a.initialization))==cv['rank128_checkpoint_sha256']
    for item in audit['files']:assert digest(Path(artifact_path(item['path'])))==item['sha256']
    a.mode='gru';a.stage='full';a.content_calibration=None;a.periodic_last=0;a.max_age=.25
    teacher,cfg,rows,prov=setup(a);write_json(out/'provenance.json',prov);base=PyramidStateDetector(teacher,cfg).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu')
    assert saved['contract']['teacher_sha256']==prov['checkpoint']['sha256'] and saved['contract']['data_contract']==prov['contract'];base.load_state_dict(saved['model']);del teacher,saved
    source=ProtectedMultiscaleRefreshDetector(base,True).cuda().eval();load_targets(source,Path(a.trained)/'protected_light_final.pth',gate['training']['protected_light']['checkpoint_sha256'])
    source.core.readout,_=compact_readout(base.core.readout,128);source.core.readout.load_state_dict(torch.load(a.initialization,map_location='cpu')['readout']);source.eval()
    for p in source.parameters():p.requires_grad_(False)
    
    fv=json.loads(Path(a.fine_verification).read_text());assert fv['status']=='VERIFIED' and fv['run']=='R-200'
    assert digest(Path(a.calibration))==fv['source_sha256']['calibration.json'] and fv['checkpoint_sha256']==gate['checkpoint_sha256']
    pv=json.loads(Path(a.primitive_verification).read_text());assert pv['status']=='VERIFIED' and pv['run']=='R-204' and pv['tests']==446
    calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    proof=json.loads(Path(a.fine_stream_verification).read_text());assert proof['status']=='VERIFIED' and proof['run']=='R-205' and proof['checkpoint_sha256']==gate['checkpoint_sha256'] and proof['calibration_sha256']==digest(Path(a.calibration))
    rv=json.loads(Path(a.reference_verification).read_text());assert rv['status']=='VERIFIED' and rv['run']=='R-184'
    refpath=Path(a.reference_source)/'ap_records.json';assert digest(refpath)==rv['source_sha256']['ap_records.json'];gold=json.loads(refpath.read_text())['control']
    joint=json.loads(Path(a.joint_verification).read_text());assert joint['status']=='VERIFIED' and joint['run']=='R-209' and joint['checkpoint_sha256']==gate['checkpoint_sha256'] and joint['calibration_sha256']==digest(Path(a.calibration)) and joint['readout_calibration_sha256']==digest(Path(a.readout_calibration))
    rc=json.loads(Path(a.readout_calibration).read_text());assert rc['relative_quantum']==.125 and rc['floor']==1e-4
    models=dict(control=source,joint_dense=FineReadoutIntegerDetector(source,calibration['layers'],rc['quanta'],'dense',False).cuda().eval())
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    assert len(rows)==5794 and sum(r['supervised'] for r in rows)==1738
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    contract=dict(joint_verification_sha256=digest(Path(a.joint_verification)),readout_calibration_sha256=digest(Path(a.readout_calibration)),checkpoint_sha256=digest(Path(a.checkpoint)),protected_checkpoint_sha256=gate['training']['protected_light']['checkpoint_sha256'],initialization_sha256=digest(Path(a.initialization)),fine_verification_sha256=digest(Path(a.fine_verification)),fine_stream_verification_sha256=digest(Path(a.fine_stream_verification)),reference_verification_sha256=digest(Path(a.reference_verification)),calibration_sha256=digest(Path(a.calibration)),frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),precision=dict(matmul_tf32=False,cudnn_tf32=False),
        boundary='Full5794/1738 original ANN and fixed jointly quantized fine/readout dense quality. Both calibrations frozen in R209 before AP; no retuning/training/epoch selection. Raw ANN AP must replay R184. Dense FP64 quality reference, no sparse deployment cost/latency claim. Dense readout state retained; actual hybrid layout needs separate engineering evidence. No borrowed isolated AP or independent test claim.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as log:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                (pred,states[n],info),fp32=measured_step(m,x,states[n],meta);assert all(bool(torch.isfinite(z).all()) for z in pred.values());e=info.get('fine_integer',{})
                fp64=sum(z['actual_fp64_flops'] for z in e.values());assert not any(z['actual_integer_macs'] for z in e.values())
                re=info.get('readout_events',{});rf64=sum(z['actual_fp64_conv_flops'] for z in re.get('layers',[]));assert not any(z['actual_integer_macs'] for z in re.get('layers',[]))
                assert info['cache_bytes']==(22579200 if n=='control' else 358041600)
                row['arms'][n]=dict(core_fp32_conv_flops=fp32,fine_fp64_reference_flops=fp64,readout_fp64_reference_flops=rf64,logical_core_conv_flops=fp32+fp64+rf64,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'],layers=e,readout_events=re)
                if meta['supervised']:
                    batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch);boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            records.append(row);log.write(json.dumps(row)+'\n')
            if len(records)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=5794))
    assert len(records)==5794 and all(model_digest(m)==hashes[n] for n,m in models.items());write_json(out/'ap_records.json',ap);assert json.loads((out/'ap_records.json').read_text())['control']==gold
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),**{k+'_mean':sum(r['arms'][n][k] for r in records)/5794 for k in ('core_fp32_conv_flops','fine_fp64_reference_flops','readout_fp64_reference_flops','logical_core_conv_flops')},cache_bytes=22579200 if n=='control' else 358041600,model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=5794,labels=1738,weights_unchanged=True,reference_raw_ap_exact=True,trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))


def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','fine-verification','primitive-verification','calibration','fine-stream-verification','reference-verification','reference-source','joint-verification','readout-calibration','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180min fixed joint fine/readout integer full AP')));signal.alarm(10800)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
