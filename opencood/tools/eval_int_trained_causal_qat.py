"""Frozen re-export proof for matched trained ANN/QAT readouts."""
from opencood.tools.pilot_int_causal_readout_qat import *
from opencood.tools.int_artifact_paths import artifact_path
from opencood.models.sub_modules.int_causal_integer_readout import IntegerReadoutDetector
from opencood.models.sub_modules.int_compact_causal_integer_state import CompactIntegerReadoutDetector
from torch import nn
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
    ev=json.loads(Path(a.event_verification).read_text());rv=json.loads(Path(a.reference_verification).read_text());assert ev['status']==rv['status']=='VERIFIED' and ev['run']=='R-172' and rv['run']==('R-148' if a.full else 'R-168')
    event_contract=Path(a.calibration).parent/'contract.json';assert digest(event_contract)==ev['source_sha256']['contract.json'];ec=json.loads(event_contract.read_text());assert ec['verification_sha256']==digest(Path(a.verification)) and ec['initialization_sha256']==digest(Path(a.initialization)) and ev['checkpoint_sha256']==gate['checkpoint_sha256']
    assert ev['calibration_sha256']==digest(Path(a.calibration));calibration=json.loads(Path(a.calibration).read_text());assert calibration['relative_quantum']==.125 and calibration['floor']==1e-4
    iv=json.loads(Path(a.integer_verification).read_text());assert iv['status']=='VERIFIED' and iv['run']=='R-181'
    icp=Path(a.integer_source)/'contract.json';assert digest(icp)==iv['source_sha256']['contract.json'];ic=json.loads(icp.read_text());assert ic['checkpoint_sha256']==gate['checkpoint_sha256'] and ic['calibration_sha256']==ev['calibration_sha256']
    qv=json.loads(Path(a.qat_verification).read_text());assert qv['status']=='VERIFIED' and qv['run']=='R-183';qcp=Path(a.qat_source)/'contract.json';assert digest(qcp)==qv['source_sha256']['contract.json']
    qc=json.loads(qcp.read_text());assert qc['calibration_sha256']==ev['calibration_sha256'] and qc['checkpoint_sha256']==gate['checkpoint_sha256']
    quality=json.loads(Path(a.quality_verification).read_text());assert quality['status']=='VERIFIED' and quality['run']=='R-182';rp=Path(a.quality_source)/'ap_records.json';assert digest(rp)==quality['source_sha256']['ap_records.json'];gold=json.loads(rp.read_text())
    if True:
        fv=json.loads(Path(a.fast_verification).read_text());assert fv['status']=='VERIFIED' and fv['run']=='R-188';fcp=Path(a.fast_source)/'contract.json';assert digest(fcp)==fv['source_sha256']['contract.json'];fc=json.loads(fcp.read_text());assert fc['checkpoint_sha256']==gate['checkpoint_sha256'] and fc['calibration_sha256']==ev['calibration_sha256']
    lv=json.loads(Path(a.learning_verification).read_text());assert lv['status']=='VERIFIED' and lv['run']=='R-185'
    lp=Path(a.learning_source);assert digest(lp/'contract.json')==lv['source_sha256']['contract.json'];lc=json.loads((lp/'contract.json').read_text())
    assert lc['checkpoint_sha256']==gate['checkpoint_sha256'] and lc['calibration_sha256']==ev['calibration_sha256'] and lc['initialization_sha256']==cv['rank128_checkpoint_sha256']
    trained={};exports={};checkpoint_hashes={}
    for name,enabled in (('ann',False),('qat',True)):
        path=lp/(name+'_readout_final.pth');assert digest(path)==lv['training'][name]['checkpoint_sha256'];saved=torch.load(path,map_location='cpu');assert saved['contract']==lc and saved['mode']==name and len(saved['updates'])==71
        model=copy.deepcopy(source);model.core.readout=CausalReadoutQAT(source.core.readout,calibration['quanta'],enabled,fast_eval=enabled);model.core.readout.load_state_dict(saved['readout']);model.eval();trained[name]=model
        export=copy.deepcopy(source);export.core.readout=nn.Sequential(*copy.deepcopy(list(model.core.readout.layers)));export.eval();exports[name]=export;checkpoint_hashes[name]=digest(path)
    eg=json.loads(Path(a.export_verification).read_text());assert eg['status']=='VERIFIED' and eg['run']=='R-190'
    ecp=Path(a.export_source)/'contract.json';assert digest(ecp)==eg['source_sha256']['contract.json'];ec=json.loads(ecp.read_text());assert ec['learning_verification_sha256']==digest(Path(a.learning_verification)) and ec['trained_checkpoint_sha256']==checkpoint_hashes
    fg=json.loads(Path(a.full_reference_verification).read_text());assert fg['status']=='VERIFIED' and fg['run']=='R-184';gp=Path(a.full_reference_source)/'ap_records.json';assert digest(gp)==fg['source_sha256']['ap_records.json'];gold=json.loads(gp.read_text())
    initial=copy.deepcopy(source);initial.core.readout=CausalReadoutQAT(source.core.readout,calibration['quanta'],True,True);initial.eval()
    models=dict(frozen=source,integer_initial=initial,ann=trained['ann'],qat=trained['qat'])
    for m in [base]+list(models.values()):
        for parameter in m.parameters():parameter.requires_grad_(False)
    assert len(rows)==5794 and sum(x['supervised'] for x in rows)==1738
    hashes={n:model_digest(m) for n,m in models.items()};states={n:None for n in models};base_state=None;records=[];ap={n:{t:dict(tp=[],fp=[],score=[],gt=0) for t in (.5,.7)} for n in models}
    contract=dict(checkpoint_sha256=digest(Path(a.checkpoint)),learning_verification_sha256=digest(Path(a.learning_verification)),export_verification_sha256=digest(Path(a.export_verification)),full_reference_verification_sha256=digest(Path(a.full_reference_verification)),trained_checkpoint_sha256=checkpoint_hashes,calibration_sha256=digest(Path(a.calibration)),frames=[r['frame'] for r in rows],code_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),precision=dict(matmul_tf32=False,cudnn_tf32=False),boundary='Full5794/1738 validation of both fixed71-update ANN/QAT terminal checkpoints and both initial references. No best-arm/epoch selection. QAT bounded FP32 radix reference actual operations separated from logical Conv FLOPs. No sparse deployment latency or actual energy measurement; no new training; validation-selected not independent test.')
    write_json(out/'contract.json',contract);dataset,stream=loader(rows,cfg,0,303)
    with torch.no_grad(),(out/'frames.jsonl').open('x',buffering=1) as f:
        for sample in stream:
            ego=to_device(sample['ego'],torch.device('cuda'));meta=sample['meta'];x,base_state=capture(base,ego,base_state,meta);row=dict(frame=meta['frame'],supervised=meta['supervised'],arms={})
            for n,m in models.items():
                (pred,states[n],info),cost=measured_step(m,x,states[n],meta);assert all(torch.isfinite(z).all() for z in pred.values())
                fp32ref=getattr(m.core.readout,'last_fp32_reference_flops',0);logical=getattr(m.core.readout,'last_logical_reference_flops',0);assert getattr(m.core.readout,'last_reference_flops',0)==0
                row['arms'][n]=dict(core_conv_flops=cost+fp32ref,core_fp32_conv_flops=cost+fp32ref,readout_fp32_reference_flops=fp32ref,logical_core_conv_flops=cost+logical,cache_bytes=info['cache_bytes'],refresh=info['full_refresh'])
                if meta['supervised']:
                    batch=dict(ego=ego);gt=dataset.post.generate_gt_bbx(batch);boxes,scores=dataset.post.post_process(batch,dict(ego=pred))
                    for t in (.5,.7):eval_utils.caluclate_tp_fp(boxes,scores,gt,ap[n],t)
            assert all(torch.equal(states[n][k],states['frozen'][k]) for n in models for k in base.state_keys)
            assert all(row['arms'][n]['logical_core_conv_flops']==row['arms']['frozen']['core_conv_flops'] for n in models)
            records.append(row);f.write(json.dumps(row)+'\n')
            if len(records)%32==0:write_json(out/'status.json',dict(status='RUNNING',scans=len(records),total=5794))
    assert len(records)==5794 and all(model_digest(m)==hashes[n] for n,m in models.items());write_json(out/'ap_records.json',ap);saved=json.loads((out/'ap_records.json').read_text());assert saved['frozen']==gold['control'] and saved['integer_initial']==gold['integer_dense']
    write_json(out/'sort_orders.json',dict(source_sha256=digest(out/'ap_records.json'),numpy_version=np.__version__,orders={n:{str(t):np.argsort(-np.asarray(b['score'])).tolist() for t,b in buckets.items()} for n,buckets in ap.items()}))
    metrics={n:dict(parameters=sum(p.numel() for p in m.parameters()),**{key+'_mean':sum(r['arms'][n][key] for r in records)/5794 for key in ('core_conv_flops','core_fp32_conv_flops','readout_fp32_reference_flops','logical_core_conv_flops')},cache_bytes=records[0]['arms'][n]['cache_bytes'],model_step_ms_mean=None,ap50=eval_utils.calculate_ap(ap[n],.5)[0],ap70=eval_utils.calculate_ap(ap[n],.7)[0]) for n,m in models.items()}
    write_json(out/'report.json',dict(status='COMPLETED',metrics=metrics,scans=5794,labels=1738,weights_unchanged=True,paired_task_states_exact=True,initial_control_raw_ap_exact=True,initial_integer_raw_ap_exact=True,trained=False,timing_valid=False));write_json(out/'status.json',dict(status='COMPLETED'))

def main():
    p=argparse.ArgumentParser()
    for n in ('root','checkpoint','audit','verification','trained','compact-verification','initialization','event-verification','calibration','reference-verification','integer-verification','integer-source','qat-verification','qat-source','quality-verification','quality-source','fast-verification','fast-source','learning-verification','learning-source','export-verification','export-source','full-reference-verification','full-reference-source','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();a.full=True;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'status.json',dict(status='RUNNING',phase='tests'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min full paired trained ANN/QAT validation')));signal.alarm(7200)
    try:worker(a,out)
    except Exception as exc:write_json(out/'status.json',dict(status='FAILED',error=repr(exc)));raise
if __name__=='__main__':main()
